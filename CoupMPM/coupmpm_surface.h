#ifndef COUPMPM_SURFACE_H
#define COUPMPM_SURFACE_H

#include "coupmpm_grid.h"
#include "coupmpm_kernel.h"
#include <mpi.h>
#include <cmath>
#include <cstring>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace LAMMPS_NS {
namespace CoupMPM {

// ============================================================
// Surface detection via mass-density gradient ∇ρ
//
// After P2G + reverse comm, the grid has nodal mass.
// 1. Compute density ρ = mass / cell_volume at each node
// 2. Compute ∇ρ at nodes via central finite differences
// 3. Communicate ∇ρ across ghost layers (forward comm)
// 4. Interpolate |∇ρ| to particles using shape functions
// 5. Threshold: surface if |∇ρ| > α * ρ_max / dx
//
// This replaces noisy neighbor-count surface detection.
// ============================================================

class SurfaceDetector {
public:
  double alpha;    // threshold parameter (default 0.1)

  // Gradient fields on grid (allocated on first use)
  std::vector<double> grad_rho_x, grad_rho_y, grad_rho_z;
  bool allocated;

  SurfaceDetector() : alpha(0.1), allocated(false) {}
  explicit SurfaceDetector(double a) : alpha(a), allocated(false) {}

  void allocate(int ntotal) {
    grad_rho_x.assign(ntotal, 0.0);
    grad_rho_y.assign(ntotal, 0.0);
    grad_rho_z.assign(ntotal, 0.0);
    allocated = true;
  }

  // Step 1 & 2: Compute density and its gradient on the grid.
  // Call AFTER P2G + reverse_comm (so grid.mass is complete).
  // The grid.density field is also populated here.
  void compute_grid_gradient(MPMGrid& grid) {
    if (!allocated) allocate(grid.ntotal);

    const double cv = grid.cell_volume();
    if (cv < 1e-30) return;
    const double inv_cv = 1.0 / cv;

    // Compute density at all nodes (including ghost)
    for (int n = 0; n < grid.ntotal; n++)
      grid.density[n] = grid.mass[n] * inv_cv;

    // Zero gradient fields
    std::memset(grad_rho_x.data(), 0, grid.ntotal * sizeof(double));
    std::memset(grad_rho_y.data(), 0, grid.ntotal * sizeof(double));
    std::memset(grad_rho_z.data(), 0, grid.ntotal * sizeof(double));

    // Central finite differences on owned + one layer of ghost
    // (we need ghost values for the stencil; they're available
    // because density is computed from mass which includes ghosts)
    const int gh = grid.ghost;
    const int klo = (grid.dim == 3) ? 1 : 0;
    const int khi = (grid.dim == 3) ? (grid.gz - 2) : 0;

    const double inv_2dx = 0.5 / grid.dx;
    const double inv_2dy = 0.5 / grid.dy;
    const double inv_2dz = (grid.dim == 3) ? (0.5 / grid.dz) : 0.0;

#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int k = klo; k <= khi; k++) {
      for (int j = 1; j <= grid.gy - 2; j++) {
        for (int i = 1; i <= grid.gx - 2; i++) {
          const int n = grid.idx(i, j, k);

          // ∂ρ/∂x via central difference
          grad_rho_x[n] = (grid.density[grid.idx(i+1, j, k)]
                         - grid.density[grid.idx(i-1, j, k)]) * inv_2dx;

          // ∂ρ/∂y
          grad_rho_y[n] = (grid.density[grid.idx(i, j+1, k)]
                         - grid.density[grid.idx(i, j-1, k)]) * inv_2dy;

          // ∂ρ/∂z (3D only)
          if (grid.dim == 3) {
            grad_rho_z[n] = (grid.density[grid.idx(i, j, k+1)]
                           - grid.density[grid.idx(i, j, k-1)]) * inv_2dz;
          }
        }
      }
    }
  }

  // Step 3 & 4 & 5: Interpolate |∇ρ| to particles, apply threshold.
  // Updates the surface flag for each particle.
  //
  // Requires: compute_grid_gradient() already called.
  // For multi-rank: call sync_gradients() to forward-communicate
  // grad_rho to ghost nodes before this function.
  //
  // comm_world: MPI communicator for global rho_max reduction.
  //             Pass MPI_COMM_NULL for single-rank.
  void detect_surface(const MPMGrid& grid,
                      const MPMKernel& kernel,
                      int nlocal,
                      double** x,
                      int* surface_flag,
                      const double domain_lo[3],
                      MPI_Comm comm_world = MPI_COMM_NULL)
  {
    const double dx_d[3] = {grid.dx, grid.dy, grid.dz};

    // Find ρ_max across the grid for threshold computation
    double local_rho_max = 0.0;
    const int gh = grid.ghost;
    const int klo = (grid.dim == 3) ? gh : 0;
    const int khi = (grid.dim == 3) ? (grid.gz - gh - 1) : 0;

    for (int k = klo; k <= khi; k++)
      for (int j = gh; j <= grid.gy - gh - 1; j++)
        for (int i = gh; i <= grid.gx - gh - 1; i++) {
          double rho_n = grid.density[grid.idx(i, j, k)];
          if (rho_n > local_rho_max) local_rho_max = rho_n;
        }

    // Threshold: |∇ρ| > α * ρ_max / dx_min
    const double dx_min = std::min(grid.dx, std::min(grid.dy,
                          (grid.dim == 3) ? grid.dz : grid.dy));

    // Global rho_max across all ranks for consistent threshold
    double global_rho_max = local_rho_max;
    if (comm_world != MPI_COMM_NULL)
      MPI_Allreduce(&local_rho_max, &global_rho_max, 1, MPI_DOUBLE,
                    MPI_MAX, comm_world);

    const double threshold = alpha * global_rho_max / dx_min;

    // Skip if everything is zero (no mass on grid)
    if (global_rho_max < MASS_TOL) {
      for (int p = 0; p < nlocal; p++) surface_flag[p] = 0;
      return;
    }

    // Interpolate |∇ρ| to each particle
    for (int p = 0; p < nlocal; p++) {
      const double xp[3] = {x[p][0], x[p][1], x[p][2]};

      int ilo[3], ihi[3];
      if (!kernel.support_range(xp, domain_lo, dx_d, grid.offset,
                                grid.ghost, grid.dim,
                                grid.gx, grid.gy, grid.gz,
                                ilo, ihi)) {
        surface_flag[p] = 0;
        continue;
      }

      double gx = 0.0, gy_val = 0.0, gz_val = 0.0;

      for (int k = ilo[2]; k <= ihi[2]; k++) {
        for (int j = ilo[1]; j <= ihi[1]; j++) {
          for (int i = ilo[0]; i <= ihi[0]; i++) {
            double x_node[3];
            grid.node_position(i, j, k, domain_lo, x_node);

            double w, grad_w[3];
            kernel.evaluate(xp, x_node, dx_d, grid.dim, w, grad_w);
            if (w < 1e-20) continue;

            const int n = grid.idx(i, j, k);
            gx     += w * grad_rho_x[n];
            gy_val += w * grad_rho_y[n];
            gz_val += w * grad_rho_z[n];
          }
        }
      }

      double grad_mag = std::sqrt(gx*gx + gy_val*gy_val + gz_val*gz_val);
      surface_flag[p] = (grad_mag > threshold) ? 1 : 0;
    }
  }

  // Optional: update area_scale from Nanson's formula.
  // For each surface particle with area0 != 0:
  //   da = J * F^{-T} · area0
  //   area_scale = |da| / |area0|
  //
  // This is only needed if using Nanson-scaled penalty contact.
  static void update_area_scale(int nlocal,
                                const double* F_def,  // [nlocal*9]
                                const int* surface_flag,
                                double** area0,
                                double* area_scale)
  {
    for (int p = 0; p < nlocal; p++) {
      if (!surface_flag[p]) {
        area_scale[p] = 0.0;
        continue;
      }

      const double* F = &F_def[p * 9];
      const double a0[3] = {area0[p][0], area0[p][1], area0[p][2]};
      double a0_mag = std::sqrt(a0[0]*a0[0] + a0[1]*a0[1] + a0[2]*a0[2]);
      if (a0_mag < 1e-20) {
        area_scale[p] = 1.0;
        continue;
      }

      double J = Mat3::det(F);
      if (std::fabs(J) < 1e-20) {
        area_scale[p] = 0.0;
        continue;
      }

      // F^{-T} = cofactor(F) / J = adj(F)^T / J
      // For Nanson: da = J * F^{-T} · dA = cofactor(F)^T · dA
      // cofactor(F)_ij = (-1)^(i+j) * minor_ij
      // Easier: da_i = J * (F_inv)_ji * dA_j
      //
      // Compute F inverse via cofactors
      double Finv[9];
      double inv_J = 1.0 / J;
      Finv[0] = inv_J * (F[4]*F[8] - F[5]*F[7]);
      Finv[1] = inv_J * (F[2]*F[7] - F[1]*F[8]);
      Finv[2] = inv_J * (F[1]*F[5] - F[2]*F[4]);
      Finv[3] = inv_J * (F[5]*F[6] - F[3]*F[8]);
      Finv[4] = inv_J * (F[0]*F[8] - F[2]*F[6]);
      Finv[5] = inv_J * (F[2]*F[3] - F[0]*F[5]);
      Finv[6] = inv_J * (F[3]*F[7] - F[4]*F[6]);
      Finv[7] = inv_J * (F[1]*F[6] - F[0]*F[7]);
      Finv[8] = inv_J * (F[0]*F[4] - F[1]*F[3]);

      // da = J * F^{-T} · dA  →  da_i = J * Σ_j Finv[j][i] * a0_j
      //                                 = J * Σ_j Finv[3*j + i] * a0_j
      double da[3] = {0, 0, 0};
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
          da[i] += J * Finv[3*j + i] * a0[j];

      double da_mag = std::sqrt(da[0]*da[0] + da[1]*da[1] + da[2]*da[2]);
      area_scale[p] = da_mag / a0_mag;
    }
  }
};

} // namespace CoupMPM
} // namespace LAMMPS_NS

#endif // COUPMPM_SURFACE_H
