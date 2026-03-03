#ifndef COUPMPM_TRANSFER_H
#define COUPMPM_TRANSFER_H

#include "coupmpm_grid.h"
#include "coupmpm_kernel.h"
#include "coupmpm_stress.h"
#include <mpi.h>
#include <vector>
#include <cstring>
#include <cmath>
#include <cassert>

namespace LAMMPS_NS {
namespace CoupMPM {

namespace Tags {
  constexpr int REVERSE_BASE = 8000;  // ghost → owner (sum)
  constexpr int FORWARD_BASE = 9000;  // owner → ghost (broadcast)
}

// ============================================================
// P2G contribution record for one particle (for migration)
//
// Stores the particle's FULL STATE at P2G time so we can
// exactly recompute and subtract its contributions (anti-P2G)
// when it migrates to another rank.
//
// Why store state instead of per-node deltas:
//   - B-spline quadratic: 27 nodes × 11 fields = 297 doubles
//   - Particle state: ~35 doubles regardless of kernel
//   - Exact same floating-point path ensures no drift
// ============================================================
struct P2GRecord {
  int local_idx;     // local atom index at P2G time
  int global_tag;    // global atom tag (robust against atom sorting)
  int body_id;       // molecule ID (for Bardenhagen)
  double xp[3];      // position at P2G time
  double vp[3];      // velocity
  double fp[3];      // force (pair + fix)
  double mp;         // mass
  double vol_p;      // current volume (vol0 * J)
  double sigma[9];   // full Cauchy stress tensor (3×3)
  double Cp[9];      // APIC affine matrix
};

// Anti-P2G: subtract a particle's P2G contributions from the grid.
// Called for particles that are about to migrate to another rank.
// Uses stored P2GRecord to exactly reproduce what was added, then subtracts.
//
// CRITICAL: This must use the EXACT same computation as p2g() to
// ensure floating-point reproducibility. Any divergence causes
// conservation drift that accumulates over time.
inline void anti_p2g(MPMGrid& grid,
                     const MPMKernel& kernel,
                     const P2GRecord& rec,
                     const double domain_lo[3],
                     bool use_bbar)
{
  const double dx_d[3] = {grid.dx, grid.dy, grid.dz};

  int ilo[3], ihi[3];
  if (!kernel.support_range(rec.xp, domain_lo, dx_d, grid.offset,
                            grid.ghost, grid.dim,
                            grid.gx, grid.gy, grid.gz,
                            ilo, ihi))
    return;

  for (int k = ilo[2]; k <= ihi[2]; k++) {
    for (int j = ilo[1]; j <= ihi[1]; j++) {
      for (int i = ilo[0]; i <= ihi[0]; i++) {
        double x_node[3];
        grid.node_position(i, j, k, domain_lo, x_node);

        double w, grad_w[3];
        kernel.evaluate(rec.xp, x_node, dx_d, grid.dim, w, grad_w);
        if (w < 1e-20) continue;

        const int n = grid.idx(i, j, k);
        const double dx_pn[3] = {
          x_node[0] - rec.xp[0],
          x_node[1] - rec.xp[1],
          x_node[2] - rec.xp[2]
        };

        // SUBTRACT mass
        grid.mass[n] -= w * rec.mp;

        // SUBTRACT APIC momentum
        {
          double va0 = rec.vp[0], va1 = rec.vp[1], va2 = rec.vp[2];
          for (int e = 0; e < 3; e++) {
            va0 += rec.Cp[0*3 + e] * dx_pn[e];
            va1 += rec.Cp[1*3 + e] * dx_pn[e];
            va2 += rec.Cp[2*3 + e] * dx_pn[e];
          }
          grid.momentum_x[n] -= w * rec.mp * va0;
          grid.momentum_y[n] -= w * rec.mp * va1;
          grid.momentum_z[n] -= w * rec.mp * va2;
        }

        // SUBTRACT internal force (note: p2g subtracts, so anti_p2g adds)
        {
          double fi_x = 0, fi_y = 0, fi_z = 0;
          for (int e = 0; e < 3; e++) {
            fi_x -= rec.vol_p * rec.sigma[0*3 + e] * grad_w[e];
            fi_y -= rec.vol_p * rec.sigma[1*3 + e] * grad_w[e];
            fi_z -= rec.vol_p * rec.sigma[2*3 + e] * grad_w[e];
          }
          grid.force_int_x[n] -= fi_x;
          grid.force_int_y[n] -= fi_y;
          grid.force_int_z[n] -= fi_z;
        }

        // SUBTRACT external force
        grid.force_ext_x[n] -= w * rec.fp[0];
        grid.force_ext_y[n] -= w * rec.fp[1];
        grid.force_ext_z[n] -= w * rec.fp[2];

        // SUBTRACT B-bar divergence contribution
        if (use_bbar) {
          double v_dot_grad = 0.0;
          for (int d = 0; d < 3; d++) {
            double va_d = rec.vp[d];
            for (int e = 0; e < 3; e++)
              va_d += rec.Cp[3*d + e] * dx_pn[e];
            v_dot_grad += va_d * grad_w[d];
          }
          grid.raw_div_v[n] -= rec.mp * v_dot_grad;
        }

        // SUBTRACT Bardenhagen per-body (if active)
        if (grid.contact_bardenhagen) {
          NodeBodyData* bd = grid.find_or_add_body(n, rec.body_id);
          if (bd) {
            bd->mass -= w * rec.mp;
            double va0 = rec.vp[0], va1 = rec.vp[1], va2 = rec.vp[2];
            for (int e = 0; e < 3; e++) {
              va0 += rec.Cp[e]     * dx_pn[e];
              va1 += rec.Cp[3 + e] * dx_pn[e];
              va2 += rec.Cp[6 + e] * dx_pn[e];
            }
            bd->momentum[0] -= w * rec.mp * va0;
            bd->momentum[1] -= w * rec.mp * va1;
            bd->momentum[2] -= w * rec.mp * va2;
          }
        }
      }
    }
  }
}

// ============================================================
// Ghost exchange for MPM grid fields
//
// Mirrors CoupLB Streaming class but exchanges MPM fields
// (mass, momentum, force) instead of distribution functions.
//
// Two communication phases:
//   reverse_comm(): ghost → owner, ADDITIVE (after P2G)
//   forward_comm(): owner → ghost, OVERWRITE (after grid solve)
//
// Adapted from couplb_streaming.h exchange pattern.
// ============================================================
class MPMGhostExchange {
public:
  int neigh[3][2];       // MPI neighbor ranks per dimension
  bool periodic[3];
  int nprocs[3];
  MPI_Comm comm;

  // Number of fields in reverse comm:
  // mass(1) + momentum(3) + force_int(3) + force_ext(3) + raw_div_v(1) = 11
  static constexpr int N_REVERSE_FIELDS = 11;
  // Number of fields in forward comm:
  // velocity_new(3) + div_v(1) = 4
  static constexpr int N_FORWARD_FIELDS = 4;

  std::vector<double> send_lo, send_hi, recv_lo, recv_hi;
  size_t max_buf;

  MPMGhostExchange() : comm(MPI_COMM_WORLD), max_buf(0) {
    for (int d = 0; d < 3; d++) {
      neigh[d][0] = neigh[d][1] = MPI_PROC_NULL;
      periodic[d] = false;
      nprocs[d] = 1;
    }
  }

  void set_comm(MPI_Comm c) { comm = c; }

  void set_neighbors(int pn[3][2]) {
    for (int d = 0; d < 3; d++) {
      neigh[d][0] = pn[d][0];
      neigh[d][1] = pn[d][1];
    }
  }

  void set_periodic(bool px, bool py, bool pz) {
    periodic[0] = px; periodic[1] = py; periodic[2] = pz;
  }

  void set_nprocs(int npx, int npy, int npz) {
    nprocs[0] = npx; nprocs[1] = npy; nprocs[2] = npz;
  }

  void allocate_buffers(const MPMGrid& grid) {
    // Max face size across all dimensions, times ghost width
    // Each ghost layer is one face; we exchange `ghost` layers.
    size_t mf = 0;
    const int nd = (grid.dim == 3) ? 3 : 2;
    for (int d = 0; d < nd; d++) {
      size_t fs = ghost_face_size(grid, d);
      if (fs > mf) mf = fs;
    }
    // Buffer for the larger of reverse and forward
    int max_fields = (N_REVERSE_FIELDS > N_FORWARD_FIELDS)
                     ? N_REVERSE_FIELDS : N_FORWARD_FIELDS;
    max_buf = static_cast<size_t>(max_fields) * mf;
    send_lo.resize(max_buf);
    send_hi.resize(max_buf);
    recv_lo.resize(max_buf);
    recv_hi.resize(max_buf);
  }

  // After P2G: sum ghost contributions back to owner
  void reverse_comm(MPMGrid& g) {
    reverse_dim(g, 0);
    reverse_dim(g, 1);
    if (g.dim == 3) reverse_dim(g, 2);
  }

  // After grid solve: broadcast updated velocities to ghost
  void forward_comm(MPMGrid& g) {
    forward_dim(g, 0);
    forward_dim(g, 1);
    if (g.dim == 3) forward_dim(g, 2);
  }

private:
  bool is_self_periodic(int d) const {
    return periodic[d] && nprocs[d] == 1;
  }

  // Size of one ghost face (all ghost layers in one direction)
  size_t ghost_face_size(const MPMGrid& g, int d) const {
    const int gh = g.ghost;
    if (d == 0) return static_cast<size_t>(gh) * g.gy * g.gz;
    if (d == 1) return static_cast<size_t>(g.gx) * gh * g.gz;
    return static_cast<size_t>(g.gx) * g.gy * gh;
  }

  // Iterate over ghost layer nodes on lo or hi side of dimension d.
  // side=0: lo ghost (indices 0..ghost-1 in dim d)
  // side=1: hi ghost (indices gd-ghost..gd-1 in dim d)
  // Also need "interior face" = the owned nodes adjacent to ghosts.
  // For reverse: pack ghost → send to owner, who adds to interior
  // For forward: pack interior face → send to neighbor's ghost

  // Reverse comm in one dimension: ghost_lo → neigh_lo, ghost_hi → neigh_hi
  // Neighbor accumulates into their interior face.
  void reverse_dim(MPMGrid& g, int d) {
    const int gd = (d == 0) ? g.gx : (d == 1) ? g.gy : g.gz;
    const int gh = g.ghost;

    if (is_self_periodic(d)) {
      // Self-periodic: accumulate ghost onto opposite interior
      accum_ghost_to_interior(g, d, 0);  // lo ghost → hi interior
      accum_ghost_to_interior(g, d, 1);  // hi ghost → lo interior
      return;
    }

    if (neigh[d][0] == MPI_PROC_NULL && neigh[d][1] == MPI_PROC_NULL) return;

    const size_t bs = static_cast<size_t>(N_REVERSE_FIELDS) * ghost_face_size(g, d);
    if (!bs) return;
    assert(bs <= max_buf);
    const int c = static_cast<int>(bs);

    // Pack ghost layers
    pack_reverse(g, d, 0, send_lo);  // lo ghost
    pack_reverse(g, d, 1, send_hi);  // hi ghost

    // Send: ghost_lo goes to neigh_lo, ghost_hi goes to neigh_hi
    // Recv: from neigh_hi into our hi interior, from neigh_lo into our lo interior
    MPI_Sendrecv(send_lo.data(), c, MPI_DOUBLE, neigh[d][0], Tags::REVERSE_BASE + d,
                 recv_hi.data(), c, MPI_DOUBLE, neigh[d][1], Tags::REVERSE_BASE + d,
                 comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(send_hi.data(), c, MPI_DOUBLE, neigh[d][1], Tags::REVERSE_BASE + 3 + d,
                 recv_lo.data(), c, MPI_DOUBLE, neigh[d][0], Tags::REVERSE_BASE + 3 + d,
                 comm, MPI_STATUS_IGNORE);

    // Accumulate received data into interior faces
    if (neigh[d][0] != MPI_PROC_NULL)
      unpack_reverse_accum(g, d, 0, recv_lo);  // add to lo interior face
    if (neigh[d][1] != MPI_PROC_NULL)
      unpack_reverse_accum(g, d, 1, recv_hi);  // add to hi interior face
  }

  // Forward comm in one dimension
  void forward_dim(MPMGrid& g, int d) {
    const int gd = (d == 0) ? g.gx : (d == 1) ? g.gy : g.gz;
    const int gh = g.ghost;

    if (is_self_periodic(d)) {
      copy_interior_to_ghost(g, d, 0);  // hi interior → lo ghost
      copy_interior_to_ghost(g, d, 1);  // lo interior → hi ghost
      return;
    }

    if (neigh[d][0] == MPI_PROC_NULL && neigh[d][1] == MPI_PROC_NULL) return;

    const size_t bs = static_cast<size_t>(N_FORWARD_FIELDS) * ghost_face_size(g, d);
    if (!bs) return;
    assert(bs <= max_buf);
    const int c = static_cast<int>(bs);

    // Pack interior faces
    pack_forward(g, d, 0, send_lo);  // lo interior face
    pack_forward(g, d, 1, send_hi);  // hi interior face

    // Send: lo interior goes to neigh_lo's hi ghost
    //       hi interior goes to neigh_hi's lo ghost
    MPI_Sendrecv(send_hi.data(), c, MPI_DOUBLE, neigh[d][1], Tags::FORWARD_BASE + d,
                 recv_lo.data(), c, MPI_DOUBLE, neigh[d][0], Tags::FORWARD_BASE + d,
                 comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(send_lo.data(), c, MPI_DOUBLE, neigh[d][0], Tags::FORWARD_BASE + 3 + d,
                 recv_hi.data(), c, MPI_DOUBLE, neigh[d][1], Tags::FORWARD_BASE + 3 + d,
                 comm, MPI_STATUS_IGNORE);

    if (neigh[d][0] != MPI_PROC_NULL)
      unpack_forward(g, d, 0, recv_lo);  // into lo ghost
    if (neigh[d][1] != MPI_PROC_NULL)
      unpack_forward(g, d, 1, recv_hi);  // into hi ghost
  }

  // --- Reverse pack/unpack ---

  // Pack ghost layer (side=0: lo, side=1: hi)
  void pack_reverse(const MPMGrid& g, int d, int side,
                    std::vector<double>& buf) const {
    size_t c = 0;
    iter_ghost(g, d, side, [&](int n) {
      buf[c++] = g.mass[n];
      buf[c++] = g.momentum_x[n];
      buf[c++] = g.momentum_y[n];
      buf[c++] = g.momentum_z[n];
      buf[c++] = g.force_int_x[n];
      buf[c++] = g.force_int_y[n];
      buf[c++] = g.force_int_z[n];
      buf[c++] = g.force_ext_x[n];
      buf[c++] = g.force_ext_y[n];
      buf[c++] = g.force_ext_z[n];
      buf[c++] = g.raw_div_v[n];
    });
  }

  // Unpack and ACCUMULATE into interior face
  void unpack_reverse_accum(MPMGrid& g, int d, int side,
                            const std::vector<double>& buf) {
    size_t c = 0;
    iter_interior_face(g, d, side, [&](int n) {
      g.mass[n]        += buf[c++];
      g.momentum_x[n]  += buf[c++];
      g.momentum_y[n]  += buf[c++];
      g.momentum_z[n]  += buf[c++];
      g.force_int_x[n] += buf[c++];
      g.force_int_y[n] += buf[c++];
      g.force_int_z[n] += buf[c++];
      g.force_ext_x[n] += buf[c++];
      g.force_ext_y[n] += buf[c++];
      g.force_ext_z[n] += buf[c++];
      g.raw_div_v[n]   += buf[c++];
    });
  }

  // Self-periodic: accumulate ghost directly into interior
  void accum_ghost_to_interior(MPMGrid& g, int d, int side) {
    // side=0: lo ghost → lo interior face (which is the neighbor's hi interior)
    // side=1: hi ghost → hi interior face
    // For self-periodic: lo ghost ← hi interior, so lo ghost should add to hi interior
    // Actually: in self-periodic, ghost_lo is a copy of hi_interior.
    // After P2G, ghost_lo has contributions that should be on hi_interior.
    // So we add ghost_lo → interior_hi and ghost_hi → interior_lo.
    iter_ghost_and_interior_periodic(g, d, side, [&](int gn, int in_) {
      g.mass[in_]        += g.mass[gn];
      g.momentum_x[in_]  += g.momentum_x[gn];
      g.momentum_y[in_]  += g.momentum_y[gn];
      g.momentum_z[in_]  += g.momentum_z[gn];
      g.force_int_x[in_] += g.force_int_x[gn];
      g.force_int_y[in_] += g.force_int_y[gn];
      g.force_int_z[in_] += g.force_int_z[gn];
      g.force_ext_x[in_] += g.force_ext_x[gn];
      g.force_ext_y[in_] += g.force_ext_y[gn];
      g.force_ext_z[in_] += g.force_ext_z[gn];
      g.raw_div_v[in_]   += g.raw_div_v[gn];
    });
  }

  // --- Forward pack/unpack ---

  // Pack interior face (side=0: lo, side=1: hi)
  void pack_forward(const MPMGrid& g, int d, int side,
                    std::vector<double>& buf) const {
    size_t c = 0;
    iter_interior_face(g, d, side, [&](int n) {
      buf[c++] = g.velocity_new_x[n];
      buf[c++] = g.velocity_new_y[n];
      buf[c++] = g.velocity_new_z[n];
      buf[c++] = g.div_v[n];
    });
  }

  // Unpack (overwrite) into ghost layer
  void unpack_forward(MPMGrid& g, int d, int side,
                      const std::vector<double>& buf) {
    size_t c = 0;
    iter_ghost(g, d, side, [&](int n) {
      g.velocity_new_x[n] = buf[c++];
      g.velocity_new_y[n] = buf[c++];
      g.velocity_new_z[n] = buf[c++];
      g.div_v[n]          = buf[c++];
    });
  }

  // Self-periodic: copy interior face to ghost
  void copy_interior_to_ghost(MPMGrid& g, int d, int side) {
    // side=0: copy hi interior → lo ghost
    // side=1: copy lo interior → hi ghost
    iter_ghost_and_interior_periodic(g, d, side, [&](int gn, int in_) {
      g.velocity_new_x[gn] = g.velocity_new_x[in_];
      g.velocity_new_y[gn] = g.velocity_new_y[in_];
      g.velocity_new_z[gn] = g.velocity_new_z[in_];
      g.div_v[gn]          = g.div_v[in_];
    });
  }

  // --- Iteration helpers ---

  // Iterate over ghost layer nodes for dimension d, side (0=lo, 1=hi)
  // Ghost consists of `ghost` layers of nodes.
  template <typename Fn>
  void iter_ghost(const MPMGrid& g, int d, int side, Fn fn) const {
    const int gh = g.ghost;
    if (d == 0) {
      const int i0 = (side == 0) ? 0 : g.gx - gh;
      for (int k = 0; k < g.gz; k++)
        for (int j = 0; j < g.gy; j++)
          for (int i = i0; i < i0 + gh; i++)
            fn(g.idx(i, j, k));
    } else if (d == 1) {
      const int j0 = (side == 0) ? 0 : g.gy - gh;
      for (int k = 0; k < g.gz; k++)
        for (int j = j0; j < j0 + gh; j++)
          for (int i = 0; i < g.gx; i++)
            fn(g.idx(i, j, k));
    } else {
      const int k0 = (side == 0) ? 0 : g.gz - gh;
      for (int k = k0; k < k0 + gh; k++)
        for (int j = 0; j < g.gy; j++)
          for (int i = 0; i < g.gx; i++)
            fn(g.idx(i, j, k));
    }
  }

  // Interior face adjacent to ghost (the first/last `ghost` owned layers)
  template <typename Fn>
  void iter_interior_face(const MPMGrid& g, int d, int side, Fn fn) const {
    const int gh = g.ghost;
    if (d == 0) {
      const int i0 = (side == 0) ? gh : g.gx - 2 * gh;
      for (int k = 0; k < g.gz; k++)
        for (int j = 0; j < g.gy; j++)
          for (int i = i0; i < i0 + gh; i++)
            fn(g.idx(i, j, k));
    } else if (d == 1) {
      const int j0 = (side == 0) ? gh : g.gy - 2 * gh;
      for (int k = 0; k < g.gz; k++)
        for (int j = j0; j < j0 + gh; j++)
          for (int i = 0; i < g.gx; i++)
            fn(g.idx(i, j, k));
    } else {
      const int k0 = (side == 0) ? gh : g.gz - 2 * gh;
      for (int k = k0; k < k0 + gh; k++)
        for (int j = 0; j < g.gy; j++)
          for (int i = 0; i < g.gx; i++)
            fn(g.idx(i, j, k));
    }
  }

  // For self-periodic: iterate ghost-interior pairs
  // side=0: lo ghost ↔ hi interior (the ghost wraps around)
  // side=1: hi ghost ↔ lo interior
  template <typename Fn>
  void iter_ghost_and_interior_periodic(const MPMGrid& g, int d, int side, Fn fn) const {
    const int gh = g.ghost;
    if (d == 0) {
      for (int layer = 0; layer < gh; layer++) {
        int gi, ii;
        if (side == 0) {
          gi = layer;                // lo ghost
          ii = g.gx - 2 * gh + layer;  // hi interior
        } else {
          gi = g.gx - gh + layer;   // hi ghost
          ii = gh + layer;           // lo interior
        }
        for (int k = 0; k < g.gz; k++)
          for (int j = 0; j < g.gy; j++)
            fn(g.idx(gi, j, k), g.idx(ii, j, k));
      }
    } else if (d == 1) {
      for (int layer = 0; layer < gh; layer++) {
        int gj, ij;
        if (side == 0) {
          gj = layer;
          ij = g.gy - 2 * gh + layer;
        } else {
          gj = g.gy - gh + layer;
          ij = gh + layer;
        }
        for (int k = 0; k < g.gz; k++)
          for (int i = 0; i < g.gx; i++)
            fn(g.idx(i, gj, k), g.idx(i, ij, k));
      }
    } else {
      for (int layer = 0; layer < gh; layer++) {
        int gk, ik;
        if (side == 0) {
          gk = layer;
          ik = g.gz - 2 * gh + layer;
        } else {
          gk = g.gz - gh + layer;
          ik = gh + layer;
        }
        for (int j = 0; j < g.gy; j++)
          for (int i = 0; i < g.gx; i++)
            fn(g.idx(i, j, gk), g.idx(i, j, ik));
      }
    }
  }
};

// ============================================================
// P2G and G2P operations
//
// These are free functions that operate on the grid and
// particle data from LAMMPS atom arrays.
// ============================================================

// P2G: Transfer particle data to grid.
// Particle arrays are raw pointers from LAMMPS atom.
// This function accumulates; grid must be zeroed before calling.
//
// If records != nullptr, fills a P2GRecord for each particle
// (used later for anti-P2G during migration).
//
// Returns number of particles processed.
inline int p2g(MPMGrid& grid,
               const MPMKernel& kernel,
               int nlocal,
               // Particle arrays (LAMMPS atom pointers)
               double** x,          // positions [nlocal][3]
               double** v,          // velocities [nlocal][3]
               double** f,          // forces (pair + fix) [nlocal][3]
               double* mass_p,      // particle mass [nlocal]
               double* vol0,        // reference volume [nlocal]
               double* F_def,       // deformation gradient [nlocal * 9]
               double* stress_v,    // Cauchy stress Voigt [nlocal * 6]
               double* Bp,          // APIC affine matrix [nlocal * 9]
               int* molecule,       // body ID [nlocal] (for Bardenhagen)
               const double domain_lo[3],
               bool use_bbar,
               std::vector<P2GRecord>* records = nullptr,
               int* tags = nullptr)  // global atom tags for record robustness
{
  const double dx_d[3] = {grid.dx, grid.dy, grid.dz};
  double Dinv[3];
  kernel.D_inverse(const_cast<double*>(dx_d), grid.dim, Dinv);

  int count = 0;
  for (int p = 0; p < nlocal; p++) {
    const double xp[3] = {x[p][0], x[p][1], x[p][2]};
    const double vp[3] = {v[p][0], v[p][1], v[p][2]};
    const double fp[3] = {f[p][0], f[p][1], f[p][2]};
    const double mp = mass_p[p];
    const double* Fp = &F_def[p * 9];
    const double* sp = &stress_v[p * 6];
    const double* Cp = &Bp[p * 9]; // APIC affine matrix (C_p = Bp * Dinv already applied? No, Bp IS the affine matrix)
    const double J = Mat3::det(Fp);
    const double vol_p = vol0[p] * J;

    // Full Cauchy stress tensor from Voigt
    // [xx, yy, zz, xy, xz, yz] → 3x3
    double sigma[9];
    sigma[0] = sp[0]; sigma[1] = sp[3]; sigma[2] = sp[4];
    sigma[3] = sp[3]; sigma[4] = sp[1]; sigma[5] = sp[5];
    sigma[6] = sp[4]; sigma[7] = sp[5]; sigma[8] = sp[2];

    // Store record for anti-P2G migration protocol
    if (records) {
      P2GRecord rec;
      rec.local_idx = p;
      rec.global_tag = tags ? tags[p] : -1;
      rec.body_id = molecule ? molecule[p] : 0;
      std::memcpy(rec.xp, xp, 3 * sizeof(double));
      std::memcpy(rec.vp, vp, 3 * sizeof(double));
      std::memcpy(rec.fp, fp, 3 * sizeof(double));
      rec.mp = mp;
      rec.vol_p = vol_p;
      std::memcpy(rec.sigma, sigma, 9 * sizeof(double));
      std::memcpy(rec.Cp, Cp, 9 * sizeof(double));
      records->push_back(rec);
    }

    // Find support range
    int ilo[3], ihi[3];
    if (!kernel.support_range(xp, domain_lo, dx_d, grid.offset,
                              grid.ghost, grid.dim,
                              grid.gx, grid.gy, grid.gz,
                              ilo, ihi))
      continue;

    for (int k = ilo[2]; k <= ihi[2]; k++) {
      for (int j = ilo[1]; j <= ihi[1]; j++) {
        for (int i = ilo[0]; i <= ihi[0]; i++) {
          double x_node[3];
          grid.node_position(i, j, k, domain_lo, x_node);

          double w, grad_w[3];
          kernel.evaluate(xp, x_node, dx_d, grid.dim, w, grad_w);

          if (w < 1e-20) continue;

          const int n = grid.idx(i, j, k);
          const double dx_pn[3] = {
            x_node[0] - xp[0],
            x_node[1] - xp[1],
            x_node[2] - xp[2]
          };

          // Mass
          grid.mass[n] += w * mp;

          // APIC momentum: m_p * w * (v_p + C_p · (x_node - x_p))
          {
            double v_apic_0 = vp[0], v_apic_1 = vp[1], v_apic_2 = vp[2];
            for (int e = 0; e < 3; e++) {
              v_apic_0 += Cp[0*3 + e] * dx_pn[e];
              v_apic_1 += Cp[1*3 + e] * dx_pn[e];
              v_apic_2 += Cp[2*3 + e] * dx_pn[e];
            }
            grid.momentum_x[n] += w * mp * v_apic_0;
            grid.momentum_y[n] += w * mp * v_apic_1;
            grid.momentum_z[n] += w * mp * v_apic_2;
          }

          // Internal force: -vol_p * σ · ∇w
          {
            double fi_x = 0, fi_y = 0, fi_z = 0;
            for (int e = 0; e < 3; e++) {
              fi_x -= vol_p * sigma[0*3 + e] * grad_w[e];
              fi_y -= vol_p * sigma[1*3 + e] * grad_w[e];
              fi_z -= vol_p * sigma[2*3 + e] * grad_w[e];
            }
            grid.force_int_x[n] += fi_x;
            grid.force_int_y[n] += fi_y;
            grid.force_int_z[n] += fi_z;
          }

          // External force (pair + fix): w * f_p
          grid.force_ext_x[n] += w * fp[0];
          grid.force_ext_y[n] += w * fp[1];
          grid.force_ext_z[n] += w * fp[2];

          // B-bar: accumulate mass-weighted velocity divergence
          if (use_bbar) {
            double v_dot_grad = 0.0;
            for (int d = 0; d < 3; d++) {
              double v_apic_d = vp[d];
              for (int e = 0; e < 3; e++)
                v_apic_d += Cp[3*d + e] * dx_pn[e];
              v_dot_grad += v_apic_d * grad_w[d];
            }
            grid.raw_div_v[n] += mp * v_dot_grad;
          }

          // Bardenhagen: per-body accumulation
          if (grid.contact_bardenhagen && molecule) {
            NodeBodyData* bd = grid.find_or_add_body(n, molecule[p]);
            if (bd) {
              bd->mass += w * mp;
              double v_apic_0 = vp[0], v_apic_1 = vp[1], v_apic_2 = vp[2];
              for (int e = 0; e < 3; e++) {
                v_apic_0 += Cp[e] * dx_pn[e];
                v_apic_1 += Cp[3 + e] * dx_pn[e];
                v_apic_2 += Cp[6 + e] * dx_pn[e];
              }
              bd->momentum[0] += w * mp * v_apic_0;
              bd->momentum[1] += w * mp * v_apic_1;
              bd->momentum[2] += w * mp * v_apic_2;
            }
          }
        }
      }
    }
    count++;
  }
  return count;
}

// G2P: Transfer grid data back to particles.
// Updates velocity, Bp, velocity gradient, position.
// Does NOT update F or stress (caller does that with B-bar correction).
//
// Output arrays:
//   v_new[nlocal][3]:  updated velocity
//   Bp_new[nlocal*9]:  updated APIC affine matrix
//   L_out[nlocal*9]:   velocity gradient (for F update)
//   div_v_smooth[nlocal]: smoothed divergence (for B-bar)
inline void g2p(const MPMGrid& grid,
                const MPMKernel& kernel,
                int nlocal,
                double** x,
                double** v,            // will be updated
                double* Bp,            // will be updated [nlocal*9]
                double* L_out,         // output: velocity gradient [nlocal*9]
                double* div_v_smooth,  // output: smoothed div_v [nlocal]
                const double domain_lo[3],
                double dt,
                bool use_bbar)
{
  const double dx_d[3] = {grid.dx, grid.dy, grid.dz};
  double Dinv[3];
  kernel.D_inverse(const_cast<double*>(dx_d), grid.dim, Dinv);

  for (int p = 0; p < nlocal; p++) {
    const double xp[3] = {x[p][0], x[p][1], x[p][2]};
    double v_new[3] = {0.0, 0.0, 0.0};
    double Bp_new[9];
    double L[9];
    std::memset(Bp_new, 0, sizeof(Bp_new));
    std::memset(L, 0, sizeof(L));
    double dv_smooth = 0.0;

    int ilo[3], ihi[3];
    if (!kernel.support_range(xp, domain_lo, dx_d, grid.offset,
                              grid.ghost, grid.dim,
                              grid.gx, grid.gy, grid.gz,
                              ilo, ihi)) {
      // Particle outside grid — zero everything
      v[p][0] = v[p][1] = v[p][2] = 0.0;
      std::memset(&Bp[p*9], 0, 9 * sizeof(double));
      std::memset(&L_out[p*9], 0, 9 * sizeof(double));
      if (div_v_smooth) div_v_smooth[p] = 0.0;
      continue;
    }

    for (int k = ilo[2]; k <= ihi[2]; k++) {
      for (int j = ilo[1]; j <= ihi[1]; j++) {
        for (int i = ilo[0]; i <= ihi[0]; i++) {
          double x_node[3];
          grid.node_position(i, j, k, domain_lo, x_node);

          double w, grad_w[3];
          kernel.evaluate(xp, x_node, dx_d, grid.dim, w, grad_w);
          if (w < 1e-20) continue;

          const int n = grid.idx(i, j, k);
          const double vn[3] = {
            grid.velocity_new_x[n],
            grid.velocity_new_y[n],
            grid.velocity_new_z[n]
          };

          const double dx_pn[3] = {
            x_node[0] - xp[0],
            x_node[1] - xp[1],
            x_node[2] - xp[2]
          };

          // Velocity
          for (int d = 0; d < 3; d++)
            v_new[d] += w * vn[d];

          // APIC affine matrix: Bp = Σ w * v_node ⊗ dx * Dinv
          for (int d = 0; d < 3; d++)
            for (int e = 0; e < 3; e++)
              Bp_new[3*d + e] += w * vn[d] * dx_pn[e] * Dinv[e];

          // Velocity gradient: L = Σ v_node ⊗ ∇w
          for (int d = 0; d < 3; d++)
            for (int e = 0; e < 3; e++)
              L[3*d + e] += vn[d] * grad_w[e];

          // B-bar smoothed divergence
          if (use_bbar)
            dv_smooth += w * grid.div_v[n];
        }
      }
    }

    // Store results
    v[p][0] = v_new[0];
    v[p][1] = v_new[1];
    v[p][2] = v_new[2];
    std::memcpy(&Bp[p*9], Bp_new, 9 * sizeof(double));
    std::memcpy(&L_out[p*9], L, 9 * sizeof(double));
    if (div_v_smooth) div_v_smooth[p] = dv_smooth;
  }
}

// Update deformation gradient with B-bar correction and compute new stress.
// Called after G2P.
inline void update_F_and_stress(
    int nlocal,
    double* F_def,          // [nlocal*9], updated in-place
    double* stress_v,       // [nlocal*6], updated in-place
    const double* L_raw,    // [nlocal*9], from G2P
    const double* div_v_s,  // [nlocal], smoothed div_v (nullptr if no B-bar)
    double* state,          // [nlocal*N_STATE], state vars
    MPMStress& constitutive,
    double dt,
    bool use_bbar)
{
  const int nstate = constitutive.n_state_vars();

  for (int p = 0; p < nlocal; p++) {
    double* Fp = &F_def[p * 9];
    double* sp = &stress_v[p * 6];
    const double* Lp = &L_raw[p * 9];
    double* st = state ? &state[p * nstate] : nullptr;

    // B-bar correction
    double L_bar[9];
    std::memcpy(L_bar, Lp, 9 * sizeof(double));

    if (use_bbar && div_v_s) {
      double tr_L = Lp[0] + Lp[4] + Lp[8];
      double correction = (div_v_s[p] - tr_L) / 3.0;
      L_bar[0] += correction;
      L_bar[4] += correction;
      L_bar[8] += correction;
    }

    // F_new = (I + dt * L_bar) * F_old
    double F_new[9];
    Mat3::update_F(Fp, L_bar, dt, F_new);
    std::memcpy(Fp, F_new, 9 * sizeof(double));

    // Compute stress
    double J = Mat3::det(F_new);
    constitutive.compute_stress(F_new, J, st, dt, sp);
  }
}

} // namespace CoupMPM
} // namespace LAMMPS_NS

#endif // COUPMPM_TRANSFER_H
