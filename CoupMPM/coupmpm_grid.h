#ifndef COUPMPM_GRID_H
#define COUPMPM_GRID_H

#include "lmptype.h"
#include <vector>
#include <cstring>
#include <cmath>
#include <cassert>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace LAMMPS_NS {
namespace CoupMPM {

// Maximum bodies tracked per node for Bardenhagen contact.
// Most nodes see 1 body; interface nodes see 2-3.
// If exceeded, extra bodies are silently dropped (with warning).
static constexpr int MAX_BODIES_PER_NODE = 4;

static constexpr double MASS_TOL = 1e-20;

// Per-body data on a single grid node (for multi-velocity contact)
struct NodeBodyData {
  tagint body_id;
  double mass;
  double momentum[3];
  double velocity[3];
  double velocity_new[3];
  double com[3];   // mass-weighted center-of-mass accumulator (sum of m*x)

  void zero() {
    body_id = -1;
    mass = 0.0;
    std::memset(momentum, 0, sizeof(momentum));
    std::memset(velocity, 0, sizeof(velocity));
    std::memset(velocity_new, 0, sizeof(velocity_new));
    std::memset(com, 0, sizeof(com));
  }
};

// ============================================================
// MPMGrid — local processor subdomain with ghost layers
//
// SoA layout for hot fields. Ghost-inclusive indexing mirrors
// CoupLB: gx=nx+2*ghost, gy=ny+2*ghost, gz=nz+2*ghost
// where ghost = kernel support radius.
//
// Unlike CoupLB which has fixed 1-cell ghost, MPM needs
// variable ghost width depending on kernel:
//   linear: 1, bspline2: 2, bspline3: 3
// ============================================================
class MPMGrid {
public:
  int dim;             // 2 or 3
  int nx, ny, nz;      // owned cells per dimension
  int ghost;           // ghost layer width (= kernel support)
  int gx, gy, gz;      // ghost-inclusive sizes
  int ntotal;           // gx * gy * gz
  double dx, dy, dz;   // grid spacing (physical units)
  int offset[3];        // global offset of this subdomain
  int Nx, Ny, Nz;       // global grid dimensions

  // --- Primary SoA fields (always allocated) ---
  std::vector<double> mass;          // total mass
  std::vector<double> momentum_x, momentum_y, momentum_z;
  std::vector<double> force_int_x, force_int_y, force_int_z;  // stress divergence
  std::vector<double> force_ext_x, force_ext_y, force_ext_z;  // pair + fix forces
  std::vector<double> velocity_x, velocity_y, velocity_z;      // pre-update
  std::vector<double> velocity_new_x, velocity_new_y, velocity_new_z; // post-update
  std::vector<double> div_v;          // velocity divergence (for B-bar)
  std::vector<double> raw_div_v;      // unnormalized div_v (mass-weighted, for two-pass B-bar)
  std::vector<double> density;        // mass density = mass / cell_volume (for surface detection)

  // --- Per-body fields (allocated only when Bardenhagen contact active) ---
  bool contact_bardenhagen;
  std::vector<NodeBodyData> body_data; // size = ntotal * MAX_BODIES_PER_NODE
  std::vector<int> num_bodies;         // size = ntotal

  MPMGrid()
    : dim(3), nx(0), ny(0), nz(0), ghost(2),
      gx(0), gy(0), gz(0), ntotal(0),
      dx(1.0), dy(1.0), dz(1.0),
      Nx(0), Ny(0), Nz(0),
      contact_bardenhagen(false) {
    offset[0] = offset[1] = offset[2] = 0;
  }

  void allocate(int dim_, int nx_, int ny_, int nz_,
                double dx_, double dy_, double dz_,
                int ghost_,
                int ox, int oy, int oz,
                int Nx_, int Ny_, int Nz_,
                bool bardenhagen = false) {
    dim = dim_;
    nx = nx_; ny = ny_; nz = nz_;
    dx = dx_; dy = dy_; dz = dz_;
    ghost = ghost_;
    offset[0] = ox; offset[1] = oy; offset[2] = oz;
    Nx = Nx_; Ny = Ny_; Nz = Nz_;

    gx = nx + 2 * ghost;
    gy = ny + 2 * ghost;
    gz = (dim == 3) ? (nz + 2 * ghost) : 1;
    ntotal = (int)((long long)gx * gy * gz);

    // Allocate primary fields
    mass.assign(ntotal, 0.0);
    momentum_x.assign(ntotal, 0.0);
    momentum_y.assign(ntotal, 0.0);
    momentum_z.assign(ntotal, 0.0);
    force_int_x.assign(ntotal, 0.0);
    force_int_y.assign(ntotal, 0.0);
    force_int_z.assign(ntotal, 0.0);
    force_ext_x.assign(ntotal, 0.0);
    force_ext_y.assign(ntotal, 0.0);
    force_ext_z.assign(ntotal, 0.0);
    velocity_x.assign(ntotal, 0.0);
    velocity_y.assign(ntotal, 0.0);
    velocity_z.assign(ntotal, 0.0);
    velocity_new_x.assign(ntotal, 0.0);
    velocity_new_y.assign(ntotal, 0.0);
    velocity_new_z.assign(ntotal, 0.0);
    div_v.assign(ntotal, 0.0);
    raw_div_v.assign(ntotal, 0.0);
    density.assign(ntotal, 0.0);

    // Per-body fields
    contact_bardenhagen = bardenhagen;
    if (contact_bardenhagen) {
      body_data.resize(ntotal * MAX_BODIES_PER_NODE);
      for (auto& bd : body_data) bd.zero();
      num_bodies.assign(ntotal, 0);
    }
  }

  // Ghost-inclusive index (like CoupLB idx)
  inline int idx(int i, int j, int k = 0) const {
    assert(i >= 0 && i < gx);
    assert(j >= 0 && j < gy);
    assert(k >= 0 && k < gz);
    return i + gx * (j + gy * k);
  }

  // Local (owned) index → ghost-inclusive index
  // Local indices run 0..nx-1, 0..ny-1, 0..nz-1
  inline int lidx(int li, int lj, int lk = 0) const {
    return idx(li + ghost, lj + ghost, (dim == 3) ? (lk + ghost) : 0);
  }

  // Physical position of grid node (ghost-inclusive index)
  inline void node_position(int i, int j, int k,
                            const double domain_lo[3],
                            double pos[3]) const {
    pos[0] = domain_lo[0] + (offset[0] + i - ghost) * dx;
    pos[1] = domain_lo[1] + (offset[1] + j - ghost) * dy;
    pos[2] = (dim == 3) ? (domain_lo[2] + (offset[2] + k - ghost) * dz) : 0.0;
  }

  // Cell volume
  inline double cell_volume() const {
    return dx * dy * ((dim == 3) ? dz : 1.0);
  }

  // Zero all grid fields for a new timestep
  void zero_grid() {
    std::memset(mass.data(), 0, ntotal * sizeof(double));
    std::memset(momentum_x.data(), 0, ntotal * sizeof(double));
    std::memset(momentum_y.data(), 0, ntotal * sizeof(double));
    std::memset(momentum_z.data(), 0, ntotal * sizeof(double));
    std::memset(force_int_x.data(), 0, ntotal * sizeof(double));
    std::memset(force_int_y.data(), 0, ntotal * sizeof(double));
    std::memset(force_int_z.data(), 0, ntotal * sizeof(double));
    std::memset(force_ext_x.data(), 0, ntotal * sizeof(double));
    std::memset(force_ext_y.data(), 0, ntotal * sizeof(double));
    std::memset(force_ext_z.data(), 0, ntotal * sizeof(double));
    std::memset(velocity_x.data(), 0, ntotal * sizeof(double));
    std::memset(velocity_y.data(), 0, ntotal * sizeof(double));
    std::memset(velocity_z.data(), 0, ntotal * sizeof(double));
    std::memset(velocity_new_x.data(), 0, ntotal * sizeof(double));
    std::memset(velocity_new_y.data(), 0, ntotal * sizeof(double));
    std::memset(velocity_new_z.data(), 0, ntotal * sizeof(double));
    std::memset(div_v.data(), 0, ntotal * sizeof(double));
    std::memset(raw_div_v.data(), 0, ntotal * sizeof(double));
    std::memset(density.data(), 0, ntotal * sizeof(double));

    if (contact_bardenhagen) {
      for (auto& bd : body_data) bd.zero();
      std::memset(num_bodies.data(), 0, ntotal * sizeof(int));
    }
  }

  // Grid momentum solve: v_new = (momentum + dt * (f_int + f_ext)) / mass
  void grid_solve(double dt) {
    const int klo = (dim == 3) ? ghost : 0;
    const int khi = (dim == 3) ? (gz - ghost - 1) : 0;

#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int k = klo; k <= khi; k++) {
      for (int j = ghost; j <= gy - ghost - 1; j++) {
        for (int i = ghost; i <= gx - ghost - 1; i++) {
          const int n = idx(i, j, k);
          if (mass[n] < MASS_TOL) {
            velocity_new_x[n] = 0.0;
            velocity_new_y[n] = 0.0;
            velocity_new_z[n] = 0.0;
            continue;
          }
          const double inv_m = 1.0 / mass[n];
          velocity_x[n] = momentum_x[n] * inv_m;
          velocity_y[n] = momentum_y[n] * inv_m;
          velocity_z[n] = momentum_z[n] * inv_m;

          velocity_new_x[n] = velocity_x[n]
            + dt * (force_int_x[n] + force_ext_x[n]) * inv_m;
          velocity_new_y[n] = velocity_y[n]
            + dt * (force_int_y[n] + force_ext_y[n]) * inv_m;
          velocity_new_z[n] = velocity_z[n]
            + dt * (force_int_z[n] + force_ext_z[n]) * inv_m;
        }
      }
    }
  }

  // B-bar pass 2: normalize raw_div_v by mass
  void normalize_div_v() {
    const int klo = (dim == 3) ? ghost : 0;
    const int khi = (dim == 3) ? (gz - ghost - 1) : 0;

#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int k = klo; k <= khi; k++) {
      for (int j = ghost; j <= gy - ghost - 1; j++) {
        for (int i = ghost; i <= gx - ghost - 1; i++) {
          const int n = idx(i, j, k);
          if (mass[n] > MASS_TOL)
            div_v[n] = raw_div_v[n] / mass[n];
          else
            div_v[n] = 0.0;
        }
      }
    }
  }

  // Compute density field (for surface detection)
  void compute_density() {
    const double cv = cell_volume();
    if (cv < 1e-30) return;
    const double inv_cv = 1.0 / cv;
    for (int n = 0; n < ntotal; n++)
      density[n] = mass[n] * inv_cv;
  }

  // Diagnostics: total mass and momentum on owned nodes
  void compute_diagnostics(double& total_mass,
                           double& mom_x, double& mom_y, double& mom_z) const {
    total_mass = mom_x = mom_y = mom_z = 0.0;
    const int klo = (dim == 3) ? ghost : 0;
    const int khi = (dim == 3) ? (gz - ghost - 1) : 0;

    for (int k = klo; k <= khi; k++)
      for (int j = ghost; j <= gy - ghost - 1; j++)
        for (int i = ghost; i <= gx - ghost - 1; i++) {
          const int n = idx(i, j, k);
          total_mass += mass[n];
          mom_x += momentum_x[n];
          mom_y += momentum_y[n];
          mom_z += momentum_z[n];
        }
  }

  // Find or add a body on a node (for Bardenhagen contact)
  // Returns pointer to NodeBodyData, or nullptr if MAX exceeded
  NodeBodyData* find_or_add_body(int node_idx, tagint bid) {
    assert(contact_bardenhagen);
    const int base = node_idx * MAX_BODIES_PER_NODE;
    const int nb = num_bodies[node_idx];

    // Search existing
    for (int b = 0; b < nb; b++) {
      if (body_data[base + b].body_id == bid)
        return &body_data[base + b];
    }

    // Add new
    if (nb >= MAX_BODIES_PER_NODE) return nullptr;
    body_data[base + nb].zero();
    body_data[base + nb].body_id = bid;
    num_bodies[node_idx] = nb + 1;
    return &body_data[base + nb];
  }
};

} // namespace CoupMPM
} // namespace LAMMPS_NS

#endif // COUPMPM_GRID_H
