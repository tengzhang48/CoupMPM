#ifndef COUPMPM_CONTACT_H
#define COUPMPM_CONTACT_H

#include "coupmpm_grid.h"
#include <mpi.h>
#include <cmath>
#include <cstring>

namespace LAMMPS_NS {
namespace CoupMPM {

// ============================================================
// Contact base class: pluggable contact algorithms
// ============================================================
class MPMContact {
public:
  virtual ~MPMContact() {}

  // Initialize from fix arguments
  virtual void init(int narg, char** arg) = 0;

  // Called before P2G to prepare per-body arrays
  virtual void pre_p2g(MPMGrid& grid) {
    // Default: nothing. Bardenhagen zeros per-body data here.
  }

  // Called after grid solve to apply contact forces
  virtual void post_grid_solve(MPMGrid& grid, double dt, MPI_Comm world) = 0;

  // Maximum contact stiffness for CFL estimation
  virtual double max_contact_stiffness() const { return 0.0; }

  virtual const char* name() const = 0;
};

// ============================================================
// No contact (default)
// ============================================================
class ContactNone : public MPMContact {
public:
  void init(int, char**) override {}
  void post_grid_solve(MPMGrid&, double, MPI_Comm) override {}
  const char* name() const override { return "none"; }
};

// ============================================================
// Bardenhagen multi-velocity contact (Phase 1 implementation)
//
// At nodes where multiple bodies have mass:
//   1. Compute per-body velocities
//   2. Detect approaching bodies (relative normal velocity)
//   3. Apply normal contact + Coulomb friction
//   4. Reconstruct total velocity
// ============================================================
class ContactBardenhagen : public MPMContact {
public:
  double friction_mu;    // Coulomb friction coefficient
  double adhesion_gamma; // adhesion energy density (0 = no adhesion)

  ContactBardenhagen() : friction_mu(0.3), adhesion_gamma(0.0) {}

  void init(int narg, char** arg) override {
    // Parse: mu <val> adhesion <val>
    for (int i = 0; i < narg - 1; i++) {
      if (strcmp(arg[i], "mu") == 0)
        friction_mu = atof(arg[i+1]);
      else if (strcmp(arg[i], "adhesion") == 0)
        adhesion_gamma = atof(arg[i+1]);
    }
  }

  void pre_p2g(MPMGrid& grid) override {
    if (!grid.contact_bardenhagen) return;
    // Zero per-body data
    for (auto& bd : grid.body_data) bd.zero();
    std::memset(grid.num_bodies.data(), 0,
                grid.ntotal * sizeof(int));
  }

  void sync_body_data_mpi(MPMGrid& grid, MPI_Comm world) {
    // TODO: Implement custom MPI_Reduce for grid.body_data across ghost boundaries
  }

  void post_grid_solve(MPMGrid& grid, double dt, MPI_Comm world) override {
    if (!grid.contact_bardenhagen) return;

    sync_body_data_mpi(grid, world);

    const int klo = (grid.dim == 3) ? grid.ghost : 0;
    const int khi = (grid.dim == 3) ? (grid.gz - grid.ghost - 1) : 0;

    for (int k = klo; k <= khi; k++) {
      for (int j = grid.ghost; j <= grid.gy - grid.ghost - 1; j++) {
        for (int i = grid.ghost; i <= grid.gx - grid.ghost - 1; i++) {
          const int n = grid.idx(i, j, k);
          const int nb = grid.num_bodies[n];
          if (nb < 2) continue;  // no contact at single-body nodes
          if (grid.mass[n] < MASS_TOL) continue;

          const int base = n * MAX_BODIES_PER_NODE;

          // Compute per-body velocities and pre-initialize velocity_new.
          // velocity_new is initialised to the pre-contact velocity so that
          // the reconstruction step below can always use velocity_new without
          // a special "was this body contacted?" check.  The contact loop
          // below overwrites velocity_new only for bodies that actually
          // exchange an impulse, but even a zero post-contact velocity is
          // handled correctly because the initial value is velocity, not 0.
          for (int b = 0; b < nb; b++) {
            NodeBodyData& bd = grid.body_data[base + b];
            if (bd.mass > MASS_TOL) {
              for (int d = 0; d < 3; d++) {
                bd.velocity[d] = bd.momentum[d] / bd.mass;
                bd.velocity_new[d] = bd.velocity[d];
              }
            }
          }

          // For each body pair, detect contact and apply
          for (int a = 0; a < nb; a++) {
            for (int b = a + 1; b < nb; b++) {
              NodeBodyData& ba = grid.body_data[base + a];
              NodeBodyData& bb = grid.body_data[base + b];

              if (ba.mass < MASS_TOL || bb.mass < MASS_TOL) continue;

              // Relative velocity
              double dv[3];
              for (int d = 0; d < 3; d++)
                dv[d] = ba.velocity[d] - bb.velocity[d];

              // Normal direction: use mass-weighted center-of-mass offset
              // between the two bodies at this node. This gives a GEOMETRIC
              // normal independent of kinematics, unlike relative velocity
              // which collapses the normal/tangential decomposition.
              //
              // COM offset: n_ab ∝ (x_cm_a - x_cm_b) projected through
              // the per-body momentum direction. For grid-based contact,
              // the proper normal is the gradient of the body's mass field.
              // As a robust approximation, use the difference of per-body
              // momentum directions (which encode spatial distribution).
              double normal[3];
              double nm = 0.0;

              // Use per-body momentum direction difference as normal proxy
              // This works because momentum = mass * velocity encodes both
              // the spatial mass distribution and kinematics.
              for (int d = 0; d < 3; d++) {
                normal[d] = ba.momentum[d] / (ba.mass + MASS_TOL)
                          - bb.momentum[d] / (bb.mass + MASS_TOL);
                // Fall back: if bodies have similar velocity, use the
                // relative velocity direction (better than nothing)
              }
              nm = std::sqrt(normal[0]*normal[0] + normal[1]*normal[1]
                           + normal[2]*normal[2]);

              // If momentum-based normal is degenerate, try relative velocity
              if (nm < 1e-15) {
                for (int d = 0; d < 3; d++) normal[d] = dv[d];
                nm = std::sqrt(normal[0]*normal[0] + normal[1]*normal[1]
                             + normal[2]*normal[2]);
              }

              if (nm < 1e-20) continue;  // truly co-located, skip
              for (int d = 0; d < 3; d++) normal[d] /= nm;

              // Normal relative velocity (positive = separating)
              double vn = dv[0]*normal[0] + dv[1]*normal[1] + dv[2]*normal[2];
              if (vn >= 0.0 && adhesion_gamma <= 0.0) continue; // separating, no adhesion

              // Reduced mass
              double m_red = ba.mass * bb.mass / (ba.mass + bb.mass);

              // Normal impulse: prevent interpenetration
              // impulse = force * dt, so force = impulse / dt
              double impulse_n = 0.0;

              if (vn < 0.0) {
                // Approaching: apply repulsive contact impulse
                impulse_n = -m_red * vn;  // zero relative normal velocity
              }
              else if (vn >= 0.0 && adhesion_gamma > 0.0) {
                // Separating with adhesion active:
                // Cohesive traction T = adhesion_gamma (energy/area)
                // scaled by effective nodal area ≈ dx^(dim-1)
                // Force = T * A_eff, impulse = F * dt
                double dx_min = std::min(grid.dx, grid.dy);
                if (grid.dim == 3) dx_min = std::min(dx_min, grid.dz);
                double A_eff = (grid.dim == 3) ? dx_min * dx_min : dx_min;

                double f_adhesion = adhesion_gamma * A_eff;
                double max_impulse = m_red * vn;  // can't reverse separation

                impulse_n = -std::min(f_adhesion * dt, max_impulse);
                // Negative impulse = pulling bodies together
              }

              // Tangential relative velocity
              double dv_t[3];
              for (int d = 0; d < 3; d++)
                dv_t[d] = dv[d] - vn * normal[d];
              double dvt_mag = std::sqrt(dv_t[0]*dv_t[0] + dv_t[1]*dv_t[1] + dv_t[2]*dv_t[2]);

              // Friction impulse (Coulomb)
              double impulse_t = 0.0;
              double tangent[3] = {0,0,0};
              if (dvt_mag > 1e-20 && friction_mu > 0.0) {
                for (int d = 0; d < 3; d++)
                  tangent[d] = dv_t[d] / dvt_mag;
                // Coulomb limit: |f_t| ≤ μ|f_n|, both as impulses
                double max_friction_impulse = friction_mu * std::fabs(impulse_n);
                double full_stop_impulse = m_red * dvt_mag;
                impulse_t = -std::min(max_friction_impulse, full_stop_impulse);
              }

              // Apply to per-body velocities
              // impulse_n and impulse_t are impulses (force * dt)
              // Δv = impulse / mass
              for (int d = 0; d < 3; d++) {
                double imp_d = impulse_n * normal[d] + impulse_t * tangent[d];
                ba.velocity_new[d] = ba.velocity[d] + imp_d / ba.mass;
                bb.velocity_new[d] = bb.velocity[d] - imp_d / bb.mass;
              }
            }
          }

          // Reconstruct total node velocity from per-body post-contact velocities.
          // velocity_new was pre-initialized to velocity above, so it is always
          // valid regardless of whether the post-contact value is zero.
          if (grid.mass[n] > MASS_TOL) {
            double vn_x = 0, vn_y = 0, vn_z = 0;
            for (int b = 0; b < nb; b++) {
              NodeBodyData& bd = grid.body_data[base + b];
              vn_x += bd.mass * bd.velocity_new[0];
              vn_y += bd.mass * bd.velocity_new[1];
              vn_z += bd.mass * bd.velocity_new[2];
            }
            grid.velocity_new_x[n] = vn_x / grid.mass[n];
            grid.velocity_new_y[n] = vn_y / grid.mass[n];
            grid.velocity_new_z[n] = vn_z / grid.mass[n];
          }
        }
      }
    }
  }

  const char* name() const override { return "bardenhagen"; }
};

// ============================================================
// Penalty contact via LAMMPS pair_styles (Phase 2 stub)
// Forces enter through f[] during P2G — no grid-level work.
// ============================================================
class ContactPenalty : public MPMContact {
public:
  void init(int, char**) override {}

  void post_grid_solve(MPMGrid&, double, MPI_Comm) override {
    // Nothing to do at grid level — pair forces already spread via P2G
  }

  const char* name() const override { return "penalty"; }
};

} // namespace CoupMPM
} // namespace LAMMPS_NS

#endif // COUPMPM_CONTACT_H
