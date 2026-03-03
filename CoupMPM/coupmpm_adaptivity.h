#ifndef COUPMPM_ADAPTIVITY_H
#define COUPMPM_ADAPTIVITY_H

#include "coupmpm_grid.h"
#include "coupmpm_stress.h"
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>

namespace LAMMPS_NS {
namespace CoupMPM {

// ============================================================
// Particle adaptivity: splitting and merging
//
// Problem: During large deformation, particles cluster in
// compression zones (wasted computation, instability) and
// spread apart in tension zones (poor quadrature, leaking).
//
// Solution: Monitor J = det(F) and nearest-neighbor distance.
//   J > J_split_hi  → particle stretched too thin → split
//   J < J_split_lo  → particle compressed too much → merge
//   Also trigger on neighbor distance > distance_split * dx
//
// Splitting strategy:
//   Parent at x_p with mass m_p, volume vol0_p, F_p, σ_p, etc.
//   Split into n_child children (2^dim: 4 in 2D, 8 in 3D)
//   placed symmetrically around x_p at distance ±δ in each dim.
//
//   Children inherit:
//     mass     = m_p / n_child
//     vol0     = vol0_p / n_child
//     F_def    = F_p  (exact copy — same deformation state)
//     stress   = σ_p  (same — constitutive state unchanged)
//     velocity = v_p  (same — no relative motion introduced)
//     Bp       = Bp_p (same — APIC affine field preserved)
//     state    = state_p (same — history vars preserved)
//     body_id  = same molecule ID
//     surface  = recompute at next surface_interval
//
//   Conservation (exact by construction):
//     Total mass:     Σ m_child = m_p  ✓
//     Total momentum: Σ m_child * v_p = m_p * v_p  ✓
//     Total volume:   Σ vol0_child = vol0_p  ✓
//     Angular momentum: Σ m_child * (x_child × v_p) ≈ m_p * (x_p × v_p)
//       (Exact if δ → 0. For finite δ, error is O(δ²) which
//        vanishes as children are placed close to parent.)
//
// Merging strategy:
//   Find particle pairs closer than merge_distance * dx.
//   Mass-weighted average of position, velocity.
//   For F: use the F of the heavier particle (or mass-weighted
//   polar decomposition — but that's expensive and the simpler
//   approach works for moderate compression).
//
// LAMMPS integration:
//   Splitting creates new atoms via atom->avec->create_atom()
//   or by directly growing arrays and setting fields.
//   Merging removes atoms via atom->avec->copy() + nlocal--.
//   Both require subsequent neighbor list rebuild.
// ============================================================

// Data for a particle to be created (child of a split)
struct ChildParticle {
  double x[3];
  double v[3];
  double mass;
  double vol0;
  double F_def[9];
  double stress_v[6];
  double Bp[9];
  double state[9];   // matches N_STATE in atom_vec_mpm
  int body_id;
  int type;
};

// Data describing a merge operation
struct MergePair {
  int i, j;           // local indices of particles to merge
  double dist;        // distance between them
};

class ParticleAdaptivity {
public:
  // --- Splitting parameters ---
  double J_split_hi;       // Split if J > this (default 2.0)
  double J_split_lo;       // Merge if J < this (default 0.3)
  double distance_split;   // Split if nearest neighbor > this * dx (default 1.8)
  double merge_distance;   // Merge if neighbor distance < this * dx (default 0.3)
  int max_split_level;     // Maximum times a lineage can split (default 3)
  int check_interval;      // Check every N steps (default 20)
  bool enabled;

  // --- Statistics ---
  int n_splits_last;
  int n_merges_last;

  ParticleAdaptivity()
    : J_split_hi(2.0), J_split_lo(0.3),
      distance_split(1.8), merge_distance(0.3),
      max_split_level(3), check_interval(20), enabled(false),
      n_splits_last(0), n_merges_last(0) {}

  // ============================================================
  // Identify particles that need splitting.
  // Returns list of local indices.
  // ============================================================
  std::vector<int> find_split_candidates(
      int nlocal, int dim,
      const double* F_def,     // [nlocal * 9]
      const double* vol0,
      double dx_min) const
  {
    std::vector<int> candidates;
    if (!enabled) return candidates;

    for (int p = 0; p < nlocal; p++) {
      const double* Fp = &F_def[p * 9];
      double J = Mat3::det(Fp);

      // J-based criterion: particle volume expanded too much
      if (J > J_split_hi) {
        candidates.push_back(p);
        continue;
      }

      // Could also check principal stretches for anisotropic splitting,
      // but J is simpler and sufficient for Phase 1.
      // Future: eigenvalues of F^T F for directional splitting.
    }
    return candidates;
  }

  // ============================================================
  // Generate child particles for a split.
  //
  // Children are placed symmetrically around the parent
  // at distance δ in each coordinate direction.
  // δ = parent_spacing / 2, where parent_spacing is estimated
  // from (vol0 * J)^(1/dim).
  //
  // In 3D: 8 children at (±δ, ±δ, ±δ)
  // In 2D: 4 children at (±δ, ±δ, 0)
  // ============================================================
  std::vector<ChildParticle> generate_children(
      int dim,
      const double xp[3], const double vp[3],
      double mass_p, double vol0_p,
      const double F_def_p[9], const double stress_p[6],
      const double Bp_p[9], const double state_p[],
      int n_state, int body_id, int type) const
  {
    const double J = Mat3::det(F_def_p);
    const double current_vol = vol0_p * std::fabs(J);

    // Child count: 2^dim
    const int n_child = (dim == 3) ? 8 : 4;

    // Placement offset: half the effective particle spacing
    // Effective spacing = (current_vol)^(1/dim)
    double spacing = (dim == 3) ? std::cbrt(current_vol) : std::sqrt(current_vol);
    double delta = spacing * 0.25;  // Place children at ±spacing/4

    std::vector<ChildParticle> children(n_child);

    // Generate offset patterns
    // 2D: (±1, ±1), 3D: (±1, ±1, ±1)
    int idx = 0;
    int kmax = (dim == 3) ? 2 : 1;
    for (int kk = 0; kk < kmax; kk++) {
      double dz = (dim == 3) ? ((kk == 0) ? -delta : delta) : 0.0;
      for (int jj = 0; jj < 2; jj++) {
        double dy = (jj == 0) ? -delta : delta;
        for (int ii = 0; ii < 2; ii++) {
          double dx = (ii == 0) ? -delta : delta;

          ChildParticle& c = children[idx];
          c.x[0] = xp[0] + dx;
          c.x[1] = xp[1] + dy;
          c.x[2] = xp[2] + dz;

          // Velocity: same as parent (no relative motion)
          c.v[0] = vp[0]; c.v[1] = vp[1]; c.v[2] = vp[2];

          // Mass and volume: divided equally
          c.mass = mass_p / n_child;
          c.vol0 = vol0_p / n_child;

          // Deformation state: exact copy
          std::memcpy(c.F_def, F_def_p, 9 * sizeof(double));
          std::memcpy(c.stress_v, stress_p, 6 * sizeof(double));
          std::memcpy(c.Bp, Bp_p, 9 * sizeof(double));

          int ncopy = (n_state < 9) ? n_state : 9;
          std::memset(c.state, 0, 9 * sizeof(double));
          if (state_p && ncopy > 0)
            std::memcpy(c.state, state_p, ncopy * sizeof(double));

          c.body_id = body_id;
          c.type = type;

          idx++;
        }
      }
    }

    return children;
  }

  // ============================================================
  // Identify particles that should be merged.
  // Returns list of (i, j) pairs where j > i.
  //
  // Simple O(N²) within each grid cell — acceptable because
  // only compressed cells have excess particles, and this runs
  // every check_interval steps.
  //
  // In LAMMPS, we can use the neighbor list for efficiency,
  // but for Phase 1, a direct J-based check is simpler:
  // merge candidates are particles with J < J_split_lo.
  // ============================================================
  std::vector<MergePair> find_merge_candidates(
      int nlocal, int dim,
      double** x,
      const double* F_def,
      const int* molecule,
      double dx_min) const
  {
    std::vector<MergePair> pairs;
    if (!enabled) return pairs;

    double merge_dist2 = (merge_distance * dx_min) * (merge_distance * dx_min);

    // Collect particles with J below merge threshold
    std::vector<int> compressed;
    for (int p = 0; p < nlocal; p++) {
      const double* Fp = &F_def[p * 9];
      double J = Mat3::det(Fp);
      if (J < J_split_lo) compressed.push_back(p);
    }

    // Find nearest same-body pairs among compressed particles
    std::vector<bool> used(nlocal, false);
    for (size_t a = 0; a < compressed.size(); a++) {
      int i = compressed[a];
      if (used[i]) continue;

      double best_dist2 = 1e30;
      int best_j = -1;

      for (size_t b = a + 1; b < compressed.size(); b++) {
        int j = compressed[b];
        if (used[j]) continue;

        // Only merge particles from the same body
        if (molecule && molecule[i] != molecule[j]) continue;

        double d2 = 0;
        for (int d = 0; d < dim; d++) {
          double dd = x[i][d] - x[j][d];
          d2 += dd * dd;
        }

        if (d2 < best_dist2 && d2 < merge_dist2) {
          best_dist2 = d2;
          best_j = j;
        }
      }

      if (best_j >= 0) {
        MergePair mp;
        mp.i = i;
        mp.j = best_j;
        mp.dist = std::sqrt(best_dist2);
        pairs.push_back(mp);
        used[i] = true;
        used[best_j] = true;
      }
    }

    return pairs;
  }

  // ============================================================
  // Compute merged particle properties from two parents.
  //
  // Position & velocity: mass-weighted average
  // F_def: from the heavier particle (simpler and more robust
  //   than polar decomposition averaging)
  // stress: recompute from F after merge (via constitutive law)
  // vol0: sum
  // mass: sum
  // Bp: mass-weighted average
  // state: from heavier particle
  // ============================================================
  static ChildParticle merge_particles(
      const double x_i[3], const double v_i[3],
      double mass_i, double vol0_i,
      const double F_i[9], const double stress_i[6],
      const double Bp_i[9], const double state_i[],
      const double x_j[3], const double v_j[3],
      double mass_j, double vol0_j,
      const double F_j[9], const double stress_j[6],
      const double Bp_j[9], const double state_j[],
      int n_state, int body_id, int type)
  {
    ChildParticle merged;
    double total_mass = mass_i + mass_j;
    double inv_m = (total_mass > 1e-30) ? (1.0 / total_mass) : 0.0;

    // Mass-weighted position and velocity
    for (int d = 0; d < 3; d++) {
      merged.x[d] = (mass_i * x_i[d] + mass_j * x_j[d]) * inv_m;
      merged.v[d] = (mass_i * v_i[d] + mass_j * v_j[d]) * inv_m;
    }

    merged.mass = total_mass;
    merged.vol0 = vol0_i + vol0_j;

    // F, stress, state from heavier particle
    const double* F_keep  = (mass_i >= mass_j) ? F_i : F_j;
    const double* s_keep  = (mass_i >= mass_j) ? stress_i : stress_j;
    const double* st_keep = (mass_i >= mass_j) ? state_i : state_j;

    std::memcpy(merged.F_def, F_keep, 9 * sizeof(double));
    std::memcpy(merged.stress_v, s_keep, 6 * sizeof(double));

    // Mass-weighted Bp average
    for (int k = 0; k < 9; k++)
      merged.Bp[k] = (mass_i * Bp_i[k] + mass_j * Bp_j[k]) * inv_m;

    int ncopy = (n_state < 9) ? n_state : 9;
    std::memset(merged.state, 0, 9 * sizeof(double));
    if (st_keep && ncopy > 0)
      std::memcpy(merged.state, st_keep, ncopy * sizeof(double));

    merged.body_id = body_id;
    merged.type = type;

    return merged;
  }
};

} // namespace CoupMPM
} // namespace LAMMPS_NS

#endif // COUPMPM_ADAPTIVITY_H
