#ifndef COUPMPM_COHESIVE_H
#define COUPMPM_COHESIVE_H

#include "coupmpm_grid.h"
#include "coupmpm_kernel.h"
#include "coupmpm_stress.h"
#include "neigh_list.h"
#include "atom.h"
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cassert>
#include <unordered_map>

namespace LAMMPS_NS {
namespace CoupMPM {

// ============================================================
// Dynamic Particle-Based Cohesive Zones
//
// Unlike traditional FEM cohesive zones (pre-defined along
// mesh interfaces) or the Crook & Homel reference-config method
// (fixed at t=0), this module creates cohesive bonds DYNAMICALLY
// when surface particles from different bodies come into range.
//
// Each bond stores its own reference state at the moment of
// formation, enabling:
//   - Cells adhering to any neighbor they touch
//   - Receptor-ligand binding at variable times
//   - History-dependent adhesion (damage accumulation)
//   - Clean transition to frictional contact after failure
//
// Biology mapping:
//   bond formation distance → receptor reach (~100-500 nm)
//   traction-separation law → receptor-ligand unbinding curve
//   damage accumulation     → bond fatigue under cyclic load
//   differential adhesion   → type-dependent sigma_max, delta_n
//
// ============================================================

// Traction-separation law types
enum class CZLawType {
  NEEDLEMAN_XU,   // Exponential (Needleman & Xu 1994)
  LINEAR_ELASTIC, // Linear spring (for stiff bonding / testing)
  RECEPTOR_LIGAND // Bell model receptor-ligand (for biology)
};

// ============================================================
// Per-bond data
//
// Each CohesiveBond represents one adhesive connection between
// two particles. Stored on the rank that owns particle i.
// ============================================================
struct CohesiveBond {
  // --- Identity ---
  tagint tag_i;     // global tag of particle i (local owner)
  tagint tag_j;     // global tag of particle j (may be ghost)
  tagint body_i;       // body (molecule) ID of particle i
  tagint body_j;       // body (molecule) ID of particle j
  int type_i;       // atom type of i (for type-dependent CZ params)
  int type_j;       // atom type of j

  // --- Reference state (captured at formation time) ---
  double x0_i[3];   // position of i when bond formed
  double x0_j[3];   // position of j when bond formed
  double n0[3];     // reference normal (i → j direction at formation)
  double delta0;    // initial gap at formation
  double A_ref;     // reference contact area (estimated at formation)

  // --- Current state ---
  double damage;    // accumulated damage [0, 1]
  double delta_max_n; // maximum normal separation experienced
  double delta_max_t; // maximum tangential separation experienced
  long step_formed; // timestep when bond was created

  // --- Flags ---
  bool active;      // is this bond still alive?

  void zero() {
    tag_i = tag_j = -1;
    body_i = body_j = -1;
    type_i = type_j = 0;
    std::memset(x0_i, 0, sizeof(x0_i));
    std::memset(x0_j, 0, sizeof(x0_j));
    std::memset(n0, 0, sizeof(n0));
    delta0 = 0.0;
    A_ref = 0.0;
    damage = 0.0;
    delta_max_n = delta_max_t = 0.0;
    step_formed = 0;
    active = false;
  }
};

// ============================================================
// Type-pair cohesive parameters
// Indexed by (type_i, type_j) for differential adhesion.
// ============================================================
struct CZParams {
  double sigma_max;   // maximum normal traction (Pa)
  double tau_max;     // maximum tangential traction (Pa)
  double delta_n;     // characteristic normal displacement (m)
  double delta_t;     // characteristic tangential displacement (m)
  double delta_n_max; // failure normal displacement (m)
  double delta_t_max; // failure tangential displacement (m)
  double formation_dist; // bond forms when gap < this (m)
  double base_rate;   // unstressed off-rate for Bell model (1/time)

  CZParams()
    : sigma_max(100.0), tau_max(100.0),
      delta_n(1e-4), delta_t(1e-4),
      delta_n_max(2e-4), delta_t_max(2e-4),
      formation_dist(5e-4), base_rate(1.0) {}
};

// ============================================================
// CohesiveZoneManager
//
// Manages all dynamic cohesive bonds. Called from fix_coupmpm
// at specific points in the timestep:
//
//   1. detect_new_bonds():  after surface detection, find
//      surface particle pairs close enough to bond.
//      Called every bond_check_interval steps.
//
//   2. compute_forces():  compute traction from each bond's
//      current separation, convert to forces on particles.
//      Called every step, before P2G.
//      Forces written to particle f[] array.
//
//   3. update_damage():  after forces, update damage state.
//      Break bonds that exceed failure criterion.
//      Called every step after compute_forces().
//
// ============================================================
class CohesiveZoneManager {
public:
  // --- Parameters ---
  CZLawType law_type;
  int max_bonds_per_particle;
  int bond_check_interval;
  bool enabled;

  // Type-pair parameters: indexed as params[type_i * max_types + type_j]
  std::vector<CZParams> params;
  int max_types;

  // --- Bond storage ---
  std::vector<CohesiveBond> bonds;

  // --- Statistics ---
  int n_formed_last;
  int n_broken_last;
  int n_active;

  CohesiveZoneManager()
    : law_type(CZLawType::NEEDLEMAN_XU),
      max_bonds_per_particle(6),
      bond_check_interval(10),
      enabled(false),
      max_types(0),
      n_formed_last(0), n_broken_last(0), n_active(0) {}

  // Initialize type-pair parameters (uniform default)
  void init_params(int ntypes, double sigma, double tau,
                   double dn, double dt_cz, double dn_max, double dt_max,
                   double form_dist) {
    max_types = ntypes;
    params.resize(ntypes * ntypes);
    for (auto& p : params) {
      p.sigma_max = sigma;
      p.tau_max = tau;
      p.delta_n = dn;
      p.delta_t = dt_cz;
      p.delta_n_max = dn_max;
      p.delta_t_max = dt_max;
      p.formation_dist = form_dist;
      p.base_rate = 1.0;
    }
  }

  // Set type-pair specific parameters (for differential adhesion)
  void set_pair_params(int ti, int tj, const CZParams& p) {
    assert(ti > 0 && ti <= max_types && tj > 0 && tj <= max_types);
    params[(ti-1) * max_types + (tj-1)] = p;
    params[(tj-1) * max_types + (ti-1)] = p; // symmetric
  }

  const CZParams& get_params(int ti, int tj) const {
    return params[(ti-1) * max_types + (tj-1)];
  }

  // ============================================================
  // Detect new bonds between surface particles.
  //
  // Scans all surface particles on this rank. For each pair
  // (i, j) from different bodies within formation distance:
  //   - Check neither already has max_bonds_per_particle
  //   - Check no existing bond between this pair
  //   - Create bond with current state as reference
  //
  // Uses a simple O(N_surface^2) search. For large simulations,
  // use LAMMPS neighbor lists instead (future optimization).
  //
  // Parameters:
  //   nlocal, nghost: local + ghost atom counts
  //   x: positions [nlocal+nghost][3]
  //   tag: global atom tags [nlocal+nghost]
  //   type: atom types [nlocal+nghost]
  //   molecule: body IDs [nlocal+nghost]
  //   surface: surface flags [nlocal+nghost]
  //   F_def: deformation gradients [nlocal*9] (for area estimation)
  //   vol0: reference volumes [nlocal]
  //   step: current timestep
  //   dim: 2 or 3
  //   dx: grid spacing (for area estimation)
  // ============================================================
  int detect_new_bonds(int nlocal, int nghost,
                       double** x, tagint* tag, int* type,
                       tagint* molecule, int* surface,
                       double* F_def, double* vol0,
                       long step, int dim, double dx,
                       NeighList* list)
  {
    if (!enabled) return 0;
    if (!list) return 0;

    // Count existing bonds per particle (local only) using a hash map for O(N) lookup
    std::unordered_map<tagint, int> tag2local;
    for (int p = 0; p < nlocal; p++) tag2local[tag[p]] = p;

    std::vector<int> bond_count(nlocal, 0);
    for (const auto& b : bonds) {
      if (!b.active) continue;
      auto it = tag2local.find(b.tag_i);
      if (it != tag2local.end()) bond_count[it->second]++;
    }

    int n_formed = 0;

    int inum = list->inum;
    int *ilist = list->ilist;
    int *numneigh = list->numneigh;
    int **firstneigh = list->firstneigh;

    for (int ii = 0; ii < inum; ii++) {
      int i = ilist[ii];
      if (!surface[i]) continue;
      if (bond_count[i] >= max_bonds_per_particle) continue;

      int *jlist = firstneigh[i];
      int jnum = numneigh[i];

      for (int jj = 0; jj < jnum; jj++) {
        int j = jlist[jj];
        j &= NEIGHMASK; // Strip LAMMPS ghost bit flags

        if (!surface[j]) continue;

        // Must be different bodies
        if (molecule[i] == molecule[j]) continue;

        // Type-pair formation distance
        const CZParams& prm = get_params(type[i], type[j]);

        // Distance check
        double dx_ij[3], dist2 = 0.0;
        for (int d = 0; d < 3; d++) {
          dx_ij[d] = x[j][d] - x[i][d];
          dist2 += dx_ij[d] * dx_ij[d];
        }
        double dist = std::sqrt(dist2);
        if (dist > prm.formation_dist) continue;

        // Check no existing bond between this tag pair
        tagint ti = tag[i], tj = tag[j];
        bool exists = false;
        for (const auto& b : bonds) {
          if (!b.active) continue;
          if ((b.tag_i == ti && b.tag_j == tj) ||
              (b.tag_i == tj && b.tag_j == ti)) {
            exists = true;
            break;
          }
        }
        if (exists) continue;

        // Check j's bond count (if j is local)
        if (j < nlocal && bond_count[j] >= max_bonds_per_particle) continue;

        // --- Create bond ---
        CohesiveBond bond;
        bond.tag_i = ti;
        bond.tag_j = tj;
        bond.body_i = molecule[i];
        bond.body_j = molecule[j];
        bond.type_i = type[i];
        bond.type_j = type[j];

        // Reference state
        std::memcpy(bond.x0_i, x[i], 3 * sizeof(double));
        std::memcpy(bond.x0_j, x[j], 3 * sizeof(double));

        // Normal: i → j direction
        if (dist > 1e-20) {
          for (int d = 0; d < 3; d++) bond.n0[d] = dx_ij[d] / dist;
        } else {
          bond.n0[0] = 1.0; bond.n0[1] = 0.0; bond.n0[2] = 0.0;
        }

        bond.delta0 = dist;  // initial gap (may be nonzero)

        // Estimate contact area from particle volume
        double vol_i = (i < nlocal && vol0) ? vol0[i] : dx * dx * dx;
        double A_est = (dim == 3) ? std::pow(vol_i, 2.0/3.0)
                                  : std::sqrt(vol_i);
        bond.A_ref = A_est;

        bond.damage = 0.0;
        bond.delta_max_n = 0.0;
        bond.delta_max_t = 0.0;
        bond.step_formed = step;
        bond.active = true;

        bonds.push_back(bond);
        bond_count[i]++;
        n_formed++;
      }
    }

    n_formed_last = n_formed;
    return n_formed;
  }

  // ============================================================
  // Compute cohesive forces from all active bonds.
  //
  // For each bond:
  //   1. Find current positions of particles i and j
  //   2. Compute current separation vector
  //   3. Decompose into normal and tangential components
  //      relative to the DEFORMED normal (Nanson-updated)
  //   4. Evaluate traction-separation law
  //   5. Convert traction to force using reference area
  //   6. Apply force to particle f[] arrays (equal & opposite)
  //
  // Forces enter the MPM cycle through f[] → P2G external force.
  // This is the same path as LAMMPS pair_style forces.
  //
  // nlocal, nghost: atom counts
  // x: current positions
  // f: force arrays (accumulated into)
  // tag: global tags (for finding particles)
  // F_def: deformation gradients (for normal update)
  // ============================================================
  void compute_forces(int nlocal, int nghost,
                      double** x, double** f,
                      tagint* tag, double* F_def,
                      int dim,
                      Atom *atom_ptr)
  {
    if (!enabled) return;

    n_active = 0;

    // Build local tag-to-index map covering local + ghost atoms
    std::unordered_map<tagint, int> tag2idx;
    for (int p = 0; p < nlocal + nghost; p++) tag2idx[tag[p]] = p;

    for (auto& bond : bonds) {
      if (!bond.active) continue;

      int li = tag2idx.count(bond.tag_i) ? tag2idx[bond.tag_i] : -1;
      int lj = tag2idx.count(bond.tag_j) ? tag2idx[bond.tag_j] : -1;

      // If either particle is not visible on this rank, skip
      if (li < 0 || lj < 0) continue;

      n_active++;

      // --- Current separation ---
      double sep[3], dist2 = 0.0;
      for (int d = 0; d < 3; d++) {
        sep[d] = x[lj][d] - x[li][d];
        dist2 += sep[d] * sep[d];
      }

      // --- Update normal using mass-weighted average of Fi and Fj cofactors ---
      // Nanson: n_deformed ~ cofactor(F) · n0 / |cofactor(F) · n0|
      double normal[3];
      if (F_def) {
        const double* Fi = &F_def[li * 9];
        const double* Fj = &F_def[lj * 9];
        double Ji = Mat3::det(Fi);
        double Jj = Mat3::det(Fj);

        double cof_i[9], cof_j[9];
        cof_i[0] = Fi[4]*Fi[8] - Fi[5]*Fi[7];
        cof_i[1] = Fi[5]*Fi[6] - Fi[3]*Fi[8];
        cof_i[2] = Fi[3]*Fi[7] - Fi[4]*Fi[6];
        cof_i[3] = Fi[2]*Fi[7] - Fi[1]*Fi[8];
        cof_i[4] = Fi[0]*Fi[8] - Fi[2]*Fi[6];
        cof_i[5] = Fi[1]*Fi[6] - Fi[0]*Fi[7];
        cof_i[6] = Fi[1]*Fi[5] - Fi[2]*Fi[4];
        cof_i[7] = Fi[2]*Fi[3] - Fi[0]*Fi[5];
        cof_i[8] = Fi[0]*Fi[4] - Fi[1]*Fi[3];

        cof_j[0] = Fj[4]*Fj[8] - Fj[5]*Fj[7];
        cof_j[1] = Fj[5]*Fj[6] - Fj[3]*Fj[8];
        cof_j[2] = Fj[3]*Fj[7] - Fj[4]*Fj[6];
        cof_j[3] = Fj[2]*Fj[7] - Fj[1]*Fj[8];
        cof_j[4] = Fj[0]*Fj[8] - Fj[2]*Fj[6];
        cof_j[5] = Fj[1]*Fj[6] - Fj[0]*Fj[7];
        cof_j[6] = Fj[1]*Fj[5] - Fj[2]*Fj[4];
        cof_j[7] = Fj[2]*Fj[3] - Fj[0]*Fj[5];
        cof_j[8] = Fj[0]*Fj[4] - Fj[1]*Fj[3];

        if (std::fabs(Ji) > 1e-20 && std::fabs(Jj) > 1e-20) {
          double mi = atom_ptr->mass[atom_ptr->type[li]];
          double mj = atom_ptr->mass[atom_ptr->type[lj]];
          double wtot = mi + mj;

          double nm = 0;
          for (int d = 0; d < 3; d++) {
            normal[d] = 0;
            for (int e = 0; e < 3; e++)
              normal[d] += ((mi * cof_i[3*d + e] + mj * cof_j[3*d + e]) / wtot)
                           * bond.n0[e];
            nm += normal[d] * normal[d];
          }
          nm = std::sqrt(nm);
          if (nm > 1e-20) {
            for (int d = 0; d < 3; d++) normal[d] /= nm;
          } else {
            std::memcpy(normal, bond.n0, 3 * sizeof(double));
          }
        } else {
          std::memcpy(normal, bond.n0, 3 * sizeof(double));
        }
      } else {
        std::memcpy(normal, bond.n0, 3 * sizeof(double));
      }

      // --- Decompose separation into normal and tangential ---
      // Subtract initial gap: displacement = current_sep - reference_sep
      double ref_sep[3];
      for (int d = 0; d < 3; d++)
        ref_sep[d] = bond.x0_j[d] - bond.x0_i[d];

      double disp[3];
      for (int d = 0; d < 3; d++)
        disp[d] = sep[d] - ref_sep[d];

      double delta_n = 0;
      for (int d = 0; d < 3; d++)
        delta_n += disp[d] * normal[d];

      double disp_t[3];
      for (int d = 0; d < 3; d++)
        disp_t[d] = disp[d] - delta_n * normal[d];

      double delta_t = std::sqrt(disp_t[0]*disp_t[0]
                               + disp_t[1]*disp_t[1]
                               + disp_t[2]*disp_t[2]);

      double tangent[3] = {0, 0, 0};
      if (delta_t > 1e-20) {
        for (int d = 0; d < 3; d++) tangent[d] = disp_t[d] / delta_t;
      }

      // Track maximum displacement for damage
      if (delta_n > bond.delta_max_n) bond.delta_max_n = delta_n;
      if (delta_t > bond.delta_max_t) bond.delta_max_t = delta_t;

      // --- Evaluate traction-separation law ---
      const CZParams& prm = get_params(bond.type_i, bond.type_j);
      double T_n = 0, T_t = 0;

      switch (law_type) {
        case CZLawType::NEEDLEMAN_XU:
          needleman_xu(delta_n, delta_t, prm, T_n, T_t);
          break;
        case CZLawType::LINEAR_ELASTIC:
          linear_elastic(delta_n, delta_t, prm, T_n, T_t);
          break;
        case CZLawType::RECEPTOR_LIGAND:
          receptor_ligand(delta_n, delta_t, prm, T_n, T_t);
          break;
      }

      // Apply damage: T_eff = (1 - D) * T
      double D = bond.damage;
      T_n *= (1.0 - D);
      T_t *= (1.0 - D);

      // --- Convert traction to force ---
      // F = T * A_ref (reference area, not current — Lagrangian)
      double force[3];
      for (int d = 0; d < 3; d++)
        force[d] = bond.A_ref * (T_n * normal[d] + T_t * tangent[d]);

      // --- Apply to particles (Newton's 3rd law) ---
      // Force on i: +force (pulled toward j)
      // Force on j: -force (pulled toward i)
      // Apply to ghosts as well; reverse_comm will accumulate ghost
      // contributions back to their owner ranks.
      if (li >= 0) {
        f[li][0] += force[0]; f[li][1] += force[1]; f[li][2] += force[2];
      }
      if (lj >= 0) {
        f[lj][0] -= force[0]; f[lj][1] -= force[1]; f[lj][2] -= force[2];
      }
    }
  }

  // ============================================================
  // Update damage and break failed bonds.
  // dt: current timestep size (used for RECEPTOR_LIGAND damage rate)
  // ============================================================
  void update_damage_and_break(int nlocal, double** x, tagint* tag,
                               int dim, Atom *atom_ptr, double dt)
  {
    if (!enabled) return;
    n_broken_last = 0;

    // Build local tag-to-index map covering local + ghost atoms
    int nghost = atom_ptr->nghost;
    std::unordered_map<tagint, int> tag2idx;
    for (int p = 0; p < nlocal + nghost; p++) tag2idx[tag[p]] = p;

    for (auto& bond : bonds) {
      if (!bond.active) continue;

      const CZParams& prm = get_params(bond.type_i, bond.type_j);

      int li = tag2idx.count(bond.tag_i) ? tag2idx[bond.tag_i] : -1;
      int lj = tag2idx.count(bond.tag_j) ? tag2idx[bond.tag_j] : -1;

      // Check failure criteria
      bool failed = false;

      // Maximum displacement criterion
      if (bond.delta_max_n > prm.delta_n_max) failed = true;
      if (bond.delta_max_t > prm.delta_t_max) failed = true;

      // Damage accumulation (linear softening beyond characteristic disp)
      double dn_ratio = bond.delta_max_n / prm.delta_n;
      double dt_ratio = bond.delta_max_t / prm.delta_t;
      double effective_ratio = std::sqrt(dn_ratio * dn_ratio + dt_ratio * dt_ratio);

      if (effective_ratio > 1.0) {
        // Beyond characteristic displacement: accumulate damage.
        // Use the combined max-displacement ratio as the denominator so
        // that the formula is consistent with the mixed effective_ratio
        // numerator (fixes the case where pure shear never reaches D=1
        // when only the normal max ratio is used for normalization).
        double dn_max_ratio = prm.delta_n_max / prm.delta_n;
        double dt_max_ratio = prm.delta_t_max / prm.delta_t;
        double combined_max_ratio = std::sqrt(dn_max_ratio * dn_max_ratio
                                            + dt_max_ratio * dt_max_ratio);
        double denom = combined_max_ratio - 1.0;
        if (std::fabs(denom) < 1e-20)
          bond.damage = 1.0;
        else
          bond.damage = std::min(1.0,
              (effective_ratio - 1.0) / denom);
      }

      // RECEPTOR_LIGAND: Bell-model damage rate (state update belongs here,
      // not in compute_forces, so damage is advanced exactly once per step).
      if (law_type == CZLawType::RECEPTOR_LIGAND) {
        double k_n = prm.sigma_max / prm.delta_n;
        double k_t = prm.tau_max / prm.delta_t;
        double T_n = k_n * bond.delta_max_n;
        double T_t = k_t * bond.delta_max_t;
        if (T_n > prm.sigma_max) T_n = prm.sigma_max;
        double T_mag = std::sqrt(T_n * T_n + T_t * T_t);
        double alpha = 2.0;
        double T_ratio = T_mag / prm.sigma_max;
        if (T_ratio > 0.5) {
          double damage_rate = prm.base_rate * std::exp(alpha * (T_ratio - 0.5));
          bond.damage += damage_rate * dt;
          if (bond.damage > 1.0) bond.damage = 1.0;
        }
      }

      if (bond.damage >= 1.0) failed = true;

      // Check if particles separated beyond any reasonable range
      if (li >= 0 && lj >= 0) {
        double d2 = 0;
        for (int d = 0; d < dim; d++) {
          double dd = x[lj][d] - x[li][d];
          d2 += dd * dd;
        }
        if (d2 > 100.0 * prm.formation_dist * prm.formation_dist)
          failed = true;
      }

      if (failed) {
        bond.active = false;
        n_broken_last++;
      }
    }

    // Compact bond list when the number of inactive (dead) bonds has grown
    // large enough that the scan overhead outweighs the compaction cost.
    // n_inactive is derived from the n_active count below, so no extra
    // O(N) scan is needed.

    // Count active
    n_active = 0;
    for (const auto& b : bonds) if (b.active) n_active++;

    int n_inactive = static_cast<int>(bonds.size()) - n_active;
    if (n_inactive > 100) {
      bonds.erase(
          std::remove_if(bonds.begin(), bonds.end(),
                         [](const CohesiveBond& b) { return !b.active; }),
          bonds.end());
    }
  }

  // ============================================================
  // Traction-separation laws
  // ============================================================

  // Needleman-Xu (1994): exponential cohesive law
  // T_n = (phi_n/delta_n) * (δ_n/δ̄_n) * exp(-δ_n/δ̄_n) * exp(-δ_t^2/δ̄_t^2)
  // This is the same law used in the Crook & Homel paper (Eq. 23-24)
  static void needleman_xu(double delta_n, double delta_t,
                           const CZParams& p,
                           double& T_n, double& T_t)
  {
    const double phi_n = std::exp(1.0) * p.sigma_max * p.delta_n;
    const double dn_ratio = delta_n / p.delta_n;
    const double dt_ratio = delta_t / p.delta_t;
    const double exp_n = std::exp(-dn_ratio);
    const double exp_t = std::exp(-dt_ratio * dt_ratio);

    // Normal traction (positive = tension pulling bodies together)
    T_n = (phi_n / p.delta_n) * dn_ratio * exp_n * exp_t;

    // Tangential traction: -dΦ/dδ_t (negative sign: resists sliding)
    // d/dδ_t [exp(-δ_t^2/δ_t^2)] = -2*(δ_t/δ_t)/δ_t * exp(...)
    // Outer minus cancels inner minus → positive magnitude, then negate
    // to make T_t oppose the slip direction.
    if (std::fabs(delta_t) > 1e-20) {
      T_t = -(phi_n / p.delta_t)
            * 2.0 * dt_ratio * (1.0 + dn_ratio) * exp_n * exp_t;
    } else {
      T_t = 0.0;
    }
  }

  // Linear elastic: T = k * delta (for testing / stiff bonding)
  static void linear_elastic(double delta_n, double delta_t,
                             const CZParams& p,
                             double& T_n, double& T_t)
  {
    // k_n = sigma_max / delta_n (stiffness at peak)
    double k_n = p.sigma_max / p.delta_n;
    double k_t = p.tau_max / p.delta_t;
    // Cohesive traction resists separation only; compression is handled
    // by the contact module, not the cohesive zone.
    T_n = (delta_n > 0.0) ? k_n * delta_n : 0.0;
    T_t = k_t * delta_t;
  }

  // Bell model receptor-ligand (for biological adhesion)
  // F_unbind = F0 * exp(-x_beta * F / kT)
  // Stochastic bond breaking modeled through damage rate:
  //   dD/dt = k_off * exp(F * x_beta / kT)
  // where k_off is the unstressed off-rate and x_beta is the
  // reactive compliance (barrier width).
  //
  // For deterministic MPM (no Brownian motion), we use a
  // simplified version: damage accumulates proportional to
  // force magnitude above a threshold.
  //
  // NOTE: damage is NOT updated here; it is advanced in
  // update_damage_and_break() so that it is modified exactly
  // once per timestep, consistent with the other CZ laws.
  static void receptor_ligand(double delta_n, double delta_t,
                              const CZParams& p,
                              double& T_n, double& T_t)
  {
    // Linear spring up to sigma_max, then softening
    double k_n = p.sigma_max / p.delta_n;
    double k_t = p.tau_max / p.delta_t;

    T_n = k_n * delta_n;
    T_t = k_t * delta_t;

    // Cap at maximum traction
    if (T_n > p.sigma_max) T_n = p.sigma_max;
  }

  // ============================================================
  // Pack/unpack for MPI migration.
  //
  // When particle i migrates, all bonds where tag_i matches
  // must travel with it. We pack bond data into the particle's
  // exchange buffer. On the receiving rank, bonds are unpacked
  // and added to the local bond list.
  //
  // Returns number of doubles packed/unpacked.
  // ============================================================

  // How many bonds does particle with global tag own?
  int count_bonds(tagint gtag) const {
    int count = 0;
    for (const auto& b : bonds)
      if (b.active && b.tag_i == gtag) count++;
    return count;
  }

  // Pack all bonds for particle with global tag into buffer.
  // Uses ubuf union to safely bit-cast 64-bit tagint into double.
  // Returns number of doubles written.
  int pack_bonds(tagint gtag, double* buf) const {
    // Local union for safe 64-bit integer ↔ double bit-cast
    union ubuf { double d; tagint t; };
    union lbuf { double d; long l; };
    int m = 0;
    for (const auto& b : bonds) {
      if (!b.active || b.tag_i != gtag) continue;
      ubuf u; lbuf lb;
      u.t = b.tag_i; buf[m++] = u.d;
      u.t = b.tag_j; buf[m++] = u.d;
      u.t = b.body_i; buf[m++] = u.d;
      u.t = b.body_j; buf[m++] = u.d;
      buf[m++] = (double)b.type_i;
      buf[m++] = (double)b.type_j;
      for (int d = 0; d < 3; d++) buf[m++] = b.x0_i[d];
      for (int d = 0; d < 3; d++) buf[m++] = b.x0_j[d];
      for (int d = 0; d < 3; d++) buf[m++] = b.n0[d];
      buf[m++] = b.delta0;
      buf[m++] = b.A_ref;
      buf[m++] = b.damage;
      buf[m++] = b.delta_max_n;
      buf[m++] = b.delta_max_t;
      lb.l = b.step_formed; buf[m++] = lb.d;
    }
    return m;
  }
  static constexpr int DOUBLES_PER_BOND = 21;

  // Unpack bonds from buffer. Returns number of doubles read.
  int unpack_bonds(const double* buf, int nbonds) {
    // Local union for safe 64-bit integer ↔ double bit-cast
    union ubuf { double d; tagint t; };
    union lbuf { double d; long l; };
    int m = 0;
    for (int b = 0; b < nbonds; b++) {
      CohesiveBond bond;
      ubuf u; lbuf lb;
      u.d = buf[m++]; bond.tag_i = u.t;
      u.d = buf[m++]; bond.tag_j = u.t;
      u.d = buf[m++]; bond.body_i = u.t;
      u.d = buf[m++]; bond.body_j = u.t;
      bond.type_i = (int)buf[m++];
      bond.type_j = (int)buf[m++];
      for (int d = 0; d < 3; d++) bond.x0_i[d] = buf[m++];
      for (int d = 0; d < 3; d++) bond.x0_j[d] = buf[m++];
      for (int d = 0; d < 3; d++) bond.n0[d] = buf[m++];
      bond.delta0 = buf[m++];
      bond.A_ref = buf[m++];
      bond.damage = buf[m++];
      bond.delta_max_n = buf[m++];
      bond.delta_max_t = buf[m++];
      lb.d = buf[m++]; bond.step_formed = lb.l;
      bond.active = true;
      bonds.push_back(bond);
    }
    return m;
  }

  // Remove all bonds owned by particle with global tag
  // (called on sending rank after pack)
  void remove_bonds(tagint gtag) {
    bonds.erase(
        std::remove_if(bonds.begin(), bonds.end(),
                       [gtag](const CohesiveBond& b) {
                         return b.active && b.tag_i == gtag;
                       }),
        bonds.end());
  }

  // ============================================================
  // Diagnostics
  // ============================================================
  double total_cohesive_energy() const {
    double E = 0;
    for (const auto& b : bonds) {
      if (!b.active) continue;
      const CZParams& prm = get_params(b.type_i, b.type_j);
      // Work of separation: phi_n = e * sigma_max * delta_n
      double phi = std::exp(1.0) * prm.sigma_max * prm.delta_n;
      E += phi * b.A_ref * b.damage;
    }
    return E;
  }

  int count_active() const {
    int n = 0;
    for (const auto& b : bonds) if (b.active) n++;
    return n;
  }

  // Count bonds between specific body pair
  int count_body_pair(int body_a, int body_b) const {
    int n = 0;
    for (const auto& b : bonds) {
      if (!b.active) continue;
      if ((b.body_i == body_a && b.body_j == body_b) ||
          (b.body_i == body_b && b.body_j == body_a))
        n++;
    }
    return n;
  }
};

} // namespace CoupMPM
} // namespace LAMMPS_NS

#endif // COUPMPM_COHESIVE_H
