/* ----------------------------------------------------------------------
   CoupMPM - fix coupmpm/cohesive

   Companion fix to fix coupmpm.  Manages dynamic cohesive zone bonds.

   Usage:
     fix ID group-ID coupmpm/cohesive          &
         law {needleman|linear|receptor}        &
         sigma <val>                            &
         delta <val>                            &
         delta_max <val>                        &
         form_dist <val>                        &
         interval <N>

   Requires fix coupmpm to be defined first.

   Lifecycle in each timestep:
     initial_integrate (parent):
       parent calls this->compute_forces_before_p2g()
         → compute cohesive forces on local+ghost atoms
         → fix-specific reverse comm (ghost forces → owners)
         → forces enter grid through P2G external force path
     end_of_step (this fix):
       → detect new bonds (every interval steps)
       → update damage, break failed bonds (every step)
     atom migration:
       → pack_exchange/unpack_exchange move bonds with particles
       → only lower-tag owner packs each bond (prevents duplication)

   Audit fixes incorporated:
     [1] maxexchange set in constructor (prevents buffer overrun)
     [2] Fix-specific reverse comm via pack_reverse_comm/unpack_reverse_comm
     [3] Neighbor list cutoff set to form_dist
     [4] Lower-tag-owner convention prevents bond duplication
     [5] Parent pointer nulled in destructor
     [6] grow_arrays / copy_arrays implemented (empty — bonds in vector)
     [7] static_cast after style check (avoids -fno-rtti issue)
---------------------------------------------------------------------- */

#include "fix_coupmpm_cohesive.h"
#include "fix_coupmpm.h"
#include "atom_vec_mpm.h"
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "update.h"
#include <cstring>
#include <cmath>
#include <cstdio>

using namespace LAMMPS_NS;
using namespace CoupMPM;

/* ---------------------------------------------------------------------- */

FixCoupMPMCohesive::FixCoupMPMCohesive(LAMMPS *lmp, int narg, char **arg)
  : Fix(lmp, narg, arg),
    parent(nullptr), list(nullptr),
    cz_sigma_tmp(100.0), cz_delta_tmp(1e-4),
    cz_delta_max_tmp(2e-4), cz_form_dist_tmp(5e-4)
{
  if (narg < 3)
    error->all(FLERR, "Illegal fix coupmpm/cohesive command");

  cohesive.enabled = true;
  parse_args(narg, arg);

  // ---------------------------------------------------------------
  // FIX [1]: Set maxexchange so LAMMPS allocates enough buffer
  // for bond data during atom migration.
  //
  // Per bond we pack: 6 ints-as-doubles (tags, body_ids, types)
  //                 + 9 doubles (x0_i, x0_j, n0)
  //                 + 6 doubles (delta0, A_ref, damage, delta_max_n/t, step)
  //                 = 21 doubles per bond
  // Plus 1 double for the bond count integer.
  //
  // Without this, the exchange buffer is sized 0 for this fix
  // and the first migrating particle with bonds causes a segfault.
  // ---------------------------------------------------------------
  static constexpr int PACK_DOUBLES_PER_BOND = 21;
  maxexchange = 1 + cohesive.max_bonds_per_particle * PACK_DOUBLES_PER_BOND;

  // ---------------------------------------------------------------
  // FIX [2]: Declare fix-specific reverse communication size.
  //
  // After computing cohesive forces, ghost atoms hold force
  // contributions that must be sent back to their owners.
  // comm->reverse_comm(this) uses these pack/unpack methods
  // with a buffer of comm_reverse doubles per atom.
  //
  // We communicate f[0], f[1], f[2] = 3 doubles.
  // ---------------------------------------------------------------
  comm_reverse = 3;
  comm_forward = 0;  // we don't need forward comm
}

/* ---------------------------------------------------------------------- */

FixCoupMPMCohesive::~FixCoupMPMCohesive()
{
  // FIX [5]: Null the parent's pointer to us so it doesn't
  // call a destroyed object if this fix is unfixed at runtime.
  if (parent) parent->fix_cohesive = nullptr;

  // Unregister atom callbacks to prevent use-after-free.
  // In modern LAMMPS, delete_callback takes (id, type).
  // Type 0 = GROW callback. Some versions also need EXCHANGE.
  atom->delete_callback(id, 0);
}

/* ---------------------------------------------------------------------- */

int FixCoupMPMCohesive::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixCoupMPMCohesive::parse_args(int narg, char **arg)
{
  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "law") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "fix coupmpm/cohesive: law needs argument");
      if (strcmp(arg[iarg+1], "needleman") == 0)
        cohesive.law_type = CZLawType::NEEDLEMAN_XU;
      else if (strcmp(arg[iarg+1], "linear") == 0)
        cohesive.law_type = CZLawType::LINEAR_ELASTIC;
      else if (strcmp(arg[iarg+1], "receptor") == 0)
        cohesive.law_type = CZLawType::RECEPTOR_LIGAND;
      else
        error->all(FLERR, "fix coupmpm/cohesive: unknown law type");
      iarg += 2;
    }
    else if (strcmp(arg[iarg], "sigma") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "fix coupmpm/cohesive: sigma needs value");
      cz_sigma_tmp = atof(arg[iarg+1]);
      iarg += 2;
    }
    else if (strcmp(arg[iarg], "delta") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "fix coupmpm/cohesive: delta needs value");
      cz_delta_tmp = atof(arg[iarg+1]);
      iarg += 2;
    }
    else if (strcmp(arg[iarg], "delta_max") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "fix coupmpm/cohesive: delta_max needs value");
      cz_delta_max_tmp = atof(arg[iarg+1]);
      iarg += 2;
    }
    else if (strcmp(arg[iarg], "form_dist") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "fix coupmpm/cohesive: form_dist needs value");
      cz_form_dist_tmp = atof(arg[iarg+1]);
      iarg += 2;
    }
    else if (strcmp(arg[iarg], "interval") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "fix coupmpm/cohesive: interval needs value");
      cohesive.bond_check_interval = atoi(arg[iarg+1]);
      iarg += 2;
    }
    else {
      error->all(FLERR, fmt::format(
        "fix coupmpm/cohesive: unknown keyword '{}'", arg[iarg]));
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixCoupMPMCohesive::init()
{
  // -------------------------------------------------------------------
  // Find the parent fix coupmpm.
  //
  // FIX [7]: Use static_cast after style string check instead of
  // dynamic_cast. This avoids -fno-rtti compilation issues on some
  // HPC clusters. The string check guarantees the type is correct.
  // -------------------------------------------------------------------
  parent = nullptr;
  for (int i = 0; i < modify->nfix; i++) {
    if (strcmp(modify->fix[i]->style, "coupmpm") == 0) {
      parent = static_cast<FixCoupMPM *>(modify->fix[i]);
      break;
    }
  }
  if (!parent)
    error->all(FLERR,
      "fix coupmpm/cohesive requires fix coupmpm to be defined first");

  // Register ourselves with the parent so it can call
  // compute_forces_before_p2g() at the right time.
  parent->fix_cohesive = this;

  // -------------------------------------------------------------------
  // Register atom callback for bond migration during exchange.
  //
  // Idempotent: delete any prior registration before re-registering,
  // so repeated init() calls (e.g., from run ... run ...) don't
  // double-register and cause pack_exchange to fire twice per atom.
  // -------------------------------------------------------------------
  atom->delete_callback(id, 0);
  atom->add_callback(0);   // type 0 = GROW (also enables exchange)

  // -------------------------------------------------------------------
  // FIX [3]: Request a half neighbor list with cutoff = form_dist.
  //
  // Without setting the cutoff, the list uses the max pair_style
  // cutoff. In a pure MPM simulation with no pair_style (using
  // Bardenhagen grid contact), the default cutoff is 0 and no
  // neighbors are found. Setting it explicitly ensures bond
  // detection works regardless of other pair interactions.
  // -------------------------------------------------------------------
  int irequest = neighbor->request(this, instance_me);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->fix  = 1;
  neighbor->requests[irequest]->half = 1;

  // Set custom cutoff for this neighbor list.
  // The cutoff must be at least form_dist so that all potential
  // bond partners appear in the list.
  // Add a small skin to avoid missing pairs at exactly form_dist.
  neighbor->requests[irequest]->cut = 1;
  neighbor->requests[irequest]->cutoff = cz_form_dist_tmp * 1.1;
}

/* ---------------------------------------------------------------------- */

void FixCoupMPMCohesive::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void FixCoupMPMCohesive::setup(int /*vflag*/)
{
  // Initialize type-pair cohesive parameters.
  // For now, all type pairs get the same parameters.
  // Use cohesive.set_pair_params(ti, tj, params) for differential adhesion.
  cohesive.init_params(atom->ntypes,
                       cz_sigma_tmp, cz_sigma_tmp,    // sigma_n, sigma_t
                       cz_delta_tmp, cz_delta_tmp,    // delta_n, delta_t
                       cz_delta_max_tmp, cz_delta_max_tmp,
                       cz_form_dist_tmp);

  if (comm->me == 0 && screen)
    fprintf(screen,
      "CoupMPM cohesive: law=%s sigma=%.4e delta=%.4e "
      "delta_max=%.4e form_dist=%.4e interval=%d\n",
      (cohesive.law_type == CZLawType::NEEDLEMAN_XU) ? "needleman" :
      (cohesive.law_type == CZLawType::LINEAR_ELASTIC) ? "linear" : "receptor",
      cz_sigma_tmp, cz_delta_tmp, cz_delta_max_tmp, cz_form_dist_tmp,
      cohesive.bond_check_interval);
}

/* ======================================================================
   Cohesive force computation — called by parent before P2G.

   Computes traction-separation forces for all active bonds and
   writes them to atom->f[]. Ghost atoms may receive forces if
   the bond partner is on this rank.

   After force computation, fix-specific reverse communication
   sends ghost forces back to their owning ranks.
   ====================================================================== */

void FixCoupMPMCohesive::compute_forces_before_p2g()
{
  if (!cohesive.enabled) return;
  if (!parent || !parent->avec) return;

  int nlocal = atom->nlocal;
  double **F_def = parent->avec->F_def;
  double *F_flat = (nlocal > 0 && F_def) ? &F_def[0][0] : nullptr;

  // Compute cohesive forces: writes to atom->f[i] for both
  // local atoms (i < nlocal) and ghost atoms (i >= nlocal).
  cohesive.compute_forces(
      nlocal, atom->nghost,
      atom->x, atom->f, atom->tag, F_flat,
      parent->dim, update->dt);

  // -------------------------------------------------------------------
  // FIX [2]: Fix-specific reverse communication.
  //
  // Ghost atoms that received cohesive forces need those forces
  // sent back to their owning rank and added to the owner's f[].
  //
  // comm->reverse_comm(this) calls our pack_reverse_comm() on
  // ghost atoms and unpack_reverse_comm() on owners, using
  // the comm_reverse=3 size declared in the constructor.
  //
  // This replaces the incorrect bare comm->reverse_comm() which
  // only handles Atom-class forces, not fix forces.
  // -------------------------------------------------------------------
  comm->reverse_comm(this);
}

/* ======================================================================
   FIX [2]: Fix-specific reverse communication methods.

   pack_reverse_comm: called on ghost atoms — packs their forces.
   unpack_reverse_comm: called on owner atoms — ADDS ghost forces.

   This ensures cohesive forces computed on ghost atoms at subdomain
   boundaries are properly accumulated onto their owners.
   ====================================================================== */

int FixCoupMPMCohesive::pack_reverse_comm(int n, int first, double *buf)
{
  int m = 0;
  int last = first + n;
  for (int i = first; i < last; i++) {
    buf[m++] = atom->f[i][0];
    buf[m++] = atom->f[i][1];
    buf[m++] = atom->f[i][2];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixCoupMPMCohesive::unpack_reverse_comm(int n, int *list, double *buf)
{
  int m = 0;
  for (int i = 0; i < n; i++) {
    int j = list[i];
    atom->f[j][0] += buf[m++];  // ADDITIVE — accumulate, don't overwrite
    atom->f[j][1] += buf[m++];
    atom->f[j][2] += buf[m++];
  }
}

/* ======================================================================
   End-of-step: detect new bonds + update damage + break failed bonds.
   ====================================================================== */

void FixCoupMPMCohesive::end_of_step()
{
  if (!cohesive.enabled) return;
  if (!parent || !parent->avec) return;

  long step = parent->step_count;
  int nlocal = atom->nlocal;
  double **F_def = parent->avec->F_def;
  double *F_flat = (nlocal > 0 && F_def) ? &F_def[0][0] : nullptr;
  double dx_min = std::min(parent->grid.dx,
                           std::min(parent->grid.dy, parent->grid.dz));

  // --- Detect new bonds (every bond_check_interval steps) ---
  if (cohesive.bond_check_interval > 0 &&
      step > 0 &&
      step % cohesive.bond_check_interval == 0)
  {
    int n_new = cohesive.detect_new_bonds(
        nlocal, atom->nghost,
        atom->x, atom->tag, atom->type,
        atom->molecule, parent->avec->surface,
        F_flat, parent->avec->vol0,
        step, parent->dim, dx_min);

    if (comm->me == 0 && screen && n_new > 0)
      fprintf(screen,
        "CoupMPM cohesive: step %ld, %d new bonds, %d total active\n",
        step, n_new, cohesive.count_active());
  }

  // --- Update damage and break failed bonds (every step) ---
  cohesive.update_damage_and_break(
      nlocal, atom->x, atom->tag, parent->dim);

  if (comm->me == 0 && screen && cohesive.n_broken_last > 0)
    fprintf(screen,
      "CoupMPM cohesive: step %ld, %d bonds broken, %d remaining\n",
      step, cohesive.n_broken_last, cohesive.count_active());
}

/* ======================================================================
   Bond cleanup for adaptivity.

   Called by fix coupmpm/adaptivity when a particle is deleted during
   merging. Deactivates all bonds involving the given atom tag to
   prevent dangling references.

   The adaptivity fix finds us via:
     modify->find_fix_by_style("coupmpm/cohesive")
   ====================================================================== */

void FixCoupMPMCohesive::deactivate_bonds_for_tag(tagint tag)
{
  for (auto &b : cohesive.bonds)
    if (b.active && (b.tag_i == tag || b.tag_j == tag))
      b.active = false;
}

/* ======================================================================
   Atom callback methods for bond migration during exchange.

   When a particle crosses an MPI subdomain boundary, LAMMPS calls
   pack_exchange on the sending rank and unpack_exchange on the
   receiving rank. We serialize/deserialize the particle's bonds.

   FIX [4]: Only the lower-tag owner packs each bond.
   If a bond has tag_i=5, tag_j=3, only the rank holding tag_j=3
   (the lower tag) packs this bond. This prevents duplication when
   both endpoints migrate in the same step.
   ====================================================================== */

void FixCoupMPMCohesive::grow_arrays(int /*nmax*/)
{
  // FIX [6]: Bonds are stored in a std::vector, not per-atom arrays.
  // Nothing to grow. This method must exist because we registered
  // an atom callback.
}

/* ---------------------------------------------------------------------- */

void FixCoupMPMCohesive::copy_arrays(int i, int j, int /*delflag*/)
{
  // When LAMMPS compacts the atom array (e.g., after deletion),
  // it copies atom j into slot i. Since bonds are stored by global
  // tag (not local index), no action is needed here — the tags
  // don't change during copy, only the local index does.
  (void)i; (void)j;
}

/* ---------------------------------------------------------------------- */

int FixCoupMPMCohesive::pack_exchange(int i, double *buf)
{
  // Pack all bonds owned by this atom for migration.
  //
  // FIX [4]: Ownership convention — a bond is "owned" by the
  // particle with the lower global tag. This prevents duplication:
  // if both endpoints migrate simultaneously, only one of them
  // (the lower-tag one) packs the bond. The higher-tag endpoint
  // packs zero bonds for this pair.
  //
  // After packing, remove the packed bonds from the local list.

  int m = 0;
  tagint my_tag = atom->tag[i];

  // Count bonds where this atom is the lower-tag owner
  int nbonds = 0;
  for (const auto &b : cohesive.bonds) {
    if (!b.active) continue;
    // Bond involves this atom AND this atom is the lower tag
    if (b.tag_i == my_tag && my_tag <= b.tag_j) nbonds++;
    if (b.tag_j == my_tag && my_tag < b.tag_i) nbonds++;
    // Note: tag_i < tag_j uses <=, tag_j < tag_i uses <
    // to ensure exactly one endpoint owns each bond
  }

  // Pack count (as double via ubuf)
  union { double d; int64_t i; } ubuf;
  ubuf.i = nbonds;
  buf[m++] = ubuf.d;

  // Pack bond data
  for (const auto &b : cohesive.bonds) {
    if (!b.active) continue;
    bool owns = false;
    if (b.tag_i == my_tag && my_tag <= b.tag_j) owns = true;
    if (b.tag_j == my_tag && my_tag < b.tag_i) owns = true;
    if (!owns) continue;

    buf[m++] = (double)b.tag_i;
    buf[m++] = (double)b.tag_j;
    buf[m++] = (double)b.body_i;
    buf[m++] = (double)b.body_j;
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
    buf[m++] = (double)b.step_formed;
  }

  // Remove packed bonds from local list
  // (they now live on the receiving rank)
  cohesive.bonds.erase(
    std::remove_if(cohesive.bonds.begin(), cohesive.bonds.end(),
      [my_tag](const CohesiveBond &b) {
        if (!b.active) return false;
        if (b.tag_i == my_tag && my_tag <= b.tag_j) return true;
        if (b.tag_j == my_tag && my_tag < b.tag_i) return true;
        return false;
      }),
    cohesive.bonds.end());

  return m;
}

/* ---------------------------------------------------------------------- */

int FixCoupMPMCohesive::unpack_exchange(int /*nlocal_new*/, double *buf)
{
  int m = 0;

  // Read bond count
  union { double d; int64_t i; } ubuf;
  ubuf.d = buf[m++];
  int nbonds = (int)ubuf.i;

  // Unpack bonds
  for (int b = 0; b < nbonds; b++) {
    CohesiveBond bond;
    bond.tag_i = (int)buf[m++];
    bond.tag_j = (int)buf[m++];
    bond.body_i = (int)buf[m++];
    bond.body_j = (int)buf[m++];
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
    bond.step_formed = (long)buf[m++];
    bond.active = true;

    cohesive.bonds.push_back(bond);
  }

  return m;
}
