/* ----------------------------------------------------------------------
   CoupMPM - fix coupmpm/cohesive

   fix ID group-ID coupmpm/cohesive               \
       law {needleman|linear|receptor}             \
       sigma <val>                                 \
       delta <val>                                 \
       delta_max <val>                             \
       form_dist <val>                             \
       interval <N>

   Companion fix to fix coupmpm.  Manages dynamic cohesive zone bonds:
     - compute_forces_before_p2g(): called by parent before P2G
     - end_of_step():               detect new bonds + update damage
     - pack/unpack_exchange():      migrate bonds with particles
---------------------------------------------------------------------- */

#include "fix_coupmpm_cohesive.h"
#include "fix_coupmpm.h"
#include "atom_vec_mpm.h"
#include "atom.h"
#include "comm.h"
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
  if (narg < 3) error->all(FLERR, "Illegal fix coupmpm/cohesive command");
  cohesive.enabled = true;
  parse_args(narg, arg);
}

/* ---------------------------------------------------------------------- */

FixCoupMPMCohesive::~FixCoupMPMCohesive()
{
  // Unregister pack_exchange / unpack_exchange callbacks so that the atom
  // machinery doesn't call a deleted fix after unfix.
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
      cz_sigma_tmp = atof(arg[iarg+1]); iarg += 2;
    }
    else if (strcmp(arg[iarg], "delta") == 0) {
      cz_delta_tmp = atof(arg[iarg+1]); iarg += 2;
    }
    else if (strcmp(arg[iarg], "delta_max") == 0) {
      cz_delta_max_tmp = atof(arg[iarg+1]); iarg += 2;
    }
    else if (strcmp(arg[iarg], "form_dist") == 0) {
      cz_form_dist_tmp = atof(arg[iarg+1]); iarg += 2;
    }
    else if (strcmp(arg[iarg], "interval") == 0) {
      cohesive.bond_check_interval = atoi(arg[iarg+1]); iarg += 2;
    }
    else {
      error->all(FLERR,
        fmt::format("fix coupmpm/cohesive: unknown keyword '{}'", arg[iarg]));
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixCoupMPMCohesive::init()
{
  // Find the parent coupmpm fix
  parent = nullptr;
  for (int i = 0; i < modify->nfix; i++) {
    if (strcmp(modify->fix[i]->style, "coupmpm") == 0) {
      parent = dynamic_cast<FixCoupMPM *>(modify->fix[i]);
      break;
    }
  }
  if (!parent)
    error->all(FLERR,
      "fix coupmpm/cohesive: fix coupmpm must be defined before fix coupmpm/cohesive");

  parent->fix_cohesive = this;

  // Ensure idempotent registration: remove any prior callback before
  // re-registering, so that repeated init() calls don't double-register
  // and cause pack_exchange/unpack_exchange to be invoked twice per atom.
  atom->delete_callback(id, 0);
  atom->add_callback(0);

  // Request a half neighbor list for bond detection
  int irequest = neighbor->request(this, instance_me);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->fix  = 1;
  neighbor->requests[irequest]->half = 1;
}

/* ---------------------------------------------------------------------- */

void FixCoupMPMCohesive::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void FixCoupMPMCohesive::setup(int /*vflag*/)
{
  cohesive.init_params(atom->ntypes,
                       cz_sigma_tmp, cz_sigma_tmp,
                       cz_delta_tmp, cz_delta_tmp,
                       cz_delta_max_tmp, cz_delta_max_tmp,
                       cz_form_dist_tmp);

  if (comm->me == 0 && screen)
    fprintf(screen,
      "CoupMPM cohesive: sigma=%.4e, delta=%.4e, form_dist=%.4e\n",
      cz_sigma_tmp, cz_delta_tmp, cz_form_dist_tmp);
}

/* ---------------------------------------------------------------------- */

void FixCoupMPMCohesive::compute_forces_before_p2g()
{
  if (!cohesive.enabled) return;

  int nlocal = atom->nlocal;
  double *F_flat = (nlocal > 0) ? &parent->avec->F_def[0][0] : nullptr;

  cohesive.compute_forces(
      nlocal, atom->nghost,
      atom->x, atom->f, atom->tag, F_flat,
      parent->dim,
      atom);

  // Reverse comm: send ghost forces back to their owning ranks
  comm->reverse_comm();
}

/* ---------------------------------------------------------------------- */

void FixCoupMPMCohesive::end_of_step()
{
  if (!cohesive.enabled) return;

  long step = parent->step_count;
  int nlocal = atom->nlocal;
  double *F_flat = (nlocal > 0) ? &parent->avec->F_def[0][0] : nullptr;
  double dx_min  = std::min(parent->grid.dx,
                            std::min(parent->grid.dy, parent->grid.dz));

  // Detect new bonds (every bond_check_interval steps)
  if (cohesive.bond_check_interval > 0 &&
      step % cohesive.bond_check_interval == 0) {
    int n_new = cohesive.detect_new_bonds(
        nlocal, atom->nghost,
        atom->x, atom->tag, atom->type,
        atom->molecule, parent->avec->surface,
        F_flat, parent->avec->vol0,
        step, parent->dim, dx_min, list);

    if (comm->me == 0 && screen && n_new > 0)
      fprintf(screen,
        "CoupMPM: step %ld, %d new cohesive bonds formed, %d total active\n",
        step, n_new, cohesive.count_active());
  }

  // Update damage and break failed bonds (every step)
  cohesive.update_damage_and_break(nlocal, atom->x, atom->tag,
                                   parent->dim, atom, update->dt);

  if (comm->me == 0 && screen && cohesive.n_broken_last > 0)
    fprintf(screen,
      "CoupMPM: step %ld, %d cohesive bonds broken, %d remaining\n",
      step, cohesive.n_broken_last, cohesive.count_active());
}

/* ---------------------------------------------------------------------- */

void FixCoupMPMCohesive::deactivate_bonds_for_tag(tagint tag)
{
  for (auto &b : cohesive.bonds)
    if (b.active && (b.tag_i == tag || b.tag_j == tag))
      b.active = false;
}

/* ---------------------------------------------------------------------- */

int FixCoupMPMCohesive::pack_exchange(int i, double *buf)
{
  int m = 0;
  union ubuf { double d; tagint t; } u;
  int nbonds = cohesive.count_bonds(atom->tag[i]);
  u.t = (tagint)nbonds;
  buf[m++] = u.d;
  m += cohesive.pack_bonds(atom->tag[i], buf + m);
  cohesive.remove_bonds(atom->tag[i]);
  return m;
}

/* ---------------------------------------------------------------------- */

int FixCoupMPMCohesive::unpack_exchange(int /*nlocal*/, double *buf)
{
  int m = 0;
  union ubuf { double d; tagint t; } u;
  u.d = buf[m++];
  int nbonds = (int)u.t;
  m += cohesive.unpack_bonds(buf + m, nbonds);
  return m;
}
