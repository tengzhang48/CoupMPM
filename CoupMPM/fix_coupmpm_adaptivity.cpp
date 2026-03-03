/* ----------------------------------------------------------------------
   CoupMPM - fix coupmpm/adaptivity

   fix ID group-ID coupmpm/adaptivity \
       J_split <val>                  \
       J_merge <val>                  \
       interval <N>

   Companion fix to fix coupmpm.  Performs particle splitting and
   merging in end_of_step() based on the local Jacobian J = det(F).
---------------------------------------------------------------------- */

#include "fix_coupmpm_adaptivity.h"
#include "fix_coupmpm.h"
#include "fix_coupmpm_cohesive.h"
#include "atom_vec_mpm.h"
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "memory.h"
#include "modify.h"
#include "update.h"
#include <cstring>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <vector>

using namespace LAMMPS_NS;
using namespace CoupMPM;

/* ---------------------------------------------------------------------- */

FixCoupMPMAdaptivity::FixCoupMPMAdaptivity(LAMMPS *lmp, int narg, char **arg)
  : Fix(lmp, narg, arg), parent(nullptr)
{
  if (narg < 3) error->all(FLERR, "Illegal fix coupmpm/adaptivity command");
  adaptivity.enabled = true;
  parse_args(narg, arg);
}

/* ---------------------------------------------------------------------- */

int FixCoupMPMAdaptivity::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixCoupMPMAdaptivity::parse_args(int narg, char **arg)
{
  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "J_split") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix coupmpm/adaptivity: J_split needs argument");
      adaptivity.J_split_hi = atof(arg[iarg+1]); iarg += 2;
    }
    else if (strcmp(arg[iarg], "J_merge") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix coupmpm/adaptivity: J_merge needs argument");
      adaptivity.J_split_lo = atof(arg[iarg+1]); iarg += 2;
    }
    else if (strcmp(arg[iarg], "interval") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix coupmpm/adaptivity: interval needs argument");
      adaptivity.check_interval = atoi(arg[iarg+1]); iarg += 2;
    }
    else {
      error->all(FLERR,
        fmt::format("fix coupmpm/adaptivity: unknown keyword '{}'", arg[iarg]));
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixCoupMPMAdaptivity::init()
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
      "fix coupmpm/adaptivity: fix coupmpm must be defined before fix coupmpm/adaptivity");

  parent->fix_adaptivity = this;
}

/* ---------------------------------------------------------------------- */

void FixCoupMPMAdaptivity::end_of_step()
{
  if (!adaptivity.enabled) return;
  if (adaptivity.check_interval <= 0) return;

  long step = parent->step_count;
  if (step == 0) return;
  if (step % adaptivity.check_interval != 0) return;

  AtomVecMPM *avec = parent->avec;
  int nlocal = atom->nlocal;
  double *F_flat = (nlocal > 0) ? &avec->F_def[0][0] : nullptr;
  double dx_min  = std::min(parent->grid.dx,
                            std::min(parent->grid.dy, parent->grid.dz));

  // --- Splitting ---
  auto split_list = adaptivity.find_split_candidates(
      nlocal, parent->dim, F_flat, avec->vol0, dx_min);

  int n_splits = 0;
  for (int idx = 0; idx < (int)split_list.size(); idx++) {
    int p = split_list[idx];
    if (p >= atom->nlocal) continue;

    double xp[3] = {atom->x[p][0], atom->x[p][1], atom->x[p][2]};
    double vp[3] = {atom->v[p][0], atom->v[p][1], atom->v[p][2]};
    double mp = parent->mass_p[p];

    auto children = adaptivity.generate_children(
        parent->dim, xp, vp, mp, avec->vol0[p],
        avec->F_def[p], avec->stress_v[p],
        avec->Bp[p], avec->mpm_state[p],
        AtomVecMPM::N_STATE,
        atom->molecule[p], atom->type[p]);

    bool first = true;
    for (const auto &child : children) {
      if (first) {
        // Overwrite parent in-place with first child
        atom->x[p][0] = child.x[0];
        atom->x[p][1] = child.x[1];
        atom->x[p][2] = child.x[2];
        atom->v[p][0] = child.v[0];
        atom->v[p][1] = child.v[1];
        atom->v[p][2] = child.v[2];
        avec->vol0[p] = child.vol0;
        std::memcpy(avec->F_def[p],   child.F_def,   9 * sizeof(double));
        std::memcpy(avec->stress_v[p],child.stress_v, 6 * sizeof(double));
        std::memcpy(avec->Bp[p],      child.Bp,       9 * sizeof(double));
        std::memcpy(avec->mpm_state[p],child.state,
                    AtomVecMPM::N_STATE * sizeof(double));
        parent->mass_p[p] = child.mass;
        first = false;
      } else {
        // Create new atom at end of local array
        int n = atom->nlocal;
        if (n == atom->nmax) avec->grow(0);
        parent->grow_work_arrays(atom->nmax);

        atom->x[n][0] = child.x[0];
        atom->x[n][1] = child.x[1];
        atom->x[n][2] = child.x[2];
        atom->v[n][0] = child.v[0];
        atom->v[n][1] = child.v[1];
        atom->v[n][2] = child.v[2];
        atom->f[n][0] = atom->f[n][1] = atom->f[n][2] = 0.0;
        atom->tag[n]      = 0;  // reset by tag_extend()
        atom->type[n]     = child.type;
        atom->mask[n]     = 1;
        atom->image[n]    = atom->image[p];
        atom->molecule[n] = child.body_id;
        // Remap position and image flags in case the child crossed a
        // periodic boundary relative to the parent particle.
        domain->remap(atom->x[n], atom->image[n]);

        avec->vol0[n] = child.vol0;
        std::memcpy(avec->F_def[n],   child.F_def,   9 * sizeof(double));
        std::memcpy(avec->stress_v[n],child.stress_v, 6 * sizeof(double));
        std::memcpy(avec->Bp[n],      child.Bp,       9 * sizeof(double));
        std::memcpy(avec->mpm_state[n],child.state,
                    AtomVecMPM::N_STATE * sizeof(double));
        avec->surface[n]     = 0;
        avec->area0[n][0]    = avec->area0[n][1] = avec->area0[n][2] = 0.0;
        avec->area_scale[n]  = 0.0;

        parent->mass_p[n] = child.mass;

        atom->nlocal++;
      }
    }
    n_splits++;
  }

  // --- Merging ---
  // Re-fetch nlocal and F_flat after potential splits
  nlocal = atom->nlocal;
  F_flat = (nlocal > 0) ? &avec->F_def[0][0] : nullptr;

  auto merge_list = adaptivity.find_merge_candidates(
      nlocal, parent->dim, atom->x, F_flat,
      atom->molecule, dx_min);

  int n_merges = 0;
  std::vector<int> to_delete;

  for (const auto &mp : merge_list) {
    int i = mp.i, j = mp.j;
    if (i >= atom->nlocal || j >= atom->nlocal) continue;

    auto merged = ParticleAdaptivity::merge_particles(
        atom->x[i], atom->v[i], parent->mass_p[i], avec->vol0[i],
        avec->F_def[i], avec->stress_v[i],
        avec->Bp[i], avec->mpm_state[i],
        atom->x[j], atom->v[j], parent->mass_p[j], avec->vol0[j],
        avec->F_def[j], avec->stress_v[j],
        avec->Bp[j], avec->mpm_state[j],
        AtomVecMPM::N_STATE,
        atom->molecule[i], atom->type[i]);

    // Overwrite i with merged result
    atom->x[i][0] = merged.x[0];
    atom->x[i][1] = merged.x[1];
    atom->x[i][2] = merged.x[2];
    atom->v[i][0] = merged.v[0];
    atom->v[i][1] = merged.v[1];
    atom->v[i][2] = merged.v[2];
    avec->vol0[i] = merged.vol0;
    std::memcpy(avec->F_def[i],   merged.F_def,   9 * sizeof(double));
    std::memcpy(avec->stress_v[i],merged.stress_v, 6 * sizeof(double));
    std::memcpy(avec->Bp[i],      merged.Bp,       9 * sizeof(double));
    std::memcpy(avec->mpm_state[i],merged.state,
                AtomVecMPM::N_STATE * sizeof(double));
    parent->mass_p[i] = merged.mass;

    to_delete.push_back(j);
    n_merges++;
  }

  // Remove deleted particles (sort descending to preserve indices)
  std::sort(to_delete.rbegin(), to_delete.rend());
  for (int del_idx : to_delete) {
    // Notify cohesive fix so dangling bond references are cleaned up
    if (parent->fix_cohesive && del_idx < atom->nlocal)
      parent->fix_cohesive->deactivate_bonds_for_tag(atom->tag[del_idx]);

    if (del_idx < atom->nlocal - 1)
      avec->copy(atom->nlocal - 1, del_idx, 1);
    atom->nlocal--;
  }

  // Fix up global state after adding/removing atoms
  if (n_splits > 0 || n_merges > 0) {
    atom->tag_extend();
    atom->natoms = 0;
    bigint nlocal_big = atom->nlocal;
    MPI_Allreduce(&nlocal_big, &atom->natoms, 1,
                  MPI_LMP_BIGINT, MPI_SUM, world);
    if (atom->map_style != Atom::MAP_NONE) {
      atom->map_init();
      atom->map_set();
    }
  }

  adaptivity.n_splits_last = n_splits;
  adaptivity.n_merges_last = n_merges;

  if (comm->me == 0 && screen && (n_splits > 0 || n_merges > 0))
    fprintf(screen,
      "CoupMPM: step %ld, adaptivity: %d splits, %d merges, "
      BIGINT_FORMAT " total atoms\n",
      step, n_splits, n_merges, atom->natoms);
}
