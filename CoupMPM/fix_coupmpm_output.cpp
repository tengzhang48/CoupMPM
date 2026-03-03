/* ----------------------------------------------------------------------
   CoupMPM - fix coupmpm/output

   fix ID group-ID coupmpm/output      \
       vtk_interval <N>                \
       vtk_prefix <prefix>             \
       surface_interval <N>            \
       surface_alpha <val>

   Companion fix to fix coupmpm.  Handles VTK output and surface
   detection in end_of_step().
---------------------------------------------------------------------- */

#include "fix_coupmpm_output.h"
#include "fix_coupmpm.h"
#include "fix_coupmpm_contact.h"
#include "atom_vec_mpm.h"
#include "atom.h"
#include "comm.h"
#include "error.h"
#include "modify.h"
#include "update.h"
#include <cstring>
#include <cmath>
#include <cstdio>

using namespace LAMMPS_NS;
using namespace CoupMPM;

/* ---------------------------------------------------------------------- */

FixCoupMPMOutput::FixCoupMPMOutput(LAMMPS *lmp, int narg, char **arg)
  : Fix(lmp, narg, arg),
    parent(nullptr),
    surface_detector(0.1),
    vtk_interval(0), surface_interval(10),
    vtk_prefix("coupmpm"), surface_alpha(0.1)
{
  if (narg < 3) error->all(FLERR, "Illegal fix coupmpm/output command");
  parse_args(narg, arg);
  surface_detector = SurfaceDetector(surface_alpha);
}

/* ---------------------------------------------------------------------- */

int FixCoupMPMOutput::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixCoupMPMOutput::parse_args(int narg, char **arg)
{
  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "vtk_interval") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix coupmpm/output: vtk_interval needs argument");
      vtk_interval = atoi(arg[iarg+1]); iarg += 2;
    }
    else if (strcmp(arg[iarg], "vtk_prefix") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix coupmpm/output: vtk_prefix needs argument");
      vtk_prefix = arg[iarg+1]; iarg += 2;
    }
    else if (strcmp(arg[iarg], "surface_interval") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix coupmpm/output: surface_interval needs argument");
      surface_interval = atoi(arg[iarg+1]); iarg += 2;
    }
    else if (strcmp(arg[iarg], "surface_alpha") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix coupmpm/output: surface_alpha needs argument");
      surface_alpha = atof(arg[iarg+1]); iarg += 2;
    }
    else {
      error->all(FLERR,
        fmt::format("fix coupmpm/output: unknown keyword '{}'", arg[iarg]));
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixCoupMPMOutput::init()
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
      "fix coupmpm/output: fix coupmpm must be defined before fix coupmpm/output");

  parent->fix_output = this;
}

/* ---------------------------------------------------------------------- */

void FixCoupMPMOutput::setup(int /*vflag*/)
{
  // Re-apply surface_alpha in case it was updated after construction
  surface_detector = SurfaceDetector(surface_alpha);
}

/* ---------------------------------------------------------------------- */

void FixCoupMPMOutput::end_of_step()
{
  long cur_step = parent->step_count;
  AtomVecMPM *avec = parent->avec;

  // --- VTK output ---
  if (vtk_interval > 0 && cur_step % vtk_interval == 0) {
    parent->grid.compute_density();

    MPMIO::write_grid_vtk(parent->grid, world, cur_step,
                          vtk_prefix, parent->domain_lo);

    int nlocal = atom->nlocal;
    double *F_flat      = (nlocal > 0) ? &avec->F_def[0][0]    : nullptr;
    double *stress_flat = (nlocal > 0) ? &avec->stress_v[0][0] : nullptr;

    MPMIO::write_particle_vtk(
        nlocal, parent->dim, atom->x, atom->v,
        stress_flat, F_flat,
        atom->molecule, avec->surface,
        world, cur_step, vtk_prefix);

    vtk_steps.push_back(cur_step);
    if (comm->me == 0)
      MPMIO::write_pvd(vtk_prefix + "_grid.pvd", vtk_prefix,
                       vtk_steps, update->dt);
  }

  // --- Surface detection via ∇ρ ---
  if (surface_interval > 0 && cur_step % surface_interval == 0) {
    surface_detector.compute_grid_gradient(parent->grid);

    surface_detector.detect_surface(
        parent->grid, parent->kernel, atom->nlocal,
        atom->x, avec->surface,
        parent->domain_lo, world);

    // Update area_scale for penalty (Nanson) contact if present
    if (parent->fix_contact && parent->fix_contact->is_penalty()) {
      int nlocal = atom->nlocal;
      double *F_flat = (nlocal > 0) ? &avec->F_def[0][0] : nullptr;
      SurfaceDetector::update_area_scale(
          nlocal, F_flat, avec->surface,
          avec->area0, avec->area_scale);
    }

    // Diagnostic: count surface particles
    {
      int n_surf = 0;
      for (int i = 0; i < atom->nlocal; i++)
        if (avec->surface[i]) n_surf++;
      int n_surf_global = 0;
      MPI_Reduce(&n_surf, &n_surf_global, 1, MPI_INT, MPI_SUM, 0, world);
      if (comm->me == 0 && screen && n_surf_global > 0)
        fprintf(screen,
          "CoupMPM: step %ld, %d surface particles detected\n",
          cur_step, n_surf_global);
    }
  }
}
