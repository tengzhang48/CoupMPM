/* ----------------------------------------------------------------------
   CoupMPM - Material Point Method Package for LAMMPS

   fix coupmpm group-ID coupmpm &
       grid dx dy dz &
       kernel {linear | bspline2 | bspline3} &
       bbar {yes | no} &
       constitutive neohookean mu val kappa val &
       dt_auto {yes | no} &
       energy_check {yes | no} &
       cfl val &
       rho0 val

   Contact, cohesive zones, adaptivity, and VTK output are handled
   by optional companion fixes:
       fix coupmpm/contact, fix coupmpm/cohesive,
       fix coupmpm/adaptivity, fix coupmpm/output
---------------------------------------------------------------------- */

#include "fix_coupmpm.h"
#include "fix_coupmpm_contact.h"
#include "fix_coupmpm_cohesive.h"
#include "fix_coupmpm_adaptivity.h"
#include "fix_coupmpm_output.h"
#include "atom_vec_mpm.h"
#include "atom.h"
#include "update.h"
#include "domain.h"
#include "comm.h"
#include "memory.h"
#include "error.h"
#include "modify.h"
#include <cstring>
#include <cmath>
#include <cstdio>

using namespace LAMMPS_NS;
using namespace CoupMPM;

/* ---------------------------------------------------------------------- */

FixCoupMPM::FixCoupMPM(LAMMPS *lmp, int narg, char **arg)
  : Fix(lmp, narg, arg),
    avec(nullptr),
    L_buffer(nullptr), div_v_buf(nullptr), mass_p(nullptr),
    step_count(0),
    fix_contact(nullptr), fix_cohesive(nullptr),
    fix_adaptivity(nullptr), fix_output(nullptr),
    use_bardenhagen_contact(false),
    grid_dx(0.1), grid_dy(0.1), grid_dz(0.1),
    dim(3), use_bbar(true), rho0(1000.0),
    dt_auto(true), energy_check(false), cfl_factor(0.3),
    Nx_global(0), Ny_global(0), Nz_global(0),
    nmax_alloc(0)
{
  if (narg < 6) error->all(FLERR, "Illegal fix coupmpm command");

  time_integrate = 1;
  scalar_flag = 1;
  global_freq = 1;
  extscalar = 1;

  parse_args(narg, arg);
  dim = domain->dimension;
}

/* ---------------------------------------------------------------------- */

FixCoupMPM::~FixCoupMPM()
{
  memory->destroy(L_buffer);
  memory->destroy(div_v_buf);
  memory->destroy(mass_p);
}

/* ---------------------------------------------------------------------- */

int FixCoupMPM::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixCoupMPM::parse_args(int narg, char **arg)
{
  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "grid") == 0) {
      if (iarg + 3 >= narg) error->all(FLERR, "fix coupmpm: grid needs dx dy dz");
      grid_dx = atof(arg[iarg+1]);
      grid_dy = atof(arg[iarg+2]);
      grid_dz = atof(arg[iarg+3]);
      iarg += 4;
    }
    else if (strcmp(arg[iarg], "kernel") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix coupmpm: kernel needs type");
      if (strcmp(arg[iarg+1], "linear") == 0)
        kernel = MPMKernel(KernelType::LINEAR);
      else if (strcmp(arg[iarg+1], "bspline2") == 0)
        kernel = MPMKernel(KernelType::BSPLINE2);
      else if (strcmp(arg[iarg+1], "bspline3") == 0)
        kernel = MPMKernel(KernelType::BSPLINE3);
      else error->all(FLERR, "fix coupmpm: unknown kernel type");
      iarg += 2;
    }
    else if (strcmp(arg[iarg], "bbar") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix coupmpm: bbar needs argument");
      use_bbar = (strcmp(arg[iarg+1], "yes") == 0);
      iarg += 2;
    }
    else if (strcmp(arg[iarg], "constitutive") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix coupmpm: constitutive needs model type");
      if (strcmp(arg[iarg+1], "neohookean") == 0) {
        double mu = 0, kappa = 0;
        for (int k = iarg + 2; k < narg - 1; k++) {
          if (strcmp(arg[k], "mu") == 0) mu = atof(arg[k+1]);
          if (strcmp(arg[k], "kappa") == 0) kappa = atof(arg[k+1]);
        }
        if (mu <= 0 || kappa <= 0)
          error->all(FLERR, "fix coupmpm: neohookean needs mu > 0 and kappa > 0");
        stress_model = std::make_unique<NeoHookean>(mu, kappa);
        iarg += 6; // constitutive neohookean mu VAL kappa VAL
      }
      else if (strcmp(arg[iarg+1], "mooneyrivlin") == 0) {
        double C1 = 0, C2 = 0, kappa = 0;
        for (int k = iarg + 2; k < narg - 1; k++) {
          if (strcmp(arg[k], "C1") == 0) C1 = atof(arg[k+1]);
          if (strcmp(arg[k], "C2") == 0) C2 = atof(arg[k+1]);
          if (strcmp(arg[k], "kappa") == 0) kappa = atof(arg[k+1]);
        }
        if (C1 <= 0 || kappa <= 0)
          error->all(FLERR, "fix coupmpm: mooneyrivlin needs C1 > 0 and kappa > 0");
        stress_model = std::make_unique<MooneyRivlin>(C1, C2, kappa);
        iarg += 8; // constitutive mooneyrivlin C1 VAL C2 VAL kappa VAL
      }
      else error->all(FLERR, "fix coupmpm: unknown constitutive type");
    }
    else if (strcmp(arg[iarg], "dt_auto") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix coupmpm: dt_auto needs argument");
      dt_auto = (strcmp(arg[iarg+1], "yes") == 0); iarg += 2;
    }
    else if (strcmp(arg[iarg], "energy_check") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix coupmpm: energy_check needs argument");
      energy_check = (strcmp(arg[iarg+1], "yes") == 0); iarg += 2;
    }
    else if (strcmp(arg[iarg], "cfl") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix coupmpm: cfl needs argument");
      cfl_factor = atof(arg[iarg+1]); iarg += 2;
    }
    else if (strcmp(arg[iarg], "rho0") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix coupmpm: rho0 needs argument");
      rho0 = atof(arg[iarg+1]); iarg += 2;
    }
    // Provide helpful errors for keywords that moved to companion fixes
    else if (strcmp(arg[iarg], "contact") == 0) {
      error->all(FLERR, "fix coupmpm: 'contact' keyword moved to fix coupmpm/contact");
    }
    else if (strcmp(arg[iarg], "vtk_interval") == 0 ||
             strcmp(arg[iarg], "vtk_prefix") == 0 ||
             strcmp(arg[iarg], "surface_interval") == 0 ||
             strcmp(arg[iarg], "surface_alpha") == 0) {
      error->all(FLERR, "fix coupmpm: output keywords moved to fix coupmpm/output");
    }
    else if (strcmp(arg[iarg], "adaptivity") == 0 ||
             strcmp(arg[iarg], "J_split") == 0 ||
             strcmp(arg[iarg], "J_merge") == 0 ||
             strcmp(arg[iarg], "adapt_interval") == 0) {
      error->all(FLERR, "fix coupmpm: adaptivity keywords moved to fix coupmpm/adaptivity");
    }
    else if (strcmp(arg[iarg], "cohesive") == 0 ||
             strcmp(arg[iarg], "cz_law") == 0 ||
             strcmp(arg[iarg], "cz_sigma") == 0 ||
             strcmp(arg[iarg], "cz_delta") == 0 ||
             strcmp(arg[iarg], "cz_delta_max") == 0 ||
             strcmp(arg[iarg], "cz_form_dist") == 0 ||
             strcmp(arg[iarg], "cz_interval") == 0) {
      error->all(FLERR, "fix coupmpm: cohesive keywords moved to fix coupmpm/cohesive");
    }
    else {
      error->all(FLERR, fmt::format("fix coupmpm: unknown keyword '{}'", arg[iarg]));
    }
  }

  if (!stress_model) stress_model = std::make_unique<NeoHookean>(1e3, 1e4);
}

/* ---------------------------------------------------------------------- */

void FixCoupMPM::init()
{
  // Verify atom_style mpm
  avec = dynamic_cast<AtomVecMPM *>(atom->avec);
  if (!avec)
    error->all(FLERR, "fix coupmpm requires atom_style mpm");

  if (grid_dx <= 0 || grid_dy <= 0 || grid_dz <= 0)
    error->all(FLERR, "fix coupmpm: grid spacing must be positive");

  if (rho0 <= 0.0)
    error->all(FLERR, "fix coupmpm: rho0 must be positive");

  if (atom->natoms == 0)
    error->all(FLERR, "fix coupmpm: no atoms defined");
}

/* ---------------------------------------------------------------------- */

void FixCoupMPM::grow_work_arrays(int nmax)
{
  if (nmax <= nmax_alloc) return;
  memory->grow(L_buffer, nmax * 9, "coupmpm:L_buffer");
  memory->grow(div_v_buf, nmax, "coupmpm:div_v_buf");
  memory->grow(mass_p, nmax, "coupmpm:mass_p");
  nmax_alloc = nmax;
}

/* ---------------------------------------------------------------------- */

void FixCoupMPM::setup_grid()
{
  domain_lo[0] = domain->boxlo[0];
  domain_lo[1] = domain->boxlo[1];
  domain_lo[2] = domain->boxlo[2];
  domain_hi[0] = domain->boxhi[0];
  domain_hi[1] = domain->boxhi[1];
  domain_hi[2] = domain->boxhi[2];

  Nx_global = (int)std::ceil((domain_hi[0] - domain_lo[0]) / grid_dx);
  Ny_global = (int)std::ceil((domain_hi[1] - domain_lo[1]) / grid_dy);
  Nz_global = (dim == 3) ?
    (int)std::ceil((domain_hi[2] - domain_lo[2]) / grid_dz) : 1;

  int *myloc = comm->myloc;
  int *procgrid = comm->procgrid;

  int nx_local = Nx_global / procgrid[0];
  int ny_local = Ny_global / procgrid[1];
  int nz_local = (dim == 3) ? (Nz_global / procgrid[2]) : 1;

  int ox = myloc[0] * nx_local;
  int oy = myloc[1] * ny_local;
  int oz = (dim == 3) ? (myloc[2] * nz_local) : 0;

  if (myloc[0] == procgrid[0] - 1) nx_local = Nx_global - ox;
  if (myloc[1] == procgrid[1] - 1) ny_local = Ny_global - oy;
  if (dim == 3 && myloc[2] == procgrid[2] - 1) nz_local = Nz_global - oz;

  int ghost_w = kernel.ghost_width();

  grid.allocate(dim, nx_local, ny_local, nz_local,
                grid_dx, grid_dy, grid_dz,
                ghost_w, ox, oy, oz,
                Nx_global, Ny_global, Nz_global,
                use_bardenhagen_contact);

  if (comm->me == 0) {
    if (screen) {
      fprintf(screen, "CoupMPM: global grid %d x %d x %d, dx=%.4e\n",
              Nx_global, Ny_global, Nz_global, grid_dx);
      fprintf(screen, "CoupMPM: kernel=%s ghost=%d bbar=%s\n",
              (kernel.type == KernelType::LINEAR)   ? "linear" :
              (kernel.type == KernelType::BSPLINE2) ? "bspline2" : "bspline3",
              ghost_w, use_bbar ? "yes" : "no");
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixCoupMPM::setup_mpi()
{
  ghost_exchange.set_comm(world);

  int pn[3][2];
  for (int d = 0; d < 3; d++) {
    pn[d][0] = comm->procneigh[d][0];
    pn[d][1] = comm->procneigh[d][1];
  }
  ghost_exchange.set_neighbors(pn);
  ghost_exchange.set_periodic(domain->periodicity[0],
                              domain->periodicity[1],
                              domain->periodicity[2]);
  ghost_exchange.set_nprocs(comm->procgrid[0],
                            comm->procgrid[1],
                            comm->procgrid[2]);
  ghost_exchange.allocate_buffers(grid);
}

/* ---------------------------------------------------------------------- */

void FixCoupMPM::setup(int /*vflag*/)
{
  setup_grid();
  setup_mpi();
  grow_work_arrays(atom->nmax);

  // Cache particle masses from per-type mass
  int nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; i++)
    mass_p[i] = atom->mass[atom->type[i]];

  // If vol0 was not set in data file, estimate from grid spacing
  for (int i = 0; i < nlocal; i++) {
    if (avec->vol0[i] <= 0.0)
      avec->vol0[i] = grid.cell_volume();
  }

  if (comm->me == 0 && screen)
    fprintf(screen, "CoupMPM: setup complete, " BIGINT_FORMAT " atoms\n",
            atom->natoms);
}

/* ---------------------------------------------------------------------- */
// Steps 1-3: P2G → reverse comm → grid solve + contact
/* ---------------------------------------------------------------------- */

void FixCoupMPM::initial_integrate(int /*vflag*/)
{
  grow_work_arrays(atom->nmax);

  int nlocal = atom->nlocal;
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;

  // Refresh mass cache
  if (fix_adaptivity && fix_adaptivity->adaptivity_enabled()) {
    for (int i = 0; i < nlocal; i++)
      mass_p[i] = avec->vol0[i] * rho0;
  } else {
    for (int i = 0; i < nlocal; i++)
      mass_p[i] = atom->mass[atom->type[i]];
  }

  // Auto timestep: always update dt to the CFL-stable value so the
  // timestep can both decrease and recover as the simulation evolves.
  if (dt_auto) {
    double dt_cfl = compute_dt_cfl();
    if (comm->me == 0 && screen && dt_cfl != update->dt)
      fprintf(screen, "CoupMPM: CFL dt=%.4e (was %.4e)\n",
              dt_cfl, update->dt);
    update->dt = dt_cfl;
  }

  double **F_def    = avec->F_def;
  double **stress_v = avec->stress_v;
  double *vol0      = avec->vol0;
  double **Bp_arr   = avec->Bp;

  double *F_flat      = (nlocal > 0) ? &F_def[0][0]    : nullptr;
  double *stress_flat = (nlocal > 0) ? &stress_v[0][0] : nullptr;
  double *Bp_flat     = (nlocal > 0) ? &Bp_arr[0][0]   : nullptr;

  const double dt = update->dt;

  // --- Step 0b: Cohesive zone forces (before P2G) ---
  if (fix_cohesive)
    fix_cohesive->compute_forces_before_p2g();

  // --- Step 1: P2G ---
  grid.zero_grid();
  if (fix_contact) fix_contact->pre_p2g(grid);

  p2g_records.clear();
  p2g_records.reserve(nlocal);

  p2g(grid, kernel, nlocal,
      x, v, f, mass_p, vol0,
      F_flat, stress_flat, Bp_flat,
      atom->molecule,
      domain_lo, use_bbar,
      &p2g_records,
      atom->tag);

  // --- Step 2: Reverse ghost communication ---
  ghost_exchange.reverse_comm(grid);

  if (use_bbar)
    grid.normalize_div_v();

  // --- Step 3: Grid solve + contact ---
  grid.grid_solve(dt);

  if (fix_contact)
    fix_contact->post_grid_solve(grid, dt, world);
}

/* ---------------------------------------------------------------------- */
// Steps 4-6: forward comm → G2P → F update → position update
/* ---------------------------------------------------------------------- */

void FixCoupMPM::final_integrate()
{
  int nlocal = atom->nlocal;
  double **x = atom->x;
  double **v = atom->v;
  const double dt = update->dt;

  double **Bp_arr = avec->Bp;
  double *Bp_flat = (nlocal > 0) ? &Bp_arr[0][0] : nullptr;

  // --- Step 4: Forward ghost communication ---
  ghost_exchange.forward_comm(grid);

  // --- Step 5: G2P ---
  g2p(grid, kernel, nlocal,
      x, v, Bp_flat, L_buffer, div_v_buf,
      domain_lo, dt, use_bbar);

  // --- Step 5b: Update F and stress ---
  double *F_flat      = (nlocal > 0) ? &avec->F_def[0][0]    : nullptr;
  double *stress_flat = (nlocal > 0) ? &avec->stress_v[0][0] : nullptr;
  double *state_flat  = (nlocal > 0) ? &avec->mpm_state[0][0]: nullptr;

  update_F_and_stress(
      nlocal, F_flat, stress_flat,
      L_buffer, use_bbar ? div_v_buf : nullptr,
      state_flat, *stress_model, dt, use_bbar, dim);

  // --- Step 5c: Update positions ---
  for (int i = 0; i < nlocal; i++) {
    x[i][0] += dt * v[i][0];
    x[i][1] += dt * v[i][1];
    if (dim == 3) x[i][2] += dt * v[i][2];
  }

  // --- Step 6: Anti-P2G for migrating particles ---
  {
    int n_migrated = 0;

    for (int i = 0; i < nlocal; i++) {
      bool outside = false;
      for (int d = 0; d < dim; d++) {
        if (x[i][d] < domain->sublo[d] || x[i][d] >= domain->subhi[d]) {
          outside = true;
          break;
        }
      }
      if (outside && i < (int)p2g_records.size()) {
        // Verify the record matches this atom; guard against mid-step reordering.
        // If the tags don't match (e.g., after atom->sort()), we skip the anti-P2G
        // for this atom to avoid subtracting a different particle's contributions.
        // The migrating particle's contributions remain in the grid for this step,
        // which introduces a one-step grid residual that is corrected next step.
        if (p2g_records[i].global_tag == atom->tag[i]) {
          anti_p2g(grid, kernel, p2g_records[i], domain_lo, use_bbar);
          n_migrated++;
        }
      }
    }

    if (n_migrated > 0)
      ghost_exchange.reverse_comm(grid);
  }

  step_count++;
}

/* ---------------------------------------------------------------------- */

double FixCoupMPM::compute_dt_cfl()
{
  double c_max = stress_model->wave_speed(rho0);
  double dx_min = std::min(grid_dx, std::min(grid_dy, grid_dz));
  return cfl_factor * dx_min / c_max;
}
