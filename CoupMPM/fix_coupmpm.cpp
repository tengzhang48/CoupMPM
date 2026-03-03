/* ----------------------------------------------------------------------
   CoupMPM - Material Point Method Package for LAMMPS

   fix coupmpm group-ID coupmpm &
       grid dx dy dz &
       kernel {linear | bspline2 | bspline3} &
       bbar {yes | no} &
       contact {none | bardenhagen [mu val] [adhesion val] | penalty} &
       constitutive neohookean mu val kappa val &
       dt_auto {yes | no} &
       energy_check {yes | no} &
       vtk_interval N &
       vtk_prefix prefix &
       surface_interval N &
       cfl val &
       rho0 val
---------------------------------------------------------------------- */

#include "fix_coupmpm.h"
#include "atom_vec_mpm.h"
#include "coupmpm_surface.h"
#include "coupmpm_adaptivity.h"
#include "coupmpm_cohesive.h"
#include "atom.h"
#include "update.h"
#include "domain.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "error.h"
#include "modify.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include <cstring>
#include <cmath>
#include <cstdio>

using namespace LAMMPS_NS;
using namespace CoupMPM;

/* ---------------------------------------------------------------------- */

FixCoupMPM::FixCoupMPM(LAMMPS *lmp, int narg, char **arg)
  : Fix(lmp, narg, arg),
    avec(nullptr),
    list(nullptr),
    L_buffer(nullptr), div_v_buf(nullptr), mass_p(nullptr),
    grid_dx(0.1), grid_dy(0.1), grid_dz(0.1),
    dim(3), use_bbar(true), dt_auto(true), energy_check(false),
    vtk_interval(0), surface_interval(10),
    vtk_prefix("coupmpm"), cfl_factor(0.3), rho0(1000.0),
    surface_alpha(0.1),
    cz_sigma_tmp(100.0), cz_delta_tmp(1e-4),
    cz_delta_max_tmp(2e-4), cz_form_dist_tmp(5e-4),
    Nx_global(0), Ny_global(0), Nz_global(0),
    step_count(0), nmax_alloc(0)
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
  if (cohesive.enabled) atom->delete_callback(id, 0);
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
  mask |= END_OF_STEP;
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
      use_bbar = (strcmp(arg[iarg+1], "yes") == 0);
      iarg += 2;
    }
    else if (strcmp(arg[iarg], "contact") == 0) {
      if (strcmp(arg[iarg+1], "none") == 0) {
        contact_model = std::make_unique<ContactNone>();
        iarg += 2;
      }
      else if (strcmp(arg[iarg+1], "bardenhagen") == 0) {
        auto *bard = new ContactBardenhagen();
        int sub_start = iarg + 2;
        int sub_end = sub_start;
        // Scan forward for contact sub-args until next top-level keyword
        const char *top_keys[] = {"constitutive","dt_auto","energy_check",
                                  "vtk_interval","vtk_prefix","bbar",
                                  "surface_interval","cfl","rho0",nullptr};
        while (sub_end < narg) {
          bool is_top = false;
          for (int k = 0; top_keys[k]; k++)
            if (strcmp(arg[sub_end], top_keys[k]) == 0) { is_top = true; break; }
          if (is_top) break;
          sub_end++;
        }
        bard->init(sub_end - sub_start, &arg[sub_start]);
        contact_model.reset(bard);
        iarg = sub_end;
      }
      else if (strcmp(arg[iarg+1], "penalty") == 0) {
        contact_model = std::make_unique<ContactPenalty>();
        iarg += 2;
      }
      else error->all(FLERR, "fix coupmpm: unknown contact type");
    }
    else if (strcmp(arg[iarg], "constitutive") == 0) {
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
        stress_model = std::make_unique<MooneyRivlin>(
            atof(arg[iarg+2]), atof(arg[iarg+3]), atof(arg[iarg+4]));
        iarg += 5;
      }
      else error->all(FLERR, "fix coupmpm: unknown constitutive type");
    }
    else if (strcmp(arg[iarg], "dt_auto") == 0) {
      dt_auto = (strcmp(arg[iarg+1], "yes") == 0); iarg += 2;
    }
    else if (strcmp(arg[iarg], "energy_check") == 0) {
      energy_check = (strcmp(arg[iarg+1], "yes") == 0); iarg += 2;
    }
    else if (strcmp(arg[iarg], "vtk_interval") == 0) {
      vtk_interval = atoi(arg[iarg+1]); iarg += 2;
    }
    else if (strcmp(arg[iarg], "vtk_prefix") == 0) {
      vtk_prefix = arg[iarg+1]; iarg += 2;
    }
    else if (strcmp(arg[iarg], "surface_interval") == 0) {
      surface_interval = atoi(arg[iarg+1]); iarg += 2;
    }
    else if (strcmp(arg[iarg], "surface_alpha") == 0) {
      surface_alpha = atof(arg[iarg+1]); iarg += 2;
    }
    else if (strcmp(arg[iarg], "cfl") == 0) {
      cfl_factor = atof(arg[iarg+1]); iarg += 2;
    }
    else if (strcmp(arg[iarg], "rho0") == 0) {
      rho0 = atof(arg[iarg+1]); iarg += 2;
    }
    else if (strcmp(arg[iarg], "adaptivity") == 0) {
      if (strcmp(arg[iarg+1], "yes") == 0) adaptivity.enabled = true;
      else adaptivity.enabled = false;
      iarg += 2;
    }
    else if (strcmp(arg[iarg], "J_split") == 0) {
      adaptivity.J_split_hi = atof(arg[iarg+1]);
      iarg += 2;
    }
    else if (strcmp(arg[iarg], "J_merge") == 0) {
      adaptivity.J_split_lo = atof(arg[iarg+1]);
      iarg += 2;
    }
    else if (strcmp(arg[iarg], "adapt_interval") == 0) {
      adaptivity.check_interval = atoi(arg[iarg+1]);
      iarg += 2;
    }
    else if (strcmp(arg[iarg], "cohesive") == 0) {
      if (strcmp(arg[iarg+1], "yes") == 0) cohesive.enabled = true;
      else cohesive.enabled = false;
      iarg += 2;
    }
    else if (strcmp(arg[iarg], "cz_law") == 0) {
      if (strcmp(arg[iarg+1], "needleman") == 0)
        cohesive.law_type = CoupMPM::CZLawType::NEEDLEMAN_XU;
      else if (strcmp(arg[iarg+1], "linear") == 0)
        cohesive.law_type = CoupMPM::CZLawType::LINEAR_ELASTIC;
      else if (strcmp(arg[iarg+1], "receptor") == 0)
        cohesive.law_type = CoupMPM::CZLawType::RECEPTOR_LIGAND;
      iarg += 2;
    }
    else if (strcmp(arg[iarg], "cz_sigma") == 0) {
      cz_sigma_tmp = atof(arg[iarg+1]); iarg += 2;
    }
    else if (strcmp(arg[iarg], "cz_delta") == 0) {
      cz_delta_tmp = atof(arg[iarg+1]); iarg += 2;
    }
    else if (strcmp(arg[iarg], "cz_delta_max") == 0) {
      cz_delta_max_tmp = atof(arg[iarg+1]); iarg += 2;
    }
    else if (strcmp(arg[iarg], "cz_form_dist") == 0) {
      cz_form_dist_tmp = atof(arg[iarg+1]); iarg += 2;
    }
    else if (strcmp(arg[iarg], "cz_interval") == 0) {
      cohesive.bond_check_interval = atoi(arg[iarg+1]); iarg += 2;
    }
    else {
      error->all(FLERR, fmt::format("fix coupmpm: unknown keyword '{}'", arg[iarg]));
    }
  }

  if (!stress_model)  stress_model  = std::make_unique<NeoHookean>(1e3, 1e4);
  if (!contact_model) contact_model = std::make_unique<ContactNone>();
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

  if (atom->natoms == 0)
    error->all(FLERR, "fix coupmpm: no atoms defined");

  // Register pack/unpack_exchange hooks for cohesive bond migration
  // and request a half neighbor list for cohesive zone bond detection
  if (cohesive.enabled) {
    atom->add_callback(0);
    int irequest = neighbor->request(this, instance_me);
    neighbor->requests[irequest]->pair = 0;
    neighbor->requests[irequest]->fix = 1;
    neighbor->requests[irequest]->half = 1;
  }
}

/* ---------------------------------------------------------------------- */

void FixCoupMPM::init_list(int /*id*/, NeighList *ptr) { list = ptr; }

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
  bool bard = (contact_model && strcmp(contact_model->name(), "bardenhagen") == 0);

  grid.allocate(dim, nx_local, ny_local, nz_local,
                grid_dx, grid_dy, grid_dz,
                ghost_w, ox, oy, oz,
                Nx_global, Ny_global, Nz_global, bard);

  if (comm->me == 0) {
    if (screen) {
      fprintf(screen, "CoupMPM: global grid %d x %d x %d, dx=%.4e\n",
              Nx_global, Ny_global, Nz_global, grid_dx);
      fprintf(screen, "CoupMPM: kernel=%s ghost=%d bbar=%s contact=%s\n",
              (kernel.type == KernelType::LINEAR)   ? "linear" :
              (kernel.type == KernelType::BSPLINE2) ? "bspline2" : "bspline3",
              ghost_w, use_bbar ? "yes" : "no",
              contact_model->name());
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

  // Initialize surface detector
  surface_detector = SurfaceDetector(surface_alpha);

  // Initialize cohesive zone manager
  if (cohesive.enabled) {
    cohesive.init_params(atom->ntypes, cz_sigma_tmp, cz_sigma_tmp,
                         cz_delta_tmp, cz_delta_tmp,
                         cz_delta_max_tmp, cz_delta_max_tmp,
                         cz_form_dist_tmp);
    if (comm->me == 0 && screen)
      fprintf(screen, "CoupMPM: cohesive zones enabled, sigma=%.4e, delta=%.4e, "
              "form_dist=%.4e\n",
              cz_sigma_tmp, cz_delta_tmp, cz_form_dist_tmp);
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
  // For particles that haven't been split, mass = per-type mass.
  // For split particles, mass = vol0 * rho0 (vol0 was properly divided
  // during splitting and is communicated via atom_vec_mpm).
  // To ensure consistency, always compute from vol0 * rho0 when adaptivity
  // is enabled. Otherwise use per-type mass.
  if (adaptivity.enabled) {
    for (int i = 0; i < nlocal; i++)
      mass_p[i] = avec->vol0[i] * rho0;
  } else {
    for (int i = 0; i < nlocal; i++)
      mass_p[i] = atom->mass[atom->type[i]];
  }

  // Auto timestep
  if (dt_auto) {
    double dt_cfl = compute_dt_cfl();
    if (dt_cfl < update->dt) {
      if (comm->me == 0 && screen)
        fprintf(screen, "CoupMPM: CFL dt=%.4e < current dt=%.4e\n",
                dt_cfl, update->dt);
      update->dt = dt_cfl;
    }
  }

  // Access atom_vec arrays
  double **F_def   = avec->F_def;
  double **stress_v = avec->stress_v;
  double *vol0     = avec->vol0;
  double **Bp_arr  = avec->Bp;

  // Flatten 2D arrays to 1D for transfer functions
  // (transfer.h expects double* with flat [nlocal*9] layout)
  // Since memory->grow gives contiguous 2D arrays, F_def[0] is the flat pointer.
  double *F_flat     = (nlocal > 0) ? &F_def[0][0]   : nullptr;
  double *stress_flat= (nlocal > 0) ? &stress_v[0][0] : nullptr;
  double *Bp_flat    = (nlocal > 0) ? &Bp_arr[0][0]   : nullptr;

  const double dt = update->dt;

  // --- Step 0b: Cohesive zone forces (before P2G) ---
  // Forces are written to f[] and will be spread to grid during P2G.
  if (cohesive.enabled) {
    double *F_flat_cz = (nlocal > 0) ? &F_def[0][0] : nullptr;
    cohesive.compute_forces(
        nlocal, atom->nghost,
        x, f, atom->tag, F_flat_cz, dim, dt,
        atom);
    // Reverse comm: send ghost forces back to their owning ranks
    // so all cohesive contributions are included before P2G.
    comm->reverse_comm();
  }

  // --- Step 1: P2G ---
  grid.zero_grid();
  if (contact_model) contact_model->pre_p2g(grid);

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

  if (contact_model)
    contact_model->post_grid_solve(grid, dt);
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
      state_flat, *stress_model, dt, use_bbar);

  // --- Step 5c: Update positions ---
  for (int i = 0; i < nlocal; i++) {
    x[i][0] += dt * v[i][0];
    x[i][1] += dt * v[i][1];
    if (dim == 3) x[i][2] += dt * v[i][2];
  }

  // --- Step 6: Anti-P2G for migrating particles ---
  //
  // After position update, some particles may have moved outside
  // this rank's subdomain. Before LAMMPS migrates them (in
  // comm->exchange()), we must subtract their P2G contributions
  // from the local grid. Otherwise those contributions become
  // "orphaned" mass/momentum that leaks conservation.
  //
  // Protocol:
  //   1. Check which particles crossed subdomain boundaries
  //   2. For each migrating particle, call anti_p2g() with its
  //      stored P2GRecord to subtract contributions
  //   3. Do a second reverse comm to propagate subtractions
  //      from ghost nodes to their owners
  //
  // LAMMPS then calls comm->exchange() which invokes
  // atom_vec_mpm::pack_exchange/unpack_exchange, moving all
  // MPM fields with the particle. The receiving rank will
  // include this particle in its next P2G step normally.
  {
    int n_migrated = 0;
    const double lo[3] = {domain->sublo[0], domain->sublo[1], domain->sublo[2]};
    const double hi[3] = {domain->subhi[0], domain->subhi[1], domain->subhi[2]};

    for (int i = 0; i < nlocal; i++) {
      bool outside = false;
      for (int d = 0; d < dim; d++) {
        if (x[i][d] < lo[d] || x[i][d] >= hi[d]) {
          // Periodic: only flag if not wrapped
          if (!domain->periodicity[d] ||
              (x[i][d] < domain->boxlo[d] || x[i][d] >= domain->boxhi[d])) {
            outside = true;
            break;
          }
          // Even with periodicity, if nprocs > 1 and particle left
          // subdomain, it needs migration
          if (comm->procgrid[d] > 1) {
            outside = true;
            break;
          }
        }
      }
      if (outside && i < (int)p2g_records.size()) {
        anti_p2g(grid, kernel, p2g_records[i], domain_lo, use_bbar);
        n_migrated++;
      }
    }

    // Second reverse comm: propagate anti-P2G corrections from
    // ghost nodes to owners. This is needed because anti_p2g()
    // may have subtracted from ghost nodes on this rank.
    if (n_migrated > 0)
      ghost_exchange.reverse_comm(grid);
  }

  step_count++;
}

/* ---------------------------------------------------------------------- */

void FixCoupMPM::end_of_step()
{
  // VTK output
  if (vtk_interval > 0 && step_count % vtk_interval == 0) {
    grid.compute_density();

    MPMIO::write_grid_vtk(grid, world, step_count, vtk_prefix, domain_lo);

    int nlocal = atom->nlocal;
    double *F_flat     = (nlocal > 0) ? &avec->F_def[0][0]    : nullptr;
    double *stress_flat= (nlocal > 0) ? &avec->stress_v[0][0] : nullptr;

    MPMIO::write_particle_vtk(
        nlocal, dim, atom->x, atom->v,
        stress_flat, F_flat,
        atom->molecule, avec->surface,
        world, step_count, vtk_prefix);

    vtk_steps.push_back(step_count);
    if (comm->me == 0)
      MPMIO::write_pvd(vtk_prefix + "_grid.pvd", vtk_prefix,
                        vtk_steps, update->dt);
  }

  // --- Surface detection via ∇ρ ---
  if (surface_interval > 0 && step_count % surface_interval == 0) {
    // Step 1-2: Compute density and gradient on grid
    // (grid.mass is still populated from the most recent P2G)
    surface_detector.compute_grid_gradient(grid);

    // Step 3-4-5: Interpolate |∇ρ| to particles, threshold → surface flag
    surface_detector.detect_surface(
        grid, kernel, atom->nlocal,
        atom->x, avec->surface,
        domain_lo, world);

    // Optional: update area_scale for Nanson contact
    if (contact_model && strcmp(contact_model->name(), "penalty") == 0) {
      int nlocal = atom->nlocal;
      double *F_flat = (nlocal > 0) ? &avec->F_def[0][0] : nullptr;
      SurfaceDetector::update_area_scale(
          nlocal, F_flat, avec->surface,
          avec->area0, avec->area_scale);
    }

    // Diagnostic: count surface particles
    if (comm->me == 0 && screen) {
      int n_surf = 0;
      for (int i = 0; i < atom->nlocal; i++)
        if (avec->surface[i]) n_surf++;
      // Reduce across ranks for global count
      int n_surf_global = 0;
      MPI_Reduce(&n_surf, &n_surf_global, 1, MPI_INT, MPI_SUM, 0, world);
      if (n_surf_global > 0)
        fprintf(screen, "CoupMPM: step %ld, %d surface particles detected\n",
                step_count, n_surf_global);
    }
  }

  // --- Particle adaptivity: splitting and merging ---
  if (adaptivity.enabled &&
      adaptivity.check_interval > 0 &&
      step_count % adaptivity.check_interval == 0 &&
      step_count > 0)
  {
    int nlocal = atom->nlocal;
    double *F_flat = (nlocal > 0) ? &avec->F_def[0][0] : nullptr;
    double dx_min = std::min(grid_dx, std::min(grid_dy, grid_dz));

    // --- Splitting ---
    auto split_list = adaptivity.find_split_candidates(
        nlocal, dim, F_flat, avec->vol0, dx_min);

    int n_splits = 0;
    // Process splits in reverse order so indices stay valid
    // as we add atoms (new atoms go at end, existing indices unchanged)
    for (int idx = 0; idx < (int)split_list.size(); idx++) {
      int p = split_list[idx];
      if (p >= atom->nlocal) continue; // safety

      double xp[3] = {atom->x[p][0], atom->x[p][1], atom->x[p][2]};
      double vp[3] = {atom->v[p][0], atom->v[p][1], atom->v[p][2]};
      double mp = mass_p[p];

      auto children = adaptivity.generate_children(
          dim, xp, vp, mp, avec->vol0[p],
          avec->F_def[p], avec->stress_v[p],
          avec->Bp[p], avec->mpm_state[p],
          AtomVecMPM::N_STATE,
          atom->molecule[p], atom->type[p]);

      // Create child atoms — first child replaces parent, rest are new
      bool first = true;
      for (const auto& child : children) {
        if (first) {
          // Overwrite parent in-place with first child
          atom->x[p][0] = child.x[0];
          atom->x[p][1] = child.x[1];
          atom->x[p][2] = child.x[2];
          atom->v[p][0] = child.v[0];
          atom->v[p][1] = child.v[1];
          atom->v[p][2] = child.v[2];
          // Update mass: for per-type mass, we can't change it.
          // Instead, scale vol0 so that effective contribution is correct.
          // mass_p stays the same (per-type), but vol0 is divided.
          avec->vol0[p] = child.vol0;
          std::memcpy(avec->F_def[p], child.F_def, 9 * sizeof(double));
          std::memcpy(avec->stress_v[p], child.stress_v, 6 * sizeof(double));
          std::memcpy(avec->Bp[p], child.Bp, 9 * sizeof(double));
          std::memcpy(avec->mpm_state[p], child.state,
                      AtomVecMPM::N_STATE * sizeof(double));
          first = false;
        } else {
          // Create new atom at end of local array
          // This is the LAMMPS-native way to add atoms
          int n = atom->nlocal;
          if (n == atom->nmax) avec->grow(0);
          grow_work_arrays(atom->nmax);

          atom->x[n][0] = child.x[0];
          atom->x[n][1] = child.x[1];
          atom->x[n][2] = child.x[2];
          atom->v[n][0] = child.v[0];
          atom->v[n][1] = child.v[1];
          atom->v[n][2] = child.v[2];
          atom->f[n][0] = atom->f[n][1] = atom->f[n][2] = 0.0;
          atom->tag[n] = 0;  // Will be reset by atom->tag_extend()
          atom->type[n] = child.type;
          atom->mask[n] = 1;
          atom->image[n] = atom->image[p]; // inherit image flags
          atom->molecule[n] = child.body_id;

          avec->vol0[n] = child.vol0;
          std::memcpy(avec->F_def[n], child.F_def, 9 * sizeof(double));
          std::memcpy(avec->stress_v[n], child.stress_v, 6 * sizeof(double));
          std::memcpy(avec->Bp[n], child.Bp, 9 * sizeof(double));
          std::memcpy(avec->mpm_state[n], child.state,
                      AtomVecMPM::N_STATE * sizeof(double));
          avec->surface[n] = 0;
          avec->area0[n][0] = avec->area0[n][1] = avec->area0[n][2] = 0.0;
          avec->area_scale[n] = 0.0;

          mass_p[n] = mass_p[p];  // same type → same mass

          atom->nlocal++;
        }
      }
      n_splits++;
    }

    // --- Merging ---
    auto merge_list = adaptivity.find_merge_candidates(
        atom->nlocal, dim, atom->x, F_flat,
        atom->molecule, dx_min);

    int n_merges = 0;
    // Process merges: replace particle i with merged result, delete j
    // Collect deletion indices, then compact
    std::vector<int> to_delete;
    for (const auto& mp : merge_list) {
      int i = mp.i, j = mp.j;
      if (i >= atom->nlocal || j >= atom->nlocal) continue;

      auto merged = ParticleAdaptivity::merge_particles(
          atom->x[i], atom->v[i], mass_p[i], avec->vol0[i],
          avec->F_def[i], avec->stress_v[i],
          avec->Bp[i], avec->mpm_state[i],
          atom->x[j], atom->v[j], mass_p[j], avec->vol0[j],
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
      std::memcpy(avec->F_def[i], merged.F_def, 9 * sizeof(double));
      std::memcpy(avec->stress_v[i], merged.stress_v, 6 * sizeof(double));
      std::memcpy(avec->Bp[i], merged.Bp, 9 * sizeof(double));
      std::memcpy(avec->mpm_state[i], merged.state,
                  AtomVecMPM::N_STATE * sizeof(double));

      to_delete.push_back(j);
      n_merges++;
    }

    // Remove deleted particles by copying last atom into their slot
    // Sort deletions in descending order to avoid index invalidation
    std::sort(to_delete.rbegin(), to_delete.rend());
    for (int del_idx : to_delete) {
      // Mark any cohesive bonds containing the deleted atom's tag as inactive
      // to prevent dangling bond references after this particle is removed.
      if (cohesive.enabled && del_idx < atom->nlocal) {
        tagint del_tag = atom->tag[del_idx];
        for (auto& b : cohesive.bonds)
          if (b.active && (b.tag_i == del_tag || b.tag_j == del_tag))
            b.active = false;
      }
      if (del_idx < atom->nlocal - 1)
        avec->copy(atom->nlocal - 1, del_idx, 1);
      atom->nlocal--;
    }

    // Fix up global state after adding/removing atoms
    if (n_splits > 0 || n_merges > 0) {
      atom->tag_extend();    // assign tags to new atoms
      atom->natoms = 0;
      MPI_Allreduce(&atom->nlocal, &atom->natoms, 1,
                    MPI_LMP_BIGINT, MPI_SUM, world);
      if (atom->map_style != Atom::MAP_NONE) {
        atom->map_init();
        atom->map_set();
      }
    }

    adaptivity.n_splits_last = n_splits;
    adaptivity.n_merges_last = n_merges;

    if (comm->me == 0 && screen && (n_splits > 0 || n_merges > 0))
      fprintf(screen, "CoupMPM: step %ld, adaptivity: %d splits, %d merges, "
              BIGINT_FORMAT " total atoms\n",
              step_count, n_splits, n_merges, atom->natoms);
  }

  // --- Cohesive zone: detect new bonds + update damage ---
  if (cohesive.enabled) {
    int nlocal = atom->nlocal;
    double *F_flat = (nlocal > 0) ? &avec->F_def[0][0] : nullptr;
    double dx_min = std::min(grid_dx, std::min(grid_dy, grid_dz));

    // Detect new bonds (every bond_check_interval steps)
    if (cohesive.bond_check_interval > 0 &&
        step_count % cohesive.bond_check_interval == 0) {
      int n_new = cohesive.detect_new_bonds(
          nlocal, atom->nghost,
          atom->x, atom->tag, atom->type,
          atom->molecule, avec->surface,
          F_flat, avec->vol0,
          step_count, dim, dx_min, list);

      if (comm->me == 0 && screen && n_new > 0)
        fprintf(screen, "CoupMPM: step %ld, %d new cohesive bonds formed, "
                "%d total active\n",
                step_count, n_new, cohesive.count_active());
    }

    // Update damage and break failed bonds (every step)
    cohesive.update_damage_and_break(nlocal, atom->x, atom->tag, dim, atom);

    if (comm->me == 0 && screen && cohesive.n_broken_last > 0)
      fprintf(screen, "CoupMPM: step %ld, %d cohesive bonds broken, "
              "%d remaining\n",
              step_count, cohesive.n_broken_last, cohesive.count_active());
  }
}

/* ---------------------------------------------------------------------- */

double FixCoupMPM::compute_dt_cfl()
{
  double c_max = stress_model->wave_speed(rho0);
  double dx_min = std::min(grid_dx, std::min(grid_dy, grid_dz));
  return cfl_factor * dx_min / c_max;
}

/* ---------------------------------------------------------------------- */

int FixCoupMPM::pack_exchange(int i, double *buf)
{
  int m = 0;
  if (cohesive.enabled) {
    // Pack number of bonds first (as bit-cast integer) so unpack knows count
    union ubuf { double d; tagint t; } u;
    int nbonds = cohesive.count_bonds(atom->tag[i]);
    u.t = (tagint)nbonds;
    buf[m++] = u.d;
    m += cohesive.pack_bonds(atom->tag[i], buf + m);
    cohesive.remove_bonds(atom->tag[i]);
  }
  return m;
}

/* ---------------------------------------------------------------------- */

int FixCoupMPM::unpack_exchange(int /*nlocal*/, double *buf)
{
  int m = 0;
  if (cohesive.enabled) {
    union ubuf { double d; tagint t; } u;
    u.d = buf[m++];
    int nbonds = (int)u.t;
    m += cohesive.unpack_bonds(buf + m, nbonds);
  }
  return m;
}
