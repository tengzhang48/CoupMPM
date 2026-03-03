// clang-format off
#ifdef FIX_CLASS
FixStyle(coupmpm, FixCoupMPM);
#else
#ifndef FIX_COUPMPM_H
#define FIX_COUPMPM_H

#include "fix.h"
#include "coupmpm_grid.h"
#include "coupmpm_kernel.h"
#include "coupmpm_transfer.h"
#include "coupmpm_stress.h"
#include "coupmpm_contact.h"
#include "coupmpm_surface.h"
#include "coupmpm_adaptivity.h"
#include "coupmpm_cohesive.h"
#include "coupmpm_io.h"
#include <memory>
#include <string>
#include <vector>

namespace LAMMPS_NS {

class AtomVecMPM;  // forward declaration
class NeighList;   // forward declaration

class FixCoupMPM : public Fix {
public:
  FixCoupMPM(class LAMMPS *, int, char **);
  ~FixCoupMPM() override;

  int setmask() override;
  void init() override;
  void init_list(int, class NeighList *) override;
  void setup(int) override;
  void initial_integrate(int) override;
  void final_integrate() override;
  void end_of_step() override;
  int pack_exchange(int i, double *buf) override;
  int unpack_exchange(int nlocal, double *buf) override;

  // Temporary per-atom work arrays (not communicated)
  double *L_buffer;    // [nmax * 9] velocity gradient
  double *div_v_buf;   // [nmax] smoothed divergence
  double *mass_p;      // [nmax] particle mass cache

private:
  AtomVecMPM *avec;  // pointer to atom_vec for MPM field access
  NeighList *list;   // LAMMPS neighbor list for cohesive bond detection

  CoupMPM::MPMGrid grid;
  CoupMPM::MPMKernel kernel;
  std::unique_ptr<CoupMPM::MPMStress> stress_model;
  std::unique_ptr<CoupMPM::MPMContact> contact_model;
  CoupMPM::MPMGhostExchange ghost_exchange;
  CoupMPM::SurfaceDetector surface_detector;
  CoupMPM::ParticleAdaptivity adaptivity;
  CoupMPM::CohesiveZoneManager cohesive;

  // P2G records for anti-P2G migration protocol
  std::vector<CoupMPM::P2GRecord> p2g_records;

  double grid_dx, grid_dy, grid_dz;
  int dim;
  bool use_bbar, dt_auto, energy_check;
  int vtk_interval, surface_interval;
  std::string vtk_prefix;
  double cfl_factor, rho0;
  double surface_alpha;
  double cz_sigma_tmp, cz_delta_tmp, cz_delta_max_tmp, cz_form_dist_tmp;
  double domain_lo[3], domain_hi[3];
  int Nx_global, Ny_global, Nz_global;
  std::vector<long> vtk_steps;
  long step_count;
  int nmax_alloc;

  void parse_args(int narg, char **arg);
  void setup_grid();
  void setup_mpi();
  void grow_work_arrays(int nmax);
  double compute_dt_cfl();
};

} // namespace LAMMPS_NS

#endif
#endif
