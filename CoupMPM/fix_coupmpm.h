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
#include <memory>
#include <string>
#include <vector>

namespace LAMMPS_NS {

class AtomVecMPM;          // forward declaration
class FixCoupMPMContact;   // companion fix forward declarations
class FixCoupMPMCohesive;
class FixCoupMPMAdaptivity;
class FixCoupMPMOutput;

class FixCoupMPM : public Fix {
public:
  FixCoupMPM(class LAMMPS *, int, char **);
  ~FixCoupMPM() override;

  int setmask() override;
  void init() override;
  void setup(int) override;
  void initial_integrate(int) override;
  void final_integrate() override;

  // ----------------------------------------------------------------
  // Public state — companion fixes access these directly
  // ----------------------------------------------------------------
  AtomVecMPM *avec;                               // MPM atom-vec
  CoupMPM::MPMGrid grid;                          // background grid
  CoupMPM::MPMKernel kernel;                      // shape-function kernel
  std::unique_ptr<CoupMPM::MPMStress> stress_model; // constitutive model
  CoupMPM::MPMGhostExchange ghost_exchange;       // MPI ghost exchange

  double domain_lo[3], domain_hi[3];
  int    dim;
  double rho0;
  bool   use_bbar;

  double *mass_p;      // [nmax] particle mass cache
  double *L_buffer;    // [nmax*9] velocity gradient (G2P work array)
  double *div_v_buf;   // [nmax] smoothed divergence (G2P work array)

  // P2G records for anti-P2G migration protocol
  std::vector<CoupMPM::P2GRecord> p2g_records;

  // Step counter (incremented at end of final_integrate)
  long step_count;

  // Companion fix pointers — set by each companion's init()
  FixCoupMPMContact    *fix_contact;
  FixCoupMPMCohesive   *fix_cohesive;
  FixCoupMPMAdaptivity *fix_adaptivity;
  FixCoupMPMOutput     *fix_output;

  // Set by contact companion if Bardenhagen method is chosen;
  // read by setup_grid() to allocate per-body node data.
  bool use_bardenhagen_contact;

  // Grow work arrays to at least nmax entries
  void grow_work_arrays(int nmax);

private:
  double grid_dx, grid_dy, grid_dz;
  bool   dt_auto, energy_check;
  double cfl_factor;
  int    Nx_global, Ny_global, Nz_global;
  int    nmax_alloc;

  void parse_args(int narg, char **arg);
  void setup_grid();
  void setup_mpi();
  double compute_dt_cfl();
};

} // namespace LAMMPS_NS

#endif
#endif
