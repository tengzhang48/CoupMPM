// clang-format off
/* -*- c++ -*- ----------------------------------------------------------
   CoupMPM - fix coupmpm/cohesive

   Companion fix to fix coupmpm. Manages dynamic particle-based
   cohesive zone bonds for adhesion, delamination, and fracture.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(coupmpm/cohesive, FixCoupMPMCohesive);
// clang-format on
#else

#ifndef FIX_COUPMPM_COHESIVE_H
#define FIX_COUPMPM_COHESIVE_H

#include "fix.h"
#include "coupmpm_cohesive.h"

namespace LAMMPS_NS {

class FixCoupMPM;   // forward declare parent

class FixCoupMPMCohesive : public Fix {
public:
  FixCoupMPMCohesive(class LAMMPS *, int, char **);
  ~FixCoupMPMCohesive() override;

  int setmask() override;
  void init() override;
  void init_list(int, class NeighList *) override;
  void setup(int) override;
  void end_of_step() override;

  // Called by parent fix_coupmpm before P2G
  void compute_forces_before_p2g();

  // Called by fix_coupmpm/adaptivity when particles are deleted
  void deactivate_bonds_for_tag(tagint tag);

  // --- LAMMPS atom callbacks for bond migration ---
  // These are called when atoms cross MPI subdomain boundaries.
  void grow_arrays(int) override;
  void copy_arrays(int, int, int) override;
  int pack_exchange(int, double *) override;
  int unpack_exchange(int, double *) override;

  // --- Fix-specific reverse communication ---
  // Sends ghost atom forces back to owners after cohesive force computation.
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;

  // The cohesive zone manager (public for parent access to diagnostics)
  CoupMPM::CohesiveZoneManager cohesive;

private:
  FixCoupMPM *parent;
  class NeighList *list;

  // Temporary parameter storage (parsed in constructor, applied in setup)
  double cz_sigma_tmp;
  double cz_delta_tmp;
  double cz_delta_max_tmp;
  double cz_form_dist_tmp;

  void parse_args(int narg, char **arg);
};

} // namespace LAMMPS_NS

#endif // FIX_COUPMPM_COHESIVE_H
#endif // FIX_CLASS
