// clang-format off
#ifdef FIX_CLASS
FixStyle(coupmpm/cohesive, FixCoupMPMCohesive);
#else
#ifndef FIX_COUPMPM_COHESIVE_H
#define FIX_COUPMPM_COHESIVE_H

#include "fix.h"
#include "coupmpm_cohesive.h"

namespace LAMMPS_NS {

class FixCoupMPM;
class NeighList;

class FixCoupMPMCohesive : public Fix {
public:
  FixCoupMPMCohesive(class LAMMPS *, int, char **);
  ~FixCoupMPMCohesive() override;

  int setmask() override;
  void init() override;
  void init_list(int, class NeighList *) override;
  void setup(int) override;
  void end_of_step() override;
  int pack_exchange(int i, double *buf) override;
  int unpack_exchange(int nlocal, double *buf) override;

  // Called by parent during initial_integrate (before P2G)
  void compute_forces_before_p2g();

  // Called by adaptivity fix when a particle is being deleted
  void deactivate_bonds_for_tag(tagint tag);

  CoupMPM::CohesiveZoneManager cohesive;

private:
  FixCoupMPM *parent;
  NeighList   *list;

  double cz_sigma_tmp, cz_delta_tmp, cz_delta_max_tmp, cz_form_dist_tmp;

  void parse_args(int narg, char **arg);
};

} // namespace LAMMPS_NS

#endif
#endif
