// clang-format off
#ifdef FIX_CLASS
FixStyle(coupmpm/adaptivity, FixCoupMPMAdaptivity);
#else
#ifndef FIX_COUPMPM_ADAPTIVITY_H
#define FIX_COUPMPM_ADAPTIVITY_H

#include "fix.h"
#include "coupmpm_adaptivity.h"

namespace LAMMPS_NS {

class FixCoupMPM;

class FixCoupMPMAdaptivity : public Fix {
public:
  FixCoupMPMAdaptivity(class LAMMPS *, int, char **);
  ~FixCoupMPMAdaptivity() override = default;

  int setmask() override;
  void init() override;
  void end_of_step() override;

  // Query used by parent to decide mass computation mode
  bool adaptivity_enabled() const { return adaptivity.enabled; }

  CoupMPM::ParticleAdaptivity adaptivity;

private:
  FixCoupMPM *parent;

  void parse_args(int narg, char **arg);
};

} // namespace LAMMPS_NS

#endif
#endif
