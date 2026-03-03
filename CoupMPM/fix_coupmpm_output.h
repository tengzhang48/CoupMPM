// clang-format off
#ifdef FIX_CLASS
FixStyle(coupmpm/output, FixCoupMPMOutput);
#else
#ifndef FIX_COUPMPM_OUTPUT_H
#define FIX_COUPMPM_OUTPUT_H

#include "fix.h"
#include "coupmpm_surface.h"
#include "coupmpm_io.h"
#include <string>
#include <vector>

namespace LAMMPS_NS {

class FixCoupMPM;

class FixCoupMPMOutput : public Fix {
public:
  FixCoupMPMOutput(class LAMMPS *, int, char **);
  ~FixCoupMPMOutput() override = default;

  int setmask() override;
  void init() override;
  void setup(int) override;
  void end_of_step() override;

private:
  FixCoupMPM *parent;
  CoupMPM::SurfaceDetector surface_detector;

  int         vtk_interval;
  int         surface_interval;
  std::string vtk_prefix;
  double      surface_alpha;

  std::vector<long> vtk_steps;

  void parse_args(int narg, char **arg);
};

} // namespace LAMMPS_NS

#endif
#endif
