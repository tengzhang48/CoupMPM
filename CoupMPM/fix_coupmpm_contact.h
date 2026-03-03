// clang-format off
#ifdef FIX_CLASS
FixStyle(coupmpm/contact, FixCoupMPMContact);
#else
#ifndef FIX_COUPMPM_CONTACT_H
#define FIX_COUPMPM_CONTACT_H

#include "fix.h"
#include "coupmpm_contact.h"
#include <memory>

namespace LAMMPS_NS {

class FixCoupMPM;

class FixCoupMPMContact : public Fix {
public:
  FixCoupMPMContact(class LAMMPS *, int, char **);
  ~FixCoupMPMContact() override = default;

  int setmask() override;
  void init() override;

  // Called by parent during initial_integrate
  void pre_p2g(CoupMPM::MPMGrid &grid);
  void post_grid_solve(CoupMPM::MPMGrid &grid, double dt, MPI_Comm world);

  bool is_bardenhagen() const;
  bool is_penalty() const;

private:
  FixCoupMPM *parent;
  std::unique_ptr<CoupMPM::MPMContact> contact_model;

  void parse_args(int narg, char **arg);
};

} // namespace LAMMPS_NS

#endif
#endif
