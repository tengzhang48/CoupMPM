/* ----------------------------------------------------------------------
   CoupMPM - fix coupmpm/contact

   fix ID group-ID coupmpm/contact method {none|bardenhagen|penalty} ...

   Companion fix to fix coupmpm.  Handles multi-body contact on the
   MPM background grid.  The parent fix calls pre_p2g() before P2G
   and post_grid_solve() after the grid solve on every timestep.
---------------------------------------------------------------------- */

#include "fix_coupmpm_contact.h"
#include "fix_coupmpm.h"
#include "error.h"
#include "modify.h"
#include <cstring>

using namespace LAMMPS_NS;
using namespace CoupMPM;

/* ---------------------------------------------------------------------- */

FixCoupMPMContact::FixCoupMPMContact(LAMMPS *lmp, int narg, char **arg)
  : Fix(lmp, narg, arg), parent(nullptr)
{
  if (narg < 3) error->all(FLERR, "Illegal fix coupmpm/contact command");
  parse_args(narg, arg);
}

/* ---------------------------------------------------------------------- */

FixCoupMPMContact::~FixCoupMPMContact()
{
  if (parent) {
    parent->fix_contact = nullptr;
    parent->use_bardenhagen_contact = false;
  }
}

/* ---------------------------------------------------------------------- */

int FixCoupMPMContact::setmask()
{
  // No standard LAMMPS callbacks — parent drives us via stored pointer
  return 0;
}

/* ---------------------------------------------------------------------- */

void FixCoupMPMContact::parse_args(int narg, char **arg)
{
  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "method") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "fix coupmpm/contact: method needs argument");

      if (strcmp(arg[iarg+1], "none") == 0) {
        contact_model = std::make_unique<ContactNone>();
        iarg += 2;
      }
      else if (strcmp(arg[iarg+1], "bardenhagen") == 0) {
        auto *bard = new ContactBardenhagen();
        int sub_start = iarg + 2;
        int sub_end = sub_start;
        // Scan forward for sub-args until next top-level keyword
        while (sub_end < narg && strcmp(arg[sub_end], "method") != 0)
          sub_end++;
        bard->init(sub_end - sub_start, &arg[sub_start]);
        contact_model.reset(bard);
        iarg = sub_end;
      }
      else if (strcmp(arg[iarg+1], "penalty") == 0) {
        contact_model = std::make_unique<ContactPenalty>();
        iarg += 2;
      }
      else {
        error->all(FLERR, "fix coupmpm/contact: unknown contact method");
      }
    }
    else {
      error->all(FLERR,
        fmt::format("fix coupmpm/contact: unknown keyword '{}'", arg[iarg]));
    }
  }

  if (!contact_model) contact_model = std::make_unique<ContactNone>();
}

/* ---------------------------------------------------------------------- */

void FixCoupMPMContact::init()
{
  // Find the parent coupmpm fix
  parent = nullptr;
  for (int i = 0; i < modify->nfix; i++) {
    if (strcmp(modify->fix[i]->style, "coupmpm") == 0) {
      parent = static_cast<FixCoupMPM *>(modify->fix[i]);
      break;
    }
  }
  if (!parent)
    error->all(FLERR,
      "fix coupmpm/contact: fix coupmpm must be defined before fix coupmpm/contact");

  parent->fix_contact = this;
  if (is_bardenhagen()) parent->use_bardenhagen_contact = true;
}

/* ---------------------------------------------------------------------- */

void FixCoupMPMContact::pre_p2g(CoupMPM::MPMGrid &grid)
{
  if (contact_model) contact_model->pre_p2g(grid);
}

/* ---------------------------------------------------------------------- */

void FixCoupMPMContact::post_grid_solve(CoupMPM::MPMGrid &grid,
                                        double dt, MPI_Comm world)
{
  if (contact_model) contact_model->post_grid_solve(grid, dt, world);
}

/* ---------------------------------------------------------------------- */

bool FixCoupMPMContact::is_bardenhagen() const
{
  return contact_model &&
         strcmp(contact_model->name(), "bardenhagen") == 0;
}

/* ---------------------------------------------------------------------- */

bool FixCoupMPMContact::is_penalty() const
{
  return contact_model &&
         strcmp(contact_model->name(), "penalty") == 0;
}
