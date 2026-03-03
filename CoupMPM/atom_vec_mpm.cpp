/* ----------------------------------------------------------------------
   CoupMPM - Material Point Method Package for LAMMPS
   atom_style mpm — full implementation with MPI pack/unpack

   Based on LAMMPS atom_vec_sphere.cpp / atom_vec_body.cpp patterns.
   All per-atom MPM fields are communicated during exchange, border,
   restart, and (selectively) during forward/reverse comm.

   Data file format:
     Atoms section:
       atom-ID  mol-ID  atom-type  x  y  z  vol0

   Initialization:
     F_def = I (identity), stress_v = 0, Bp = 0, state = 0,
     surface = 0, area0 = 0, area_scale = 0
---------------------------------------------------------------------- */

#include "atom_vec_mpm.h"
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "math_const.h"
#include "memory.h"
#include "modify.h"

#include <cstring>
#include <cstdlib>

using namespace LAMMPS_NS;

// Growth increment for atom arrays — matches the standard LAMMPS convention.
static constexpr int DELTA = 16384;

/* ---------------------------------------------------------------------- */

AtomVecMPM::AtomVecMPM(LAMMPS *lmp) : AtomVec(lmp)
{
  molecular = Atom::MOLECULAR;
  atom->molecule_flag = 1;
  mass_type = 1;  // per-type mass (use mass command, not rmass)

  // Communication sizes (base class fields handled separately in modern LAMMPS)
  // These are ADDITIONAL values beyond what the base class packs.
  //
  // Forward comm: only x (3) is strictly needed by LAMMPS.
  //   We don't add MPM fields to forward comm — they're updated
  //   locally by the fix each step. Ghost atoms get velocity_new
  //   through the grid ghost exchange, not atom comm.
  //
  // Reverse comm: only f (3) — standard force accumulation.
  //
  // Exchange/border/restart: ALL MPM fields must travel.

  // These affect buffer allocation in comm class:
  size_forward = 3;      // x only
  size_reverse = 3;      // f only
  // x(3) + tag(1) + type(1) + mask(1) + molecule(1) + v(3) = 10 base fields.
  // For safety, set generously. These are just buffer-size hints.
  size_border = 10 + N_MPM_DOUBLES + N_MPM_INTS;
  size_data_atom = 7;    // id mol type x y z vol0

  // Initialize pointers
  F_def = nullptr;
  stress_v = nullptr;
  vol0 = nullptr;
  Bp = nullptr;
  mpm_state = nullptr;
  surface = nullptr;
  area0 = nullptr;
  area_scale = nullptr;

  tag = nullptr; type = nullptr; mask = nullptr; image = nullptr;
  x = nullptr; v = nullptr; f = nullptr; molecule = nullptr;
}

/* ---------------------------------------------------------------------- */

AtomVecMPM::~AtomVecMPM()
{
  memory->destroy(F_def);
  memory->destroy(stress_v);
  memory->destroy(vol0);
  memory->destroy(Bp);
  memory->destroy(mpm_state);
  memory->destroy(surface);
  memory->destroy(area0);
  memory->destroy(area_scale);
}

/* ---------------------------------------------------------------------- */

void AtomVecMPM::init_mpm_fields(int i)
{
  // F_def = identity
  for (int k = 0; k < 9; k++) F_def[i][k] = 0.0;
  F_def[i][0] = F_def[i][4] = F_def[i][8] = 1.0;

  // Everything else = zero
  for (int k = 0; k < 6; k++) stress_v[i][k] = 0.0;
  vol0[i] = 0.0;
  for (int k = 0; k < 9; k++) Bp[i][k] = 0.0;
  for (int k = 0; k < N_STATE; k++) mpm_state[i][k] = 0.0;
  surface[i] = 0;
  for (int k = 0; k < 3; k++) area0[i][k] = 0.0;
  area_scale[i] = 0.0;
}

/* ---------------------------------------------------------------------- */
// Pack all MPM fields for atom i into buf. Returns count.
/* ---------------------------------------------------------------------- */

int AtomVecMPM::pack_mpm(int i, double *buf) const
{
  int m = 0;
  for (int k = 0; k < 9; k++) buf[m++] = F_def[i][k];
  for (int k = 0; k < 6; k++) buf[m++] = stress_v[i][k];
  buf[m++] = vol0[i];
  for (int k = 0; k < 9; k++) buf[m++] = Bp[i][k];
  for (int k = 0; k < N_STATE; k++) buf[m++] = mpm_state[i][k];
  buf[m++] = ubuf(surface[i]).d;
  for (int k = 0; k < 3; k++) buf[m++] = area0[i][k];
  buf[m++] = area_scale[i];
  return m; // should equal N_MPM_DOUBLES + N_MPM_INTS
}

/* ---------------------------------------------------------------------- */
// Unpack all MPM fields from buf into atom i. Returns count.
/* ---------------------------------------------------------------------- */

int AtomVecMPM::unpack_mpm(int i, const double *buf)
{
  int m = 0;
  for (int k = 0; k < 9; k++) F_def[i][k] = buf[m++];
  for (int k = 0; k < 6; k++) stress_v[i][k] = buf[m++];
  vol0[i] = buf[m++];
  for (int k = 0; k < 9; k++) Bp[i][k] = buf[m++];
  for (int k = 0; k < N_STATE; k++) mpm_state[i][k] = buf[m++];
  surface[i] = (int) ubuf(buf[m++]).i;
  for (int k = 0; k < 3; k++) area0[i][k] = buf[m++];
  area_scale[i] = buf[m++];
  return m;
}

/* ---------------------------------------------------------------------- */

void AtomVecMPM::grow(int n)
{
  int nmax = atom->nmax;
  if (n == 0) {
    // Called when nlocal == nmax: grow by a fixed delta to amortize
    // reallocation cost (standard LAMMPS pattern).
    nmax += DELTA;
  } else if (n > nmax) {
    nmax = n + DELTA;
  }

  // Standard LAMMPS arrays
  tag = memory->grow(atom->tag, nmax, "atom:tag");
  type = memory->grow(atom->type, nmax, "atom:type");
  mask = memory->grow(atom->mask, nmax, "atom:mask");
  image = memory->grow(atom->image, nmax, "atom:image");
  x = memory->grow(atom->x, nmax, 3, "atom:x");
  v = memory->grow(atom->v, nmax, 3, "atom:v");
  f = memory->grow(atom->f, nmax, 3, "atom:f");
  molecule = memory->grow(atom->molecule, nmax, "atom:molecule");

  // MPM custom arrays
  memory->grow(F_def, nmax, 9, "atom:F_def");
  memory->grow(stress_v, nmax, 6, "atom:stress_v");
  memory->grow(vol0, nmax, "atom:vol0");
  memory->grow(Bp, nmax, 9, "atom:Bp");
  memory->grow(mpm_state, nmax, N_STATE, "atom:mpm_state");
  memory->grow(surface, nmax, "atom:surface");
  memory->grow(area0, nmax, 3, "atom:area0");
  memory->grow(area_scale, nmax, "atom:area_scale");

  // Let fixes grow their per-atom arrays
  if (atom->extra_grow)
    for (int iextra = 0; iextra < atom->extra_grow; iextra++)
      modify->fix[atom->extra_GROW[iextra]]->grow_arrays(nmax);

  atom->nmax = nmax;
}

/* ---------------------------------------------------------------------- */

void AtomVecMPM::grow_pointers()
{
  // Update local pointers after reallocation
  tag = atom->tag;
  type = atom->type;
  mask = atom->mask;
  image = atom->image;
  x = atom->x;
  v = atom->v;
  f = atom->f;
  molecule = atom->molecule;
  // MPM arrays are class members, already updated by grow()
}

/* ---------------------------------------------------------------------- */

void AtomVecMPM::copy(int i, int j, int delflag)
{
  // Copy atom i's data to position j (used when atoms are deleted/sorted)

  tag[j] = tag[i];
  type[j] = type[i];
  mask[j] = mask[i];
  image[j] = image[i];
  x[j][0] = x[i][0]; x[j][1] = x[i][1]; x[j][2] = x[i][2];
  v[j][0] = v[i][0]; v[j][1] = v[i][1]; v[j][2] = v[i][2];
  molecule[j] = molecule[i];

  // MPM fields
  memcpy(F_def[j], F_def[i], 9 * sizeof(double));
  memcpy(stress_v[j], stress_v[i], 6 * sizeof(double));
  vol0[j] = vol0[i];
  memcpy(Bp[j], Bp[i], 9 * sizeof(double));
  memcpy(mpm_state[j], mpm_state[i], N_STATE * sizeof(double));
  surface[j] = surface[i];
  memcpy(area0[j], area0[i], 3 * sizeof(double));
  area_scale[j] = area_scale[i];

  // Let fixes copy their per-atom data
  if (atom->extra_grow)
    for (int iextra = 0; iextra < atom->extra_grow; iextra++)
      modify->fix[atom->extra_GROW[iextra]]->copy_arrays(i, j, delflag);
}

/* ======================================================================
   Forward communication: send position data to ghost atoms.
   Standard: just x. We don't send MPM fields in forward comm —
   the grid ghost exchange handles velocity_new distribution, and
   F/stress are updated locally after G2P.
   ====================================================================== */

int AtomVecMPM::pack_comm(int n, int *list, double *buf,
                          int pbc_flag, int *pbc)
{
  int m = 0;
  if (pbc_flag == 0) {
    for (int i = 0; i < n; i++) {
      const int j = list[i];
      buf[m++] = x[j][0];
      buf[m++] = x[j][1];
      buf[m++] = x[j][2];
    }
  } else {
    double dx = 0, dy = 0, dz = 0;
    if (domain->triclinic == 0) {
      dx = pbc[0] * domain->xprd;
      dy = pbc[1] * domain->yprd;
      dz = pbc[2] * domain->zprd;
    } else {
      dx = pbc[0] * domain->xprd + pbc[5] * domain->xy + pbc[4] * domain->xz;
      dy = pbc[1] * domain->yprd + pbc[3] * domain->yz;
      dz = pbc[2] * domain->zprd;
    }
    for (int i = 0; i < n; i++) {
      const int j = list[i];
      buf[m++] = x[j][0] + dx;
      buf[m++] = x[j][1] + dy;
      buf[m++] = x[j][2] + dz;
    }
  }
  return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecMPM::pack_comm_vel(int n, int *list, double *buf,
                              int pbc_flag, int *pbc)
{
  int m = 0;
  if (pbc_flag == 0) {
    for (int i = 0; i < n; i++) {
      const int j = list[i];
      buf[m++] = x[j][0]; buf[m++] = x[j][1]; buf[m++] = x[j][2];
      buf[m++] = v[j][0]; buf[m++] = v[j][1]; buf[m++] = v[j][2];
    }
  } else {
    double dx = 0, dy = 0, dz = 0;
    if (domain->triclinic == 0) {
      dx = pbc[0] * domain->xprd;
      dy = pbc[1] * domain->yprd;
      dz = pbc[2] * domain->zprd;
    } else {
      dx = pbc[0]*domain->xprd + pbc[5]*domain->xy + pbc[4]*domain->xz;
      dy = pbc[1]*domain->yprd + pbc[3]*domain->yz;
      dz = pbc[2]*domain->zprd;
    }
    for (int i = 0; i < n; i++) {
      const int j = list[i];
      buf[m++] = x[j][0] + dx; buf[m++] = x[j][1] + dy; buf[m++] = x[j][2] + dz;
      buf[m++] = v[j][0]; buf[m++] = v[j][1]; buf[m++] = v[j][2];
    }
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void AtomVecMPM::unpack_comm(int n, int first, double *buf)
{
  int m = 0;
  const int last = first + n;
  for (int i = first; i < last; i++) {
    x[i][0] = buf[m++]; x[i][1] = buf[m++]; x[i][2] = buf[m++];
  }
}

/* ---------------------------------------------------------------------- */

void AtomVecMPM::unpack_comm_vel(int n, int first, double *buf)
{
  int m = 0;
  const int last = first + n;
  for (int i = first; i < last; i++) {
    x[i][0] = buf[m++]; x[i][1] = buf[m++]; x[i][2] = buf[m++];
    v[i][0] = buf[m++]; v[i][1] = buf[m++]; v[i][2] = buf[m++];
  }
}

/* ======================================================================
   Reverse communication: accumulate forces from ghost atoms.
   Standard: just f. No custom MPM fields needed in reverse.
   ====================================================================== */

int AtomVecMPM::pack_reverse(int n, int first, double *buf)
{
  int m = 0;
  const int last = first + n;
  for (int i = first; i < last; i++) {
    buf[m++] = f[i][0]; buf[m++] = f[i][1]; buf[m++] = f[i][2];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void AtomVecMPM::unpack_reverse(int n, int *list, double *buf)
{
  int m = 0;
  for (int i = 0; i < n; i++) {
    const int j = list[i];
    f[j][0] += buf[m++]; f[j][1] += buf[m++]; f[j][2] += buf[m++];
  }
}

/* ======================================================================
   Exchange: atom migrates to another processor.
   Pack EVERYTHING — this is the complete atom state.
   ====================================================================== */

int AtomVecMPM::pack_exchange(int i, double *buf)
{
  int m = 1; // buf[0] reserved for message size

  buf[m++] = x[i][0]; buf[m++] = x[i][1]; buf[m++] = x[i][2];
  buf[m++] = v[i][0]; buf[m++] = v[i][1]; buf[m++] = v[i][2];
  buf[m++] = ubuf(tag[i]).d;
  buf[m++] = ubuf(type[i]).d;
  buf[m++] = ubuf(mask[i]).d;
  buf[m++] = ubuf(image[i]).d;
  buf[m++] = ubuf(molecule[i]).d;

  // All MPM fields
  m += pack_mpm(i, &buf[m]);

  // Let fixes pack their per-atom data
  if (atom->extra_grow)
    for (int iextra = 0; iextra < atom->extra_grow; iextra++)
      m += modify->fix[atom->extra_GROW[iextra]]->pack_exchange(i, &buf[m]);

  buf[0] = m;
  return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecMPM::unpack_exchange(double *buf)
{
  int nlocal = atom->nlocal;
  if (nlocal == atom->nmax) grow(0);

  int m = 1;

  x[nlocal][0] = buf[m++]; x[nlocal][1] = buf[m++]; x[nlocal][2] = buf[m++];
  v[nlocal][0] = buf[m++]; v[nlocal][1] = buf[m++]; v[nlocal][2] = buf[m++];
  tag[nlocal] = (tagint) ubuf(buf[m++]).i;
  type[nlocal] = (int) ubuf(buf[m++]).i;
  mask[nlocal] = (int) ubuf(buf[m++]).i;
  image[nlocal] = (imageint) ubuf(buf[m++]).i;
  molecule[nlocal] = (tagint) ubuf(buf[m++]).i;

  // All MPM fields
  m += unpack_mpm(nlocal, &buf[m]);

  // Let fixes unpack their per-atom data
  if (atom->extra_grow)
    for (int iextra = 0; iextra < atom->extra_grow; iextra++)
      m += modify->fix[atom->extra_GROW[iextra]]->unpack_exchange(nlocal, &buf[m]);

  atom->nlocal++;
  return m;
}

/* ======================================================================
   Border: build ghost atom list.
   Includes all fields that ghost atoms need.
   ====================================================================== */

int AtomVecMPM::pack_border(int n, int *list, double *buf,
                            int pbc_flag, int *pbc)
{
  int m = 0;
  if (pbc_flag == 0) {
    for (int i = 0; i < n; i++) {
      const int j = list[i];
      buf[m++] = x[j][0]; buf[m++] = x[j][1]; buf[m++] = x[j][2];
      buf[m++] = ubuf(tag[j]).d;
      buf[m++] = ubuf(type[j]).d;
      buf[m++] = ubuf(mask[j]).d;
      buf[m++] = ubuf(molecule[j]).d;
      m += pack_mpm(j, &buf[m]);
    }
  } else {
    double dx = 0, dy = 0, dz = 0;
    if (domain->triclinic == 0) {
      dx = pbc[0] * domain->xprd;
      dy = pbc[1] * domain->yprd;
      dz = pbc[2] * domain->zprd;
    } else {
      dx = pbc[0]*domain->xprd + pbc[5]*domain->xy + pbc[4]*domain->xz;
      dy = pbc[1]*domain->yprd + pbc[3]*domain->yz;
      dz = pbc[2]*domain->zprd;
    }
    for (int i = 0; i < n; i++) {
      const int j = list[i];
      buf[m++] = x[j][0] + dx;
      buf[m++] = x[j][1] + dy;
      buf[m++] = x[j][2] + dz;
      buf[m++] = ubuf(tag[j]).d;
      buf[m++] = ubuf(type[j]).d;
      buf[m++] = ubuf(mask[j]).d;
      buf[m++] = ubuf(molecule[j]).d;
      m += pack_mpm(j, &buf[m]);
    }
  }
  return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecMPM::pack_border_vel(int n, int *list, double *buf,
                                int pbc_flag, int *pbc)
{
  int m = 0;
  if (pbc_flag == 0) {
    for (int i = 0; i < n; i++) {
      const int j = list[i];
      buf[m++] = x[j][0]; buf[m++] = x[j][1]; buf[m++] = x[j][2];
      buf[m++] = ubuf(tag[j]).d;
      buf[m++] = ubuf(type[j]).d;
      buf[m++] = ubuf(mask[j]).d;
      buf[m++] = ubuf(molecule[j]).d;
      buf[m++] = v[j][0]; buf[m++] = v[j][1]; buf[m++] = v[j][2];
      m += pack_mpm(j, &buf[m]);
    }
  } else {
    double dx = 0, dy = 0, dz = 0;
    if (domain->triclinic == 0) {
      dx = pbc[0] * domain->xprd;
      dy = pbc[1] * domain->yprd;
      dz = pbc[2] * domain->zprd;
    } else {
      dx = pbc[0]*domain->xprd + pbc[5]*domain->xy + pbc[4]*domain->xz;
      dy = pbc[1]*domain->yprd + pbc[3]*domain->yz;
      dz = pbc[2]*domain->zprd;
    }
    for (int i = 0; i < n; i++) {
      const int j = list[i];
      buf[m++] = x[j][0] + dx; buf[m++] = x[j][1] + dy; buf[m++] = x[j][2] + dz;
      buf[m++] = ubuf(tag[j]).d;
      buf[m++] = ubuf(type[j]).d;
      buf[m++] = ubuf(mask[j]).d;
      buf[m++] = ubuf(molecule[j]).d;
      buf[m++] = v[j][0]; buf[m++] = v[j][1]; buf[m++] = v[j][2];
      m += pack_mpm(j, &buf[m]);
    }
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void AtomVecMPM::unpack_border(int n, int first, double *buf)
{
  int m = 0;
  const int last = first + n;
  for (int i = first; i < last; i++) {
    if (i == atom->nmax) grow(0);
    x[i][0] = buf[m++]; x[i][1] = buf[m++]; x[i][2] = buf[m++];
    tag[i] = (tagint) ubuf(buf[m++]).i;
    type[i] = (int) ubuf(buf[m++]).i;
    mask[i] = (int) ubuf(buf[m++]).i;
    molecule[i] = (tagint) ubuf(buf[m++]).i;
    m += unpack_mpm(i, &buf[m]);
  }
}

/* ---------------------------------------------------------------------- */

void AtomVecMPM::unpack_border_vel(int n, int first, double *buf)
{
  int m = 0;
  const int last = first + n;
  for (int i = first; i < last; i++) {
    if (i == atom->nmax) grow(0);
    x[i][0] = buf[m++]; x[i][1] = buf[m++]; x[i][2] = buf[m++];
    tag[i] = (tagint) ubuf(buf[m++]).i;
    type[i] = (int) ubuf(buf[m++]).i;
    mask[i] = (int) ubuf(buf[m++]).i;
    molecule[i] = (tagint) ubuf(buf[m++]).i;
    v[i][0] = buf[m++]; v[i][1] = buf[m++]; v[i][2] = buf[m++];
    m += unpack_mpm(i, &buf[m]);
  }
}

/* ======================================================================
   Restart I/O
   ====================================================================== */

int AtomVecMPM::pack_restart(int i, double *buf)
{
  int m = 1; // buf[0] = message size

  buf[m++] = x[i][0]; buf[m++] = x[i][1]; buf[m++] = x[i][2];
  buf[m++] = ubuf(tag[i]).d;
  buf[m++] = ubuf(type[i]).d;
  buf[m++] = ubuf(mask[i]).d;
  buf[m++] = ubuf(image[i]).d;
  buf[m++] = v[i][0]; buf[m++] = v[i][1]; buf[m++] = v[i][2];
  buf[m++] = ubuf(molecule[i]).d;

  m += pack_mpm(i, &buf[m]);

  // Let fixes pack restart data
  if (atom->extra_restart)
    for (int iextra = 0; iextra < atom->extra_restart; iextra++)
      m += modify->fix[atom->extra_RESTART[iextra]]->pack_restart(i, &buf[m]);

  buf[0] = m;
  return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecMPM::unpack_restart(double *buf)
{
  int nlocal = atom->nlocal;
  if (nlocal == atom->nmax) {
    grow(0);
    if (atom->extra_store) {
      // Handled by fix code if needed
    }
  }

  int m = 1;

  x[nlocal][0] = buf[m++]; x[nlocal][1] = buf[m++]; x[nlocal][2] = buf[m++];
  tag[nlocal] = (tagint) ubuf(buf[m++]).i;
  type[nlocal] = (int) ubuf(buf[m++]).i;
  mask[nlocal] = (int) ubuf(buf[m++]).i;
  image[nlocal] = (imageint) ubuf(buf[m++]).i;
  v[nlocal][0] = buf[m++]; v[nlocal][1] = buf[m++]; v[nlocal][2] = buf[m++];
  molecule[nlocal] = (tagint) ubuf(buf[m++]).i;

  m += unpack_mpm(nlocal, &buf[m]);

  // Let fixes unpack
  double **extra = atom->extra;
  if (atom->extra_store) {
    int size = static_cast<int>(buf[0]) - m;
    for (int j = 0; j < size; j++)
      extra[nlocal][j] = buf[m++];
  }

  atom->nlocal++;
  return m;
}

/* ======================================================================
   Data file reading
   Format: atom-ID mol-ID atom-type x y z vol0
   ====================================================================== */

void AtomVecMPM::data_atom(double *coord, imageint imagetmp,
                           const char *const *values, tagint thistag)
{
  int nlocal = atom->nlocal;
  if (nlocal == atom->nmax) grow(0);

  tag[nlocal] = thistag;

  molecule[nlocal] = utils::tnumeric(FLERR, values[1], true, lmp);
  type[nlocal] = utils::inumeric(FLERR, values[2], true, lmp);
  if (type[nlocal] <= 0 || type[nlocal] > atom->ntypes)
    error->one(FLERR, "Invalid atom type in Atoms section of data file");

  x[nlocal][0] = coord[0];
  x[nlocal][1] = coord[1];
  x[nlocal][2] = coord[2];

  image[nlocal] = imagetmp;
  mask[nlocal] = 1;
  v[nlocal][0] = 0.0;
  v[nlocal][1] = 0.0;
  v[nlocal][2] = 0.0;

  // Initialize MPM fields (F=I, stress=0, etc.)
  init_mpm_fields(nlocal);
  // Restore vol0 since init_mpm_fields zeroed it
  vol0[nlocal] = utils::numeric(FLERR, values[6], true, lmp);

  atom->nlocal++;
}

/* ---------------------------------------------------------------------- */

int AtomVecMPM::data_atom_hybrid(int nlocal, const char *const *values)
{
  // For hybrid atom styles: read vol0
  vol0[nlocal] = utils::numeric(FLERR, values[0], true, lmp);
  return 1;
}

/* ---------------------------------------------------------------------- */

void AtomVecMPM::data_atom_post(int ilocal)
{
  // Called after data_atom for any post-processing.
  // F_def already set to identity in data_atom.
  // Nothing extra needed.
}

/* ---------------------------------------------------------------------- */

void AtomVecMPM::create_atom_post(int ilocal)
{
  // Called when an atom is created via create_atoms command.
  // Initialize MPM fields.
  init_mpm_fields(ilocal);
}

/* ---------------------------------------------------------------------- */

bigint AtomVecMPM::memory_usage()
{
  bigint bytes = 0;
  if (atom->memcheck("tag")) bytes += (bigint)atom->nmax * sizeof(tagint);
  if (atom->memcheck("type")) bytes += (bigint)atom->nmax * sizeof(int);
  if (atom->memcheck("mask")) bytes += (bigint)atom->nmax * sizeof(int);
  if (atom->memcheck("image")) bytes += (bigint)atom->nmax * sizeof(imageint);
  if (atom->memcheck("x")) bytes += (bigint)atom->nmax * 3 * sizeof(double);
  if (atom->memcheck("v")) bytes += (bigint)atom->nmax * 3 * sizeof(double);
  if (atom->memcheck("f")) bytes += (bigint)atom->nmax * 3 * sizeof(double);
  if (atom->memcheck("molecule")) bytes += (bigint)atom->nmax * sizeof(tagint);

  // MPM arrays
  bytes += (bigint)atom->nmax * 9 * sizeof(double);       // F_def
  bytes += (bigint)atom->nmax * 6 * sizeof(double);       // stress_v
  bytes += (bigint)atom->nmax * sizeof(double);            // vol0
  bytes += (bigint)atom->nmax * 9 * sizeof(double);       // Bp
  bytes += (bigint)atom->nmax * N_STATE * sizeof(double);  // mpm_state
  bytes += (bigint)atom->nmax * sizeof(int);               // surface
  bytes += (bigint)atom->nmax * 3 * sizeof(double);       // area0
  bytes += (bigint)atom->nmax * sizeof(double);            // area_scale

  return bytes;
}
