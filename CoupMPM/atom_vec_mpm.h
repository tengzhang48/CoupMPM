// clang-format off
/* -*- c++ -*- ----------------------------------------------------------
   CoupMPM - Material Point Method Package for LAMMPS

   atom_style mpm

   Standard LAMMPS fields:
     tag, type, mask, image, x[3], v[3], f[3], molecule

   MPM-specific per-atom fields:
     F_def[9]      Deformation gradient (row-major 3x3)
     stress_v[6]   Cauchy stress (Voigt: xx yy zz xy xz yz)
     vol0          Reference volume
     Bp[9]         APIC affine velocity matrix (row-major 3x3)
     mpm_state[N]  Internal constitutive state variables
     surface       Dynamic surface flag (0 or 1)
     area0[3]      Reference area normal vector
     area_scale    Current area magnitude

   Data file format per atom line:
     atom-ID  mol-ID  atom-type  x  y  z  vol0

   All other MPM fields are initialized automatically:
     F_def = identity, everything else = zero.
------------------------------------------------------------------------- */

#ifdef ATOM_CLASS
// clang-format off
AtomStyle(mpm, AtomVecMPM);
// clang-format on
#else

#ifndef ATOM_VEC_MPM_H
#define ATOM_VEC_MPM_H

#include "atom_vec.h"

namespace LAMMPS_NS {

class AtomVecMPM : public AtomVec {
public:
  // Compile-time constant: max internal state vars per particle.
  // 9 = one Maxwell branch (F_viscous[9]). Increase for more branches.
  static constexpr int N_STATE = 9;

  // ---------- Public per-atom arrays ----------
  // Accessed by fix_coupmpm via:
  //   auto *avec = dynamic_cast<AtomVecMPM*>(atom->avec);

  double **F_def;       // [nmax][9] deformation gradient
  double **stress_v;    // [nmax][6] Cauchy stress Voigt
  double *vol0;         // [nmax] reference volume
  double **Bp;          // [nmax][9] APIC affine matrix
  double **mpm_state;   // [nmax][N_STATE] constitutive state vars
  int    *surface;      // [nmax] surface flag
  double **area0;       // [nmax][3] reference area normal
  double *area_scale;   // [nmax] current area magnitude

  // ---------- Lifecycle ----------
  AtomVecMPM(class LAMMPS *);
  ~AtomVecMPM() override;

  // ---------- Core atom_vec interface ----------
  void grow(int) override;
  void grow_pointers() override;
  void copy(int, int, int) override;

  int pack_comm(int, int *, double *, int, int *) override;
  int pack_comm_vel(int, int *, double *, int, int *) override;
  void unpack_comm(int, int, double *) override;
  void unpack_comm_vel(int, int, double *) override;

  int pack_reverse(int, int, double *) override;
  void unpack_reverse(int, int *, double *) override;

  int pack_exchange(int, double *) override;
  int unpack_exchange(double *) override;

  int pack_border(int, int *, double *, int, int *) override;
  int pack_border_vel(int, int *, double *, int, int *) override;
  void unpack_border(int, int, double *) override;
  void unpack_border_vel(int, int, double *) override;

  int pack_restart(int, double *) override;
  int unpack_restart(double *) override;

  void data_atom(double *, imageint, const char *const *, tagint) override;
  int data_atom_hybrid(int, const char *const *) override;
  void data_atom_post(int) override;
  void create_atom_post(int) override;

  bigint memory_usage() override;

  // Custom doubles per atom for sizing:
  // F_def(9) + stress_v(6) + vol0(1) + Bp(9) + state(N_STATE) + area0(3) + area_scale(1) = 29+N_STATE
  // Plus surface as int-cast-to-double = 1
  static constexpr int N_MPM_DOUBLES = 9 + 6 + 1 + 9 + N_STATE + 3 + 1;
  static constexpr int N_MPM_INTS = 1;

private:
  tagint *tag;
  int *type, *mask;
  imageint *image;
  double **x, **v, **f;
  tagint *molecule;

  void init_mpm_fields(int i);
  int pack_mpm(int i, double *buf) const;
  int unpack_mpm(int i, const double *buf);
};

} // namespace LAMMPS_NS

#endif
#endif
