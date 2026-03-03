#ifndef COUPMPM_STRESS_H
#define COUPMPM_STRESS_H

#include <cmath>
#include <cstring>
#include <cassert>

namespace LAMMPS_NS {
namespace CoupMPM {

// ============================================================
// Small matrix utilities for 3x3 stored as flat double[9]
// Row-major: M[i][j] = m[3*i + j]
// Stress in Voigt: [xx, yy, zz, xy, xz, yz]
// ============================================================
namespace Mat3 {

inline void identity(double m[9]) {
  std::memset(m, 0, 9 * sizeof(double));
  m[0] = m[4] = m[8] = 1.0;
}

inline double det(const double m[9]) {
  return m[0]*(m[4]*m[8] - m[5]*m[7])
       - m[1]*(m[3]*m[8] - m[5]*m[6])
       + m[2]*(m[3]*m[7] - m[4]*m[6]);
}

inline double trace(const double m[9]) {
  return m[0] + m[4] + m[8];
}

// C = A * B (3x3)
inline void multiply(const double a[9], const double b[9], double c[9]) {
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      c[3*i+j] = 0.0;
      for (int k = 0; k < 3; k++)
        c[3*i+j] += a[3*i+k] * b[3*k+j];
    }
}

// B = F * F^T (left Cauchy-Green tensor)
inline void left_cauchy_green(const double F[9], double b[9]) {
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      b[3*i+j] = 0.0;
      for (int k = 0; k < 3; k++)
        b[3*i+j] += F[3*i+k] * F[3*j+k];
    }
}

// F_new = (I + dt*L) * F_old
inline void update_F(const double F_old[9], const double L[9],
                     double dt, double F_new[9]) {
  double IdtL[9];
  identity(IdtL);
  for (int i = 0; i < 9; i++) IdtL[i] += dt * L[i];
  multiply(IdtL, F_old, F_new);
}

} // namespace Mat3

// ============================================================
// MPMStress: abstract constitutive law interface
// ============================================================
class MPMStress {
public:
  virtual ~MPMStress() {}

  // Compute Cauchy stress from deformation gradient.
  // F[9]: deformation gradient (row-major)
  // J: det(F)
  // state[]: internal state variables (modified in-place)
  // dt: timestep (for rate-dependent models)
  // stress_out[6]: Cauchy stress in Voigt [xx, yy, zz, xy, xz, yz]
  virtual void compute_stress(const double F[9], double J,
                              double state[], double dt,
                              double stress_out[6]) = 0;

  // Elastic strain energy density
  virtual double compute_energy(const double F[9], double J,
                                const double state[]) = 0;

  // Estimate wave speed for CFL (sqrt of effective modulus / density)
  virtual double wave_speed(double rho) const = 0;

  // Number of internal state variables per particle
  virtual int n_state_vars() const = 0;
};

// ============================================================
// Neo-Hookean hyperelastic:
//   ψ = μ/2 (I₁ - 3 - 2 ln J) + κ/2 (J - 1)²
//   σ = (μ/J)(b - I) + κ(J-1)I
//
// Parameters: μ (shear modulus), κ (bulk modulus)
// ============================================================
class NeoHookean : public MPMStress {
public:
  double mu;    // shear modulus
  double kappa; // bulk modulus

  NeoHookean() : mu(1.0e3), kappa(1.0e4) {}
  NeoHookean(double mu_, double kappa_) : mu(mu_), kappa(kappa_) {}

  void compute_stress(const double F[9], double J,
                      double /*state*/[], double /*dt*/,
                      double stress_out[6]) override {
    // b = F * F^T
    double b[9];
    Mat3::left_cauchy_green(F, b);

    const double inv_J = 1.0 / J;
    const double p = kappa * (J - 1.0); // pressure-like term

    // σ = (μ/J)(b - I) + κ(J-1)I
    // Voigt: [xx, yy, zz, xy, xz, yz]
    stress_out[0] = mu * inv_J * (b[0] - 1.0) + p; // xx
    stress_out[1] = mu * inv_J * (b[4] - 1.0) + p; // yy
    stress_out[2] = mu * inv_J * (b[8] - 1.0) + p; // zz
    stress_out[3] = mu * inv_J * b[1];              // xy = (b_01 + b_10)/2, but b is symmetric
    stress_out[4] = mu * inv_J * b[2];              // xz
    stress_out[5] = mu * inv_J * b[5];              // yz
  }

  double compute_energy(const double F[9], double J,
                        const double /*state*/[]) override {
    double b[9];
    Mat3::left_cauchy_green(F, b);
    const double I1 = b[0] + b[4] + b[8];
    // J should always be positive for valid deformation.
    // If J ≤ 0, the element has inverted — log is undefined.
    // Clamp to small positive value for robustness.
    const double J_safe = (J > 1e-20) ? J : 1e-20;
    const double lnJ = std::log(J_safe);
    return 0.5 * mu * (I1 - 3.0 - 2.0 * lnJ)
         + 0.5 * kappa * (J - 1.0) * (J - 1.0);
  }

  double wave_speed(double rho) const override {
    // P-wave speed: sqrt((κ + 4μ/3) / ρ)
    return std::sqrt((kappa + 4.0 * mu / 3.0) / rho);
  }

  int n_state_vars() const override { return 0; }
};

// ============================================================
// Mooney-Rivlin (placeholder — implement in Phase 2)
// ============================================================
class MooneyRivlin : public MPMStress {
public:
  double C1, C2, kappa;

  MooneyRivlin(double c1, double c2, double k)
    : C1(c1), C2(c2), kappa(k) {}

  void compute_stress(const double F[9], double J,
                      double state[], double dt,
                      double stress_out[6]) override {
    // TODO: implement Mooney-Rivlin stress
    // For now, fall back to Neo-Hookean with μ = 2(C1+C2)
    NeoHookean nh(2.0*(C1+C2), kappa);
    nh.compute_stress(F, J, state, dt, stress_out);
  }

  double compute_energy(const double F[9], double J,
                        const double state[]) override {
    NeoHookean nh(2.0*(C1+C2), kappa);
    return nh.compute_energy(F, J, state);
  }

  double wave_speed(double rho) const override {
    return std::sqrt((kappa + 4.0 * 2.0 * (C1+C2) / 3.0) / rho);
  }

  int n_state_vars() const override { return 0; }
};

} // namespace CoupMPM
} // namespace LAMMPS_NS

#endif // COUPMPM_STRESS_H
