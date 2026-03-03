#ifndef COUPMPM_KERNEL_H
#define COUPMPM_KERNEL_H

#include <cmath>
#include <cassert>

namespace LAMMPS_NS {
namespace CoupMPM {

// ============================================================
// MPMKernel: shape function interface
//
// evaluate() computes weight and gradient for a particle at
// position x_p and a grid node at position x_n, given grid
// spacing dx. Positions are in PHYSICAL coordinates.
//
// support() returns the half-width in grid cells.
// D_inverse() returns the APIC inertia tensor inverse (diagonal
// for uniform grids).
// ============================================================

enum class KernelType { LINEAR, BSPLINE2, BSPLINE3 };

// ------------------------------------------------------------
// 1D basis functions
// ------------------------------------------------------------

// Linear: C0, support = [-1, 1]
inline double basis_linear(double r) {
  const double ar = std::fabs(r);
  if (ar < 1.0) return 1.0 - ar;
  return 0.0;
}
inline double grad_linear(double r) {
  const double ar = std::fabs(r);
  if (ar < 1.0) return (r < 0.0) ? 1.0 : -1.0;
  return 0.0;
}

// Quadratic B-spline: C1, support = [-1.5, 1.5]
inline double basis_bspline2(double r) {
  const double ar = std::fabs(r);
  if (ar < 0.5) return 0.75 - ar * ar;
  if (ar < 1.5) { const double t = 1.5 - ar; return 0.5 * t * t; }
  return 0.0;
}
inline double grad_bspline2(double r) {
  const double ar = std::fabs(r);
  if (ar < 0.5) return -2.0 * r;
  if (ar < 1.5) return (r > 0.0) ? -(1.5 - ar) : (1.5 - ar);
  return 0.0;
}

// Cubic B-spline: C2, support = [-2, 2]
inline double basis_bspline3(double r) {
  const double ar = std::fabs(r);
  if (ar < 1.0) return 0.5 * ar * ar * ar - ar * ar + 2.0 / 3.0;
  if (ar < 2.0) { const double t = 2.0 - ar; return t * t * t / 6.0; }
  return 0.0;
}
inline double grad_bspline3(double r) {
  const double ar = std::fabs(r);
  const double s = (r >= 0.0) ? 1.0 : -1.0;
  if (ar < 1.0) return s * (1.5 * ar * ar - 2.0 * ar);
  if (ar < 2.0) { const double t = 2.0 - ar; return -s * 0.5 * t * t; }
  return 0.0;
}

// ============================================================
// MPMKernel class: wraps 1D basis into 3D tensor product
// ============================================================
class MPMKernel {
public:
  KernelType type;

  MPMKernel() : type(KernelType::BSPLINE2) {}
  explicit MPMKernel(KernelType t) : type(t) {}

  // Half-width in grid cells (number of nodes to each side)
  int support() const {
    switch (type) {
      case KernelType::LINEAR:   return 1;  // nodes at -1..+1 from nearest
      case KernelType::BSPLINE2: return 2;  // support 1.5 cells; ceiling is 2
      case KernelType::BSPLINE3: return 2;  // nodes at -2..+1 from floor
    }
    return 2;
  }

  // Ghost layer width needed (= support radius in cells)
  int ghost_width() const {
    switch (type) {
      case KernelType::LINEAR:   return 1;
      case KernelType::BSPLINE2: return 2;
      case KernelType::BSPLINE3: return 3;
    }
    return 2;
  }

  // Number of nodes in each dimension that a particle interacts with
  int nodes_per_dim() const {
    switch (type) {
      case KernelType::LINEAR:   return 2;
      case KernelType::BSPLINE2: return 3;
      case KernelType::BSPLINE3: return 4;
    }
    return 3;
  }

  // APIC D_inverse (diagonal, uniform grid)
  // D = Σ_i w_i (x_i - x_p) ⊗ (x_i - x_p)
  // For uniform grids, D is diagonal: D_dd = dx_d^2 * c
  // where c depends on kernel type.
  // D_inv = 1 / (dx_d^2 * c)
  void D_inverse(const double dx_d[3], int dim, double Dinv[3]) const {
    double c;
    switch (type) {
      case KernelType::LINEAR:   c = 1.0 / 4.0;  break;  // D = dx^2/4
      case KernelType::BSPLINE2: c = 1.0 / 3.0;  break;  // D = dx^2/3
      case KernelType::BSPLINE3: c = 3.0 / 16.0; break;  // D = 3dx^2/16
      default:                   c = 1.0 / 3.0;  break;  // safe fallback
    }
    for (int d = 0; d < dim; d++)
      Dinv[d] = 1.0 / (c * dx_d[d] * dx_d[d]);
    if (dim == 2) Dinv[2] = 0.0;
  }

  // Evaluate 1D weight and gradient (in normalized coordinates r = (x_p - x_n) / dx)
  inline void eval_1d(double r, double& w, double& dw) const {
    switch (type) {
      case KernelType::LINEAR:
        w = basis_linear(r);
        dw = grad_linear(r);
        return;
      case KernelType::BSPLINE2:
        w = basis_bspline2(r);
        dw = grad_bspline2(r);
        return;
      case KernelType::BSPLINE3:
        w = basis_bspline3(r);
        dw = grad_bspline3(r);
        return;
    }
  }

  // Evaluate 3D weight and gradient for particle at x_p, node at x_n.
  // grad is in physical coordinates (divided by dx).
  // Returns weight.
  double evaluate(const double x_p[3], const double x_n[3],
                  const double dx_d[3], int dim,
                  double& weight, double grad[3]) const {
    double wx, wy, wz = 1.0;
    double dwx, dwy, dwz = 0.0;

    double rx = (x_p[0] - x_n[0]) / dx_d[0];
    double ry = (x_p[1] - x_n[1]) / dx_d[1];
    eval_1d(rx, wx, dwx);
    eval_1d(ry, wy, dwy);

    if (dim == 3) {
      double rz = (x_p[2] - x_n[2]) / dx_d[2];
      eval_1d(rz, wz, dwz);
    }

    weight = wx * wy * wz;
    grad[0] = dwx * wy * wz / dx_d[0];
    grad[1] = wx * dwy * wz / dx_d[1];
    grad[2] = (dim == 3) ? (wx * wy * dwz / dx_d[2]) : 0.0;

    return weight;
  }

  // Get the range of grid node indices that a particle at position x_p
  // can influence. Returns [lo, hi] inclusive in each dimension.
  // Indices are in ghost-inclusive grid coordinates.
  //
  // x_p: particle position (physical)
  // domain_lo: physical coordinate of global grid origin
  // dx_d: grid spacing per dimension
  // offset: subdomain offset in global grid
  // ghost_w: ghost layer width
  //
  // Returns false if the particle is completely outside the local grid.
  bool support_range(const double x_p[3],
                     const double domain_lo[3],
                     const double dx_d[3],
                     const int offset_d[3],
                     int ghost_w, int dim_,
                     int gx, int gy, int gz,
                     int ilo[3], int ihi[3]) const {
    // Particle position in local ghost-inclusive grid coordinates
    for (int d = 0; d < 3; d++) {
      if (d >= dim_ && dim_ == 2) {
        ilo[d] = 0; ihi[d] = 0;
        continue;
      }
      double rp = (x_p[d] - domain_lo[d]) / dx_d[d] - offset_d[d] + ghost_w;

      int lo, hi;
      switch (type) {
        case KernelType::LINEAR:
          lo = (int)std::floor(rp);
          hi = lo + 1;
          break;
        case KernelType::BSPLINE2:
          // Nearest node, then ±1
          lo = (int)std::round(rp) - 1;
          hi = lo + 2;
          break;
        case KernelType::BSPLINE3:
          lo = (int)std::floor(rp) - 1;
          hi = lo + 3;
          break;
        default:
          lo = (int)std::round(rp) - 1;
          hi = lo + 2;
      }

      // Clamp to grid bounds
      const int gmax = (d == 0) ? gx - 1 : (d == 1) ? gy - 1 : gz - 1;
      ilo[d] = (lo < 0) ? 0 : lo;
      ihi[d] = (hi > gmax) ? gmax : hi;

      if (ilo[d] > ihi[d]) return false;
    }
    return true;
  }
};

} // namespace CoupMPM
} // namespace LAMMPS_NS

#endif // COUPMPM_KERNEL_H
