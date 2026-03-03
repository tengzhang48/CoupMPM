# Theory: Core MPM and Affine Particle-In-Cell (APIC) Transfers

## 1. Overview

The Material Point Method (MPM) is an Eulerian–Lagrangian numerical method in which continuum bodies are discretized into a set of Lagrangian material points (particles) carrying all state variables — mass, velocity, deformation gradient, and internal stress — while a background Eulerian grid serves as a scratch-space for solving the equations of motion. At each timestep, particle data are mapped to the grid (Particle-to-Grid, **P2G**), the grid momentum equations are integrated, and the updated grid velocities are mapped back to the particles (Grid-to-Particle, **G2P**).

CoupMPM implements the **Affine Particle-In-Cell (APIC)** formulation of Jiang et al. (2015, 2017). Each particle carries, in addition to its translational velocity $\mathbf{v}_p$, a $3\times3$ affine velocity matrix $\mathbf{C}_p$ (stored in the `Bp` array) that encodes the locally affine structure of the velocity field in the particle's neighbourhood. The APIC formulation is provably angular-momentum conserving and free of the numerical dissipation that afflicts classical PIC while remaining more stable than FLIP.

---

## 2. Governing Equations

The quasi-static and dynamic behaviour of a continuum body $\Omega$ is governed by the balance of linear momentum:

$$
\rho \dot{\mathbf{v}} = \nabla \cdot \boldsymbol{\sigma} + \rho \mathbf{b}, \quad \mathbf{x} \in \Omega,
$$

where $\rho$ is the current mass density, $\mathbf{v}$ the velocity field, $\boldsymbol{\sigma}$ the Cauchy stress tensor, and $\mathbf{b}$ the body force per unit mass. The weak (Galerkin) form of this equation over a test domain $\Omega$ reads:

$$
\int_\Omega \rho \dot{\mathbf{v}} \cdot \boldsymbol{\eta} \, dV
= - \int_\Omega \boldsymbol{\sigma} : \nabla \boldsymbol{\eta} \, dV
+ \int_{\partial \Omega} \boldsymbol{\sigma} \mathbf{n} \cdot \boldsymbol{\eta} \, dA
+ \int_\Omega \rho \mathbf{b} \cdot \boldsymbol{\eta} \, dV
\quad \forall \boldsymbol{\eta},
$$

where integration by parts has been applied to the divergence term. In the MPM discretization, the particle quadrature rule replaces the volume integrals, and the grid shape functions provide the test and trial spaces.

---

## 3. Shape Functions and the Kernel $w_{ip}$

Let $\mathbf{x}_p$ be the position of particle $p$ and $\mathbf{x}_i$ the position of grid node $i$. The nodal weight $w_{ip}$ and its gradient $\nabla w_{ip}$ are evaluated as a tensor product of one-dimensional basis functions $N$ in each spatial direction:

$$
w_{ip} = \prod_{d \in \{x,y,z\}} N\!\left(\frac{x_p^{(d)} - x_i^{(d)}}{\Delta x^{(d)}}\right),
\qquad
\frac{\partial w_{ip}}{\partial x^{(d)}} = \frac{1}{\Delta x^{(d)}} N'\!\left(\frac{x_p^{(d)} - x_i^{(d)}}{\Delta x^{(d)}}\right) \prod_{e \neq d} N\!\left(\frac{x_p^{(e)} - x_i^{(e)}}{\Delta x^{(e)}}\right),
$$

where $\Delta x^{(d)}$ is the grid spacing in direction $d$. Three kernel choices are implemented in `coupmpm_kernel.h`:

| Kernel | Order | 1-D support | $N(r)$ |
|---|---|---|---|
| Linear | $C^0$ | $\lvert r \rvert < 1$ | $1 - \lvert r \rvert$ |
| Quadratic B-spline | $C^1$ | $\lvert r \rvert < 1.5$ | $\tfrac{3}{4} - r^2$ for $\lvert r \rvert < \tfrac{1}{2}$; $\tfrac{1}{2}({\tfrac{3}{2} - \lvert r \rvert})^2$ for $\tfrac{1}{2} \le \lvert r \rvert < \tfrac{3}{2}$ |
| Cubic B-spline | $C^2$ | $\lvert r \rvert < 2$ | $\tfrac{1}{2}\lvert r \rvert^3 - r^2 + \tfrac{2}{3}$ for $\lvert r \rvert < 1$; $\tfrac{1}{6}(2 - \lvert r \rvert)^3$ for $1 \le \lvert r \rvert < 2$ |

The weights satisfy the partition-of-unity property $\sum_i w_{ip} = 1$ and the support of each basis is strictly local, ensuring a sparse interaction stencil (at most $2^3$, $3^3$, or $4^3$ nodes for linear, quadratic, and cubic kernels, respectively).

---

## 4. Particle-to-Grid (P2G) Transfer

Prior to each P2G step, all grid fields are zeroed. Contributions from all particles are then accumulated on the grid.

### 4.1 Mass Mapping

The discrete mass at grid node $i$ is

$$
m_i = \sum_p w_{ip} \, m_p,
$$

where $m_p$ is the mass of particle $p$. This is a direct weak-form quadrature of the density field using particles as integration points.

In `p2g()`:
```
grid.mass[n] += w * mp;
```

### 4.2 APIC Momentum Mapping

In the classical PIC scheme the momentum transferred to node $i$ is $m_p w_{ip} \mathbf{v}_p$, which dissipates angular momentum. APIC augments the particle velocity with a locally affine correction: each particle $p$ carries a $3\times3$ affine matrix $\mathbf{C}_p$ such that the effective velocity at an arbitrary position $\mathbf{x}$ near the particle is

$$
\tilde{\mathbf{v}}_p(\mathbf{x}) = \mathbf{v}_p + \mathbf{C}_p \, (\mathbf{x} - \mathbf{x}_p).
$$

The APIC P2G momentum transfer evaluates this affine velocity at each node position:

$$
(m\mathbf{v})_i = \sum_p w_{ip} \, m_p \left[ \mathbf{v}_p + \mathbf{C}_p \, (\mathbf{x}_i - \mathbf{x}_p) \right].
$$

In component form, the $\alpha$-component of momentum at node $i$ is:

$$
(mv_\alpha)_i = \sum_p w_{ip} \, m_p \left( v_p^\alpha + \sum_{\beta} C_p^{\alpha\beta} \, (x_i^\beta - x_p^\beta) \right).
$$

In `p2g()`:
```cpp
double v_apic_0 = vp[0];
for (int e = 0; e < 3; e++)
    v_apic_0 += Cp[0*3 + e] * dx_pn[e];   // dx_pn = x_node - x_p
grid.momentum_x[n] += w * mp * v_apic_0;
```

### 4.3 Internal Force (Stress Divergence)

The discrete internal nodal force is obtained by the standard MPM quadrature of the weak-form stress-divergence term:

$$
\mathbf{f}_i^{\mathrm{int}} = - \sum_p V_p \, \boldsymbol{\sigma}_p \cdot \nabla w_{ip},
$$

where $V_p = V_p^0 \, J_p$ is the current volume of particle $p$, $V_p^0$ its reference volume, $J_p = \det(\mathbf{F}_p)$ the Jacobian of the deformation gradient, and $\boldsymbol{\sigma}_p$ the symmetric Cauchy stress tensor at particle $p$. In index notation:

$$
f_i^{\mathrm{int},\alpha} = - \sum_p V_p \sum_{\beta} \sigma_p^{\alpha\beta} \, \frac{\partial w_{ip}}{\partial x^\beta}.
$$

The Cauchy stress is stored internally in full $3\times3$ form, reconstructed from the six-component Voigt vector $[\sigma_{xx}, \sigma_{yy}, \sigma_{zz}, \sigma_{xy}, \sigma_{xz}, \sigma_{yz}]$.

In `p2g()`:
```cpp
for (int e = 0; e < 3; e++)
    fi_x -= vol_p * sigma[0*3 + e] * grad_w[e];
grid.force_int_x[n] += fi_x;
```

### 4.4 External Force Mapping

Particle-level external forces $\mathbf{f}_p$ (arising from LAMMPS pair interactions and applied fixes) are mapped to the grid by the same shape-function quadrature:

$$
\mathbf{f}_i^{\mathrm{ext}} = \sum_p w_{ip} \, \mathbf{f}_p.
$$

In `p2g()`:
```cpp
grid.force_ext_x[n] += w * fp[0];
```

### 4.5 B-Bar Volumetric Strain Rate Accumulation

To suppress volumetric locking in nearly incompressible materials, the B-bar method (Hughes 1980) accumulates a mass-weighted velocity divergence on the grid during P2G:

$$
\hat{d}_i = \sum_p m_p \, \nabla w_{ip} \cdot \tilde{\mathbf{v}}_p(\mathbf{x}_i),
$$

where the velocity is evaluated with the APIC affine correction. After ghost exchange, this field is normalized by the nodal mass to yield the smoothed divergence $\bar{d}_i = \hat{d}_i / m_i$, which is broadcast to particles during G2P.

In `p2g()`:
```cpp
double v_dot_grad = 0.0;
for (int d = 0; d < 3; d++) {
    double v_apic_d = vp[d];
    for (int e = 0; e < 3; e++)
        v_apic_d += Cp[3*d + e] * dx_pn[e];
    v_dot_grad += v_apic_d * grad_w[d];
}
grid.raw_div_v[n] += mp * v_dot_grad;
```

---

## 5. Grid Equations of Motion

After all particle contributions have been accumulated and ghost-node sums have been communicated (reverse MPI exchange), each grid node $i$ with non-zero mass is updated by the explicit forward-Euler integration:

$$
m_i \mathbf{v}_i^{n+1} = (m\mathbf{v})_i^n + \Delta t \left( \mathbf{f}_i^{\mathrm{int}} + \mathbf{f}_i^{\mathrm{ext}} \right),
$$

$$
\mathbf{v}_i^{n+1} = \frac{(m\mathbf{v})_i^n}{m_i} + \frac{\Delta t}{m_i} \left( \mathbf{f}_i^{\mathrm{int}} + \mathbf{f}_i^{\mathrm{ext}} \right).
$$

Boundary conditions and multi-body contact corrections are applied to $\mathbf{v}_i^{n+1}$ before the G2P step.

---

## 6. Grid-to-Particle (G2P) Transfer

### 6.1 Velocity Update

The new particle velocity is interpolated from the updated grid velocities:

$$
\mathbf{v}_p^{n+1} = \sum_i w_{ip} \, \mathbf{v}_i^{n+1}.
$$

In `g2p()`:
```cpp
for (int d = 0; d < 3; d++)
    v_new[d] += w * vn[d];
```

### 6.2 APIC Affine Matrix Construction

The central operation of the APIC G2P step is the reconstruction of the particle's affine matrix $\mathbf{C}_p^{n+1}$. It is defined as the second moment of the grid velocity field with respect to the particle position, weighted by the shape functions and scaled by the inverse of the particle inertia tensor $\mathbf{D}_p^{-1}$:

$$
\mathbf{C}_p^{n+1} = \left( \sum_i w_{ip} \, \mathbf{v}_i^{n+1} \otimes (\mathbf{x}_i - \mathbf{x}_p) \right) \mathbf{D}_p^{-1}.
$$

In index notation:

$$
C_p^{\alpha\beta} = \sum_i w_{ip} \, v_i^\alpha \, (x_i^\beta - x_p^\beta) \, (D_p^{-1})^{\beta\beta},
$$

where $\mathbf{D}_p^{-1}$ is diagonal for tensor-product grids (see Section 7).

In `g2p()`:
```cpp
for (int d = 0; d < 3; d++)
    for (int e = 0; e < 3; e++)
        Bp_new[3*d + e] += w * vn[d] * dx_pn[e] * Dinv[e];
```

Here `Dinv[e]` is $(D_p^{-1})^{ee}$, `dx_pn[e]` $= x_i^e - x_p^e$, and the result is stored directly as the `Bp` array.

### 6.3 Velocity Gradient

The velocity gradient tensor at each particle is computed by the same shape-function interpolation used in the finite-element method:

$$
\mathbf{L}_p = \sum_i \mathbf{v}_i^{n+1} \otimes \nabla w_{ip},
\qquad
L_p^{\alpha\beta} = \sum_i v_i^\alpha \, \frac{\partial w_{ip}}{\partial x^\beta}.
$$

In `g2p()`:
```cpp
for (int d = 0; d < 3; d++)
    for (int e = 0; e < 3; e++)
        L[3*d + e] += vn[d] * grad_w[e];
```

This velocity gradient is subsequently used to update the deformation gradient.

### 6.4 B-Bar Smoothed Divergence

The nodal smoothed divergence $\bar{d}_i$ computed during P2G (Section 4.5) is interpolated back to each particle:

$$
\bar{d}_p = \sum_i w_{ip} \, \bar{d}_i.
$$

In `g2p()`:
```cpp
if (use_bbar)
    dv_smooth += w * grid.div_v[n];
```

---

## 7. The APIC Affine Matrix $\mathbf{C}_p$ and Angular Momentum Conservation

### 7.1 Particle Inertia Tensor $\mathbf{D}_p$

The APIC formulation requires defining the particle inertia tensor:

$$
\mathbf{D}_p = \sum_i w_{ip} \, (\mathbf{x}_i - \mathbf{x}_p) \otimes (\mathbf{x}_i - \mathbf{x}_p).
$$

For a uniform Cartesian grid, $\mathbf{D}_p$ is diagonal with entries $D_p^{dd} = c_k \, (\Delta x^{(d)})^2$ where the kernel-dependent constant $c_k$ is:

| Kernel | $c_k$ | $D_p^{dd}$ | $(D_p^{-1})^{dd}$ |
|---|---|---|---|
| Linear | $\tfrac{1}{4}$ | $\tfrac{(\Delta x^{(d)})^2}{4}$ | $\tfrac{4}{(\Delta x^{(d)})^2}$ |
| Quadratic B-spline | $\tfrac{1}{3}$ | $\tfrac{(\Delta x^{(d)})^2}{3}$ | $\tfrac{3}{(\Delta x^{(d)})^2}$ |
| Cubic B-spline | $\tfrac{3}{16}$ | $\tfrac{3(\Delta x^{(d)})^2}{16}$ | $\tfrac{16}{3(\Delta x^{(d)})^2}$ |

In `coupmpm_kernel.h`, `D_inverse()` computes `Dinv[d] = 1.0 / (c * dx_d[d] * dx_d[d])` for each kernel type.

### 7.2 Angular Momentum Conservation

Jiang et al. (2017) proved that the combined P2G–G2P cycle with affine reconstruction is **exactly** angular momentum conserving under the following consistency condition between the P2G momentum transfer and the G2P affine reconstruction:

The P2G momentum transfer (Equation in Section 4.2) can be expanded as:

$$
(m\mathbf{v})_i = \sum_p m_p w_{ip} \mathbf{v}_p + \sum_p m_p w_{ip} \mathbf{C}_p (\mathbf{x}_i - \mathbf{x}_p).
$$

The G2P velocity reconstruction (Section 6.1) gives:

$$
\mathbf{v}_p^{n+1} = \sum_i w_{ip} \mathbf{v}_i^{n+1},
$$

and the G2P affine reconstruction (Section 6.2) gives:

$$
\mathbf{C}_p^{n+1} = \left(\sum_i w_{ip} \, \mathbf{v}_i^{n+1} \otimes (\mathbf{x}_i - \mathbf{x}_p)\right) \mathbf{D}_p^{-1}.
$$

The key identity that guarantees angular momentum conservation is:

$$
\sum_p m_p \mathbf{C}_p^{n+1} \mathbf{D}_p = \sum_p m_p \sum_i w_{ip} \mathbf{v}_i^{n+1} \otimes (\mathbf{x}_i - \mathbf{x}_p),
$$

which ensures that the affine moment of the particle velocity field exactly reproduces the second moment of the grid velocity field. When used in the subsequent P2G step, this guarantees that the nodal angular momentum $\mathbf{L}_i = \sum_p m_p w_{ip} (\mathbf{x}_i - \mathbf{x}_p) \times \tilde{\mathbf{v}}_p(\mathbf{x}_i)$ is preserved through each P2G–G2P cycle without dissipation.

In contrast, classical PIC (which sets $\mathbf{C}_p = \mathbf{0}$) transfers only translational momentum $m_p \mathbf{v}_p$ to the grid, losing all information about the sub-cell angular structure of the velocity field at each transfer step. This causes the well-known angular-momentum dissipation in PIC.

---

## 8. B-Bar Volumetric Anti-Locking

The standard MPM velocity gradient $\mathbf{L}_p$ from Section 6.3 suffers from volumetric locking when the material is nearly incompressible. The B-bar method corrects the volumetric part of $\mathbf{L}_p$ by replacing the local divergence $\mathrm{tr}(\mathbf{L}_p)$ with the smoothed nodal divergence $\bar{d}_p$ interpolated from the grid:

$$
\bar{\mathbf{L}}_p = \mathbf{L}_p + \frac{\bar{d}_p - \mathrm{tr}(\mathbf{L}_p)}{3} \mathbf{I},
$$

where $\mathbf{I}$ is the $3\times3$ identity tensor. This correction adds a uniform volumetric strain-rate increment to $\mathbf{L}_p$ to match the smoothed nodal divergence while leaving the deviatoric part unchanged.

In `update_F_and_stress()`:
```cpp
double tr_L = Lp[0] + Lp[4] + Lp[8];
double correction = (div_v_s[p] - tr_L) / 3.0;
L_bar[0] += correction;
L_bar[4] += correction;
L_bar[8] += correction;
```

---

## 9. Deformation Gradient Update

After the B-bar–corrected velocity gradient $\bar{\mathbf{L}}_p$ is obtained, the deformation gradient is updated by the multiplicative forward-Euler scheme:

$$
\mathbf{F}_p^{n+1} = \left(\mathbf{I} + \Delta t \, \bar{\mathbf{L}}_p\right) \mathbf{F}_p^n.
$$

The current volume is $V_p = V_p^0 \, J_p^{n+1}$ with $J_p^{n+1} = \det(\mathbf{F}_p^{n+1})$, and the Cauchy stress $\boldsymbol{\sigma}_p^{n+1}$ is then computed from the constitutive model (Neo-Hookean or Mooney-Rivlin) using $\mathbf{F}_p^{n+1}$.

In `update_F_and_stress()`:
```cpp
Mat3::update_F(Fp, L_bar, dt, F_new);  // F_new = (I + dt*L_bar)*F_old
```

---

## 10. Summary of the Per-Timestep Algorithm

The complete APIC MPM timestep in CoupMPM proceeds as follows:

1. **Zero grid**: Reset all nodal fields ($m_i$, $(m\mathbf{v})_i$, $\mathbf{f}_i^{\mathrm{int}}$, $\mathbf{f}_i^{\mathrm{ext}}$, $\hat{d}_i$) to zero.
2. **P2G** (`p2g()`): For each particle $p$ and each node $i$ in its support:
   - Accumulate mass: $m_i \mathrel{+}= w_{ip} m_p$
   - Accumulate APIC momentum: $(m v_\alpha)_i \mathrel{+}= w_{ip} m_p \left(v_p^\alpha + \sum_\beta C_p^{\alpha\beta}(x_i^\beta - x_p^\beta)\right)$
   - Accumulate internal force: $f_i^{\mathrm{int},\alpha} \mathrel{-}= V_p \sum_\beta \sigma_p^{\alpha\beta} \partial_\beta w_{ip}$
   - Accumulate external force: $\mathbf{f}_i^{\mathrm{ext}} \mathrel{+}= w_{ip} \mathbf{f}_p$
   - (If B-bar) Accumulate raw divergence: $\hat{d}_i \mathrel{+}= m_p \nabla w_{ip} \cdot \tilde{\mathbf{v}}_p(\mathbf{x}_i)$
3. **Reverse MPI exchange**: Sum ghost-node contributions onto owner nodes.
4. **Grid solve**: Compute $\mathbf{v}_i^{n+1} = (m\mathbf{v})_i / m_i + (\Delta t / m_i)(\mathbf{f}_i^{\mathrm{int}} + \mathbf{f}_i^{\mathrm{ext}})$; normalize B-bar: $\bar{d}_i = \hat{d}_i / m_i$.
5. **Contact / boundary conditions**: Apply velocity corrections to $\mathbf{v}_i^{n+1}$.
6. **Forward MPI exchange**: Broadcast $\mathbf{v}_i^{n+1}$ and $\bar{d}_i$ to ghost nodes.
7. **G2P** (`g2p()`): For each particle $p$ and each node $i$ in its support:
   - Interpolate velocity: $\mathbf{v}_p^{n+1} \mathrel{+}= w_{ip} \mathbf{v}_i^{n+1}$
   - Reconstruct affine matrix: $C_p^{\alpha\beta} \mathrel{+}= w_{ip} v_i^\alpha (x_i^\beta - x_p^\beta) (D_p^{-1})^{\beta\beta}$
   - Compute velocity gradient: $L_p^{\alpha\beta} \mathrel{+}= v_i^\alpha \partial_\beta w_{ip}$
   - (If B-bar) Interpolate smoothed divergence: $\bar{d}_p \mathrel{+}= w_{ip} \bar{d}_i$
8. **F and stress update** (`update_F_and_stress()`): Apply B-bar correction, advance $\mathbf{F}_p$, evaluate constitutive model.
9. **Particle position update**: $\mathbf{x}_p^{n+1} = \mathbf{x}_p^n + \Delta t \, \mathbf{v}_p^{n+1}$.

---

## References

- Jiang, C., Schroeder, C., Selle, A., Teran, J., & Stomakhin, A. (2015). The affine particle-in-cell method. *ACM Transactions on Graphics*, 34(4), 51. https://doi.org/10.1145/2766996
- Jiang, C., Schroeder, C., & Teran, J. (2017). An angular momentum conserving affine-particle-in-cell method. *Journal of Computational Physics*, 338, 137–164. https://doi.org/10.1016/j.jcp.2017.02.050
- Sulsky, D., Chen, Z., & Schreyer, H. L. (1994). A particle method for history-dependent materials. *Computer Methods in Applied Mechanics and Engineering*, 118(1–2), 179–196. https://doi.org/10.1016/0045-7825(94)90112-0
- Hughes, T. J. R. (1980). Generalization of selective integration procedures to anisotropic and nonlinear media. *International Journal for Numerical Methods in Engineering*, 15(9), 1413–1418. https://doi.org/10.1002/nme.1620150914
- Bardenhagen, S. G., Brackbill, J. U., & Sulsky, D. (2000). The material-point method for granular materials. *Computer Methods in Applied Mechanics and Engineering*, 187(3–4), 529–541. https://doi.org/10.1016/S0045-7825(99)00338-2
