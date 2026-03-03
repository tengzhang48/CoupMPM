# Theory: B-Bar Volumetric Anti-Locking in MPM

## 1. Overview

Nearly incompressible materials — rubber-like elastomers, biological soft tissues, and metals
deforming plastically — obey a near-isochoric constraint: the volumetric deformation
$J = \det(\mathbf{F}) \approx 1$ at all times. When such materials are discretised with
low-order interpolation (linear shape functions, or the equivalent in MPM), the discrete
velocity field cannot simultaneously satisfy the momentum equations and the volumetric
constraint everywhere. The result is **volumetric locking**: the solution is artificially
stiff, displacements are severely underestimated, and pressure fields exhibit spurious
oscillations.

CoupMPM resolves this pathology with the **B-bar method** originally proposed by Hughes
(1980) for finite elements and adapted here to the APIC-MPM context. The core idea is to
replace the *local*, particle-level volumetric strain rate — which is prone to locking —
with a *smoothed*, grid-level estimate that distributes the incompressibility constraint
more uniformly across the support neighbourhood. The resulting corrected velocity gradient
$\bar{\mathbf{L}}_p$ is then used to advance the deformation gradient $\mathbf{F}_p$,
ensuring that the discrete formulation remains nearly locking-free even for fully
incompressible constitutive models.

---

## 2. The Origin of Volumetric Locking

### 2.1 Incompressibility Constraint

For a hyperelastic material with bulk modulus $\kappa \gg \mu$ (shear modulus), the
Cauchy stress decomposes into deviatoric and volumetric parts:

$$
\boldsymbol{\sigma} = \boldsymbol{\sigma}^{\mathrm{dev}} + p \, \mathbf{I},
\qquad
p = \kappa \, (J - 1),
$$

where $p$ is the hydrostatic pressure and $J = \det(\mathbf{F})$. In the incompressible
limit $\kappa \to \infty$, the kinematic constraint $J = 1$ must be enforced exactly; for
finite but large $\kappa$, even a small error in $J$ generates an enormous pressure, which
pollutes the entire stress field.

### 2.2 Why Discrete Velocity Fields Lock

The discretised rate of volume change at particle $p$ is given by the divergence of the
interpolated velocity field:

$$
\dot{J}_p / J_p = \mathrm{tr}(\mathbf{L}_p) = \nabla \cdot \mathbf{v} \big|_p
= \sum_i v_i^\alpha \, \frac{\partial w_{ip}}{\partial x^\alpha},
$$

where $w_{ip}$ are the MPM shape-function weights, $\mathbf{v}_i$ the grid nodal
velocities, and summation over the repeated Greek index $\alpha$ is implied.

Because MPM (like low-order FEM) uses the *same* shape functions for both the momentum
equation and the incompressibility constraint, the two conditions over-constrain the nodal
degrees of freedom in the incompressible limit. In practice this locks the volumetric mode:
the stiffness matrix develops a near-null space that forces the displacement solution
toward zero. The locking is most severe for linear kernels and diminishes but does not
vanish for higher-order B-splines.

---

## 3. B-Bar Formulation

### 3.1 Classical B-Bar Idea

Hughes (1980) showed that locking can be eliminated — or greatly reduced — by replacing the
*local* volumetric strain operator $\mathbf{B}_{\mathrm{vol}}$ with a *smoothed* operator
$\bar{\mathbf{B}}_{\mathrm{vol}}$ evaluated at a reduced integration point (the element
centroid in FEM). In the MPM context this translates to replacing the local particle-level
velocity divergence $\mathrm{tr}(\mathbf{L}_p)$ with a smoothed counterpart $\bar{d}_p$
computed by mass-weighted averaging over the grid stencil.

### 3.2 Smoothed Nodal Velocity Divergence

#### Step 1 — P2G accumulation of raw, mass-weighted divergence

During the Particle-to-Grid (P2G) step, each particle $p$ contributes to a *raw*
(un-normalised) divergence field on the grid. The particle velocity evaluated at node $i$
with the APIC affine correction is

$$
\tilde{\mathbf{v}}_p(\mathbf{x}_i)
= \mathbf{v}_p + \mathbf{C}_p \, (\mathbf{x}_i - \mathbf{x}_p),
$$

where $\mathbf{v}_p$ is the particle translational velocity and $\mathbf{C}_p$ is the APIC
affine matrix. The APIC-augmented velocity divergence contribution from particle $p$ to
node $i$ is

$$
\hat{d}_i \;\mathrel{+}=\; m_p \, \nabla w_{ip} \cdot \tilde{\mathbf{v}}_p(\mathbf{x}_i)
= m_p \sum_{\alpha} \frac{\partial w_{ip}}{\partial x^\alpha}
  \left[ v_p^\alpha + \sum_{\beta} C_p^{\alpha\beta}(x_i^\beta - x_p^\beta) \right].
$$

Accumulating over all particles in the support of node $i$ gives the unnormalised nodal
quantity

$$
\hat{d}_i = \sum_p m_p \, \nabla w_{ip} \cdot \tilde{\mathbf{v}}_p(\mathbf{x}_i).
$$

In `p2g()` this reads:
```cpp
// B-bar: accumulate mass-weighted velocity divergence
if (use_bbar) {
  double v_dot_grad = 0.0;
  for (int d = 0; d < 3; d++) {
    double v_apic_d = vp[d];
    for (int e = 0; e < 3; e++)
      v_apic_d += Cp[3*d + e] * dx_pn[e];
    v_dot_grad += v_apic_d * grad_w[d];
  }
  grid.raw_div_v[n] += mp * v_dot_grad;
}
```

#### Step 2 — Ghost exchange and mass normalisation

After all P2G contributions have been accumulated, a reverse (ghost → owner) MPI
communication sums ghost-node contributions onto owner nodes. The smoothed nodal divergence
is then obtained by normalising with the nodal mass:

$$
\bar{d}_i = \frac{\hat{d}_i}{m_i}, \qquad m_i > 0.
$$

This normalisation is equivalent to a mass-weighted spatial average of the local velocity
divergences over the support neighbourhood of node $i$, providing the desired smoothing
effect. The normalised field is stored in `grid.div_v[n]` and broadcast to ghost nodes via
a forward MPI exchange before the G2P step.

### 3.3 Particle Smoothed Divergence — G2P Interpolation

During the Grid-to-Particle (G2P) step, the smoothed nodal divergence is interpolated back
to each particle using the same shape functions:

$$
\bar{d}_p = \sum_i w_{ip} \, \bar{d}_i.
$$

This gives a particle-level smoothed velocity divergence that reflects the average
volumetric behaviour of the neighbourhood rather than the purely local value derived from
the velocity gradient. In `g2p()`:

```cpp
// B-bar smoothed divergence
if (use_bbar)
  dv_smooth += w * grid.div_v[n];
```

---

## 4. Isochoric Correction to the Velocity Gradient

### 4.1 Corrected Velocity Gradient $\bar{\mathbf{L}}_p$

Given the particle's raw velocity gradient $\mathbf{L}_p$ (computed from the standard
shape-function interpolation in G2P) and the smoothed divergence $\bar{d}_p$, the B-bar
corrected velocity gradient is

$$
\bar{\mathbf{L}}_p = \mathbf{L}_p
+ \underbrace{\frac{\bar{d}_p - \mathrm{tr}(\mathbf{L}_p)}{3}}_{\delta d_p} \mathbf{I},
$$

where $\mathrm{tr}(\mathbf{L}_p) = L_p^{11} + L_p^{22} + L_p^{33}$ is the local velocity
divergence, and $\mathbf{I}$ is the $3\times3$ identity tensor.

The scalar correction $\delta d_p = (\bar{d}_p - \mathrm{tr}(\mathbf{L}_p)) / 3$ is added
equally to all three diagonal entries of $\mathbf{L}_p$, so that

$$
\mathrm{tr}(\bar{\mathbf{L}}_p) = \mathrm{tr}(\mathbf{L}_p) + 3\,\delta d_p = \bar{d}_p.
$$

The off-diagonal (deviatoric shear) components are left entirely unchanged:
$\bar{L}_p^{\alpha\beta} = L_p^{\alpha\beta}$ for $\alpha \neq \beta$. The correction is
therefore **purely volumetric** — it alters only the isotropic part of the velocity
gradient while preserving the deviatoric kinematics.

In `update_F_and_stress()`:
```cpp
// B-bar correction
double L_bar[9];
std::memcpy(L_bar, Lp, 9 * sizeof(double));

if (use_bbar && div_v_s) {
  double tr_L = Lp[0] + Lp[4] + Lp[8];
  double correction = (div_v_s[p] - tr_L) / 3.0;
  L_bar[0] += correction;
  L_bar[4] += correction;
  L_bar[8] += correction;
}
```

### 4.2 Physical Interpretation

The additive split

$$
\bar{\mathbf{L}}_p = \underbrace{\mathbf{L}_p - \tfrac{1}{3}\mathrm{tr}(\mathbf{L}_p)\mathbf{I}}_{\text{deviatoric (unchanged)}}
+ \underbrace{\tfrac{1}{3}\bar{d}_p \, \mathbf{I}}_{\text{volumetric (smoothed)}}
$$

reveals that B-bar replaces only the volumetric part $\tfrac{1}{3}\mathrm{tr}(\mathbf{L}_p)\mathbf{I}$
with the grid-averaged equivalent $\tfrac{1}{3}\bar{d}_p \mathbf{I}$. The deviatoric
part $\mathbf{L}_p^{\mathrm{dev}} = \mathbf{L}_p - \tfrac{1}{3}\mathrm{tr}(\mathbf{L}_p)\mathbf{I}$
is computed at the standard MPM accuracy and remains unmodified.

This approach is directly analogous to the selective reduced integration technique in FEM:
the volumetric strain energy (which is responsible for the locking pressure constraint) is
evaluated at the coarser, grid-level resolution, while the deviatoric strain energy (which
carries the physically important shear stiffness) is evaluated at the full particle-level
resolution.

---

## 5. Deformation Gradient Update

With the B-bar corrected velocity gradient $\bar{\mathbf{L}}_p$ in hand, the deformation
gradient is updated by the explicit multiplicative forward-Euler integrator:

$$
\mathbf{F}_p^{n+1} = \left(\mathbf{I} + \Delta t \, \bar{\mathbf{L}}_p \right) \mathbf{F}_p^n.
$$

The Jacobian (volume ratio) at the new time is

$$
J_p^{n+1} = \det\!\left(\mathbf{F}_p^{n+1}\right),
$$

and the updated current particle volume is $V_p^{n+1} = V_p^0 \, J_p^{n+1}$, where
$V_p^0$ is the (fixed) reference volume. The Cauchy stress $\boldsymbol{\sigma}_p^{n+1}$
is subsequently evaluated from the chosen constitutive model (Neo-Hookean or
Mooney-Rivlin) using $\mathbf{F}_p^{n+1}$.

In `update_F_and_stress()`:
```cpp
// F_new = (I + dt * L_bar) * F_old
double F_new[9];
Mat3::update_F(Fp, L_bar, dt, F_new);
std::memcpy(Fp, F_new, 9 * sizeof(double));

// Compute stress
double J = Mat3::det(F_new);
constitutive.compute_stress(F_new, J, st, dt, sp);
```

---

## 6. Conservation Properties and Accuracy

### 6.1 Mass Conservation

The B-bar correction modifies only the velocity gradient used for the $\mathbf{F}$-update;
it does not alter the velocity field itself or the P2G momentum transfer. Consequently,
the total mass $M = \sum_p m_p$ is unchanged, and nodal mass conservation is unaffected.

### 6.2 Momentum Conservation

Because the corrected velocity gradient $\bar{\mathbf{L}}_p$ is derived entirely from the
same nodal velocities $\mathbf{v}_i^{n+1}$ that drive the P2G momentum mapping, global
linear momentum is conserved to machine precision.

### 6.3 Consistency with the Incompressibility Constraint

In the smooth (continuum) limit where every particle has identical neighbourhood
statistics, $\bar{d}_p \to \mathrm{tr}(\mathbf{L}_p)$ and the correction vanishes:
the B-bar formulation reduces to the standard MPM. The correction is active only where
the discrete velocity field exhibits spurious volumetric modes inconsistent with the
imposed incompressibility, which is precisely where locking would otherwise manifest.

### 6.4 Order of Accuracy

The smoothing operation $\hat{d}_i \to \bar{d}_i$ is a mass-weighted average over the
kernel support, which introduces an $\mathcal{O}(h^2)$ smoothing error (where $h$ is the
grid spacing) on smooth solutions. Since the standard MPM velocity gradient is itself
first-order in space for linear kernels and second-order for quadratic B-splines, the B-bar
correction does not reduce the asymptotic accuracy of the scheme; the overall spatial
accuracy is maintained at the same order as the underlying kernel.

---

## 7. Summary: B-Bar Algorithm per Timestep

The following steps are inserted into the standard APIC-MPM timestep to implement B-bar
anti-locking. Steps that are specific to B-bar are marked with **[B]**.

1. **P2G** — for each particle $p$ and each node $i$ in its support:
   - Standard contributions (mass, APIC momentum, internal and external forces).
   - **[B]** Accumulate raw divergence:
     $$
     \hat{d}_i \mathrel{+}= m_p \sum_\alpha \frac{\partial w_{ip}}{\partial x^\alpha}
     \left[ v_p^\alpha + \sum_\beta C_p^{\alpha\beta}(x_i^\beta - x_p^\beta) \right].
     $$

2. **Reverse MPI exchange** — sum ghost-node $\hat{d}_i$ onto owner nodes.

3. **Grid solve** — integrate momentum and **[B]** normalise:
   $$
   \bar{d}_i = \hat{d}_i \, / \, m_i.
   $$

4. **Forward MPI exchange** — broadcast $\mathbf{v}_i^{n+1}$ and **[B]** $\bar{d}_i$ to ghost nodes.

5. **G2P** — for each particle $p$ and each node $i$ in its support:
   - Standard contributions (velocity, APIC matrix, velocity gradient).
   - **[B]** Interpolate smoothed divergence:
     $$
     \bar{d}_p \mathrel{+}= w_{ip} \, \bar{d}_i.
     $$

6. **F and stress update** (`update_F_and_stress`) — **[B]** apply isochoric correction:
   $$
   \bar{\mathbf{L}}_p = \mathbf{L}_p + \frac{\bar{d}_p - \mathrm{tr}(\mathbf{L}_p)}{3}\,\mathbf{I},
   $$
   then advance the deformation gradient and evaluate the constitutive model:
   $$
   \mathbf{F}_p^{n+1} = \left(\mathbf{I} + \Delta t\,\bar{\mathbf{L}}_p\right)\mathbf{F}_p^n.
   $$

---

## References

- Hughes, T. J. R. (1980). Generalization of selective integration procedures to anisotropic
  and nonlinear media. *International Journal for Numerical Methods in Engineering*, 15(9),
  1413–1418. https://doi.org/10.1002/nme.1620150914
- Nagtegaal, J. C., Parks, D. M., & Rice, J. R. (1974). On numerically accurate finite
  element solutions in the fully plastic range. *Computer Methods in Applied Mechanics and
  Engineering*, 4(2), 153–177. https://doi.org/10.1016/0045-7825(74)90032-2
- de Souza Neto, E. A., Perić, D., Dutko, M., & Owen, D. R. J. (1996). Design of simple
  low order finite elements for large strain analysis of nearly incompressible solids.
  *International Journal of Solids and Structures*, 33(20–22), 3277–3296.
  https://doi.org/10.1016/0020-7683(95)00259-6
- Jiang, C., Schroeder, C., Selle, A., Teran, J., & Stomakhin, A. (2015). The affine
  particle-in-cell method. *ACM Transactions on Graphics*, 34(4), 51.
  https://doi.org/10.1145/2766996
- Coombs, W. M., Charlton, T. J., Cortis, M., & Augarde, C. E. (2018). Overcoming
  volumetric locking in material point methods. *Computer Methods in Applied Mechanics and
  Engineering*, 333, 1–21. https://doi.org/10.1016/j.cma.2018.01.010
- Sulsky, D., Chen, Z., & Schreyer, H. L. (1994). A particle method for history-dependent
  materials. *Computer Methods in Applied Mechanics and Engineering*, 118(1–2), 179–196.
  https://doi.org/10.1016/0045-7825(94)90112-0
