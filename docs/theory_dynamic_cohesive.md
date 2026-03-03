# Theory: Dynamic Particle-Based Cohesive Zones

## 1. Overview

Traditional cohesive zone methods in the Finite Element Method (FEM) require cohesive
interface elements to be inserted *a priori* along anticipated crack paths or cell
boundaries in a fixed reference mesh (Needleman 1987; Xu & Needleman 1994). This
dependence on a global reference configuration makes such methods unsuitable for problems
where the topology of adhesive contacts changes continuously — most prominently in
**cell sorting** (Steinberg 1963; Graner & Glazier 1992) and **tissue fracture** under
large motions.

CoupMPM implements a **Dynamic Particle-Based Cohesive Zone** (DPBCZ) framework that
overcomes this limitation. Cohesive bonds are not pre-specified; instead, they are formed
*at runtime* whenever two surface particles from different bodies come within a prescribed
formation distance. Each bond captures its own reference state (positions, normal,
reference area) at the moment of creation, so no global reference configuration is
required. Key consequences are:

- A single simulation can simultaneously handle **cell sorting** (bonds forming between
  cells that contact for the first time) and **tissue fracture** (bonds breaking under
  excessive load or fatigue).
- Bonds are history-dependent: each bond accumulates a scalar damage variable $D \in
  [0, 1]$ that irreversibly softens the traction.
- After complete failure ($D = 1$), the bond is deactivated and the pair transitions
  seamlessly to the frictional contact algorithm of Bardenhagen et al. (2000)
  implemented in `coupmpm_contact.h`.
- Type-pair parameters (`sigma_max`, `delta_n`, `base_rate`, …) allow **differential
  adhesion** — different cell-type pairs can have distinct binding strengths, reproducing
  the Differential Adhesion Hypothesis at the continuum scale.

The DPBCZ is implemented in `coupmpm_cohesive.h` and integrated into the MPM timestep
via `fix_coupmpm.cpp`.

---

## 2. Bond Lifecycle

### 2.1 Formation

Every `bond_check_interval` timesteps the method `detect_new_bonds()` scans all pairs
$(i, j)$ of surface particles from *different* bodies that are within the LAMMPS neighbor
list. A bond is created when

$$
\| \mathbf{x}_j - \mathbf{x}_i \| < d_{\mathrm{form}},
$$

where $d_{\mathrm{form}}$ is the type-pair formation distance. At the moment of creation
the bond stores its **reference state**:

$$
\mathbf{x}_{0,i}, \quad \mathbf{x}_{0,j}, \quad
\mathbf{n}_0 = \frac{\mathbf{x}_j - \mathbf{x}_i}{\|\mathbf{x}_j - \mathbf{x}_i\|},
\quad \delta_0 = \|\mathbf{x}_j - \mathbf{x}_i\|, \quad
A_{\mathrm{ref}} = (V_i^0)^{2/3} \text{ (3-D)}.
$$

### 2.2 Force Computation

At every timestep `compute_forces()` evaluates the traction on each active bond and
accumulates forces directly into the particle force array `f[]`. These forces enter the
MPM P2G step as external forces, exactly as LAMMPS pair-style interactions do.

The total cohesive force on particle $i$ is

$$
\mathbf{f}_i^{\mathrm{coh}}
= A_{\mathrm{ref}} \, (1 - D) \left[ T_n \, \hat{\mathbf{n}} + T_t \, \hat{\mathbf{t}} \right],
$$

and an equal and opposite force is applied to particle $j$, preserving Newton's Third Law.

### 2.3 Damage Update and Failure

After `compute_forces()`, `update_damage_and_break()` advances the damage variable and
deactivates any bond that has exceeded the failure criterion (Section 4.3 and 5.2).

---

## 3. Separation Decomposition

Let $\boldsymbol{\delta}$ denote the relative displacement of the bond since formation:

$$
\boldsymbol{\delta} = (\mathbf{x}_j - \mathbf{x}_i) - (\mathbf{x}_{0,j} - \mathbf{x}_{0,i}).
$$

The deformed bond normal $\hat{\mathbf{n}}$ is computed via the Nanson relation
(Section 6). The separation is then decomposed as

$$
\delta_n = \boldsymbol{\delta} \cdot \hat{\mathbf{n}},
\qquad
\boldsymbol{\delta}_t = \boldsymbol{\delta} - \delta_n \, \hat{\mathbf{n}},
\qquad
\delta_t = \|\boldsymbol{\delta}_t\|,
\qquad
\hat{\mathbf{t}} = \frac{\boldsymbol{\delta}_t}{\delta_t}.
$$

A positive $\delta_n > 0$ corresponds to opening (mode I); a non-zero $\delta_t$
corresponds to sliding (mode II/III).

---

## 4. Modified Needleman-Xu Traction-Separation Law

### 4.1 Cohesive Potential

The Needleman-Xu (1994) cohesive law derives from a mixed-mode potential energy density.
Following Xu & Needleman (1994), define the mode-I work of separation

$$
\phi_n = e \, \sigma_{\max} \, \delta_n^*,
$$

where $e = \exp(1)$, $\sigma_{\max}$ is the peak normal traction, and $\delta_n^*$ is the
characteristic normal separation. The normal and tangential ratios are

$$
\bar{\delta}_n = \frac{\delta_n}{\delta_n^*}, \qquad \bar{\delta}_t = \frac{\delta_t}{\delta_t^*}.
$$

### 4.2 Normal Traction

The normal traction is obtained from $T_n = -\partial \Phi / \partial \delta_n$:

$$
\boxed{
T_n = \frac{\phi_n}{\delta_n^*} \,
      \bar{\delta}_n \,
      \exp\!\left(-\bar{\delta}_n\right)
      \exp\!\left(-\bar{\delta}_t^{\,2}\right).
}
$$

This form peaks at $\bar{\delta}_n = 1$ (i.e., $\delta_n = \delta_n^*$), where $T_n =
\sigma_{\max}$, and decays exponentially for larger separations. The coupling factor
$\exp(-\bar{\delta}_t^{\,2})$ reduces the normal traction when sliding is present.

In `needleman_xu()`:
```cpp
const double phi_n = std::exp(1.0) * p.sigma_max * p.delta_n;
const double dn_ratio = delta_n / p.delta_n;
const double dt_ratio = delta_t / p.delta_t;
const double exp_n = std::exp(-dn_ratio);
const double exp_t = std::exp(-dt_ratio * dt_ratio);

T_n = (phi_n / p.delta_n) * dn_ratio * exp_n * exp_t;
```

### 4.3 Tangential Traction

The tangential traction is obtained from $T_t = -\partial \Phi / \partial \delta_t$. The
derivative of the coupling factor introduces a factor of $2\bar{\delta}_t / \delta_t^*$.
An additional factor of $(1 + \bar{\delta}_n)$ arises from the interaction term in the
full Xu-Needleman potential. The **negative sign** in the expression below ensures that
the tangential traction **opposes the slip** direction (i.e., it acts as a restoring
traction):

$$
\boxed{
T_t = -\frac{\phi_n}{\delta_t^*} \,
       2 \bar{\delta}_t \,
       \left(1 + \bar{\delta}_n\right)
       \exp\!\left(-\bar{\delta}_n\right)
       \exp\!\left(-\bar{\delta}_t^{\,2}\right),
\quad \delta_t > 0.
}
$$

This signed scalar magnitude is multiplied by the unit tangent $\hat{\mathbf{t}}$ when
assembling the force vector, so that the traction vector resists sliding in the actual
direction of slip.

In `needleman_xu()`:
```cpp
if (std::fabs(delta_t) > 1e-20) {
  T_t = -(phi_n / p.delta_t)
        * 2.0 * dt_ratio * (1.0 + dn_ratio) * exp_n * exp_t;
} else {
  T_t = 0.0;
}
```

### 4.4 Damage and Failure

Outside the traction laws, a scalar damage variable $D$ is evolved by
`update_damage_and_break()`. Defining the effective separation ratio

$$
\bar{r} = \sqrt{\left(\frac{\delta_n^{\max}}{\delta_n^*}\right)^2
              + \left(\frac{\delta_t^{\max}}{\delta_t^*}\right)^2},
$$

where $\delta_n^{\max}$ and $\delta_t^{\max}$ are the running maxima of the normal and
tangential separations, and the combined failure threshold

$$
\bar{r}_{\mathrm{fail}} = \sqrt{\left(\frac{\delta_n^{\max,\mathrm{fail}}}{\delta_n^*}\right)^2
                               + \left(\frac{\delta_t^{\max,\mathrm{fail}}}{\delta_t^*}\right)^2},
$$

damage accumulates for $\bar{r} > 1$ according to

$$
D = \min\!\left(1,\; \frac{\bar{r} - 1}{\bar{r}_{\mathrm{fail}} - 1}\right),
$$

and the effective traction is $(1-D) \, T$. The bond is deactivated when $D \geq 1$ or
when either $\delta_n^{\max} > \delta_n^{\max,\mathrm{fail}}$ or
$\delta_t^{\max} > \delta_t^{\max,\mathrm{fail}}$.

---

## 5. Bell Model for Receptor-Ligand Biological Adhesion

### 5.1 Physical Background

The Bell (1978) model describes the thermally activated unbinding of receptor-ligand
complexes under applied force. The unstressed off-rate $k_{\mathrm{off}}^0$ (bond
breaking per unit time in the absence of load) is accelerated exponentially by a
mechanical force $F$ acting along the unbinding reaction coordinate of length
$x_\beta$ (the *reactive compliance* or *bond compliance*):

$$
k_{\mathrm{off}}(F) = k_{\mathrm{off}}^0 \, \exp\!\left(\frac{F \, x_\beta}{k_B T}\right).
$$

For a deterministic continuum model (no Brownian fluctuations), the stochastic
bond-breaking rate is mapped onto a continuous damage rate by normalizing the applied
traction magnitude $T_{\mathrm{mag}}$ by the peak traction $\sigma_{\max}$. The
force-sensitive exponential is centred at $T_{\mathrm{mag}} / \sigma_{\max} = 0.5$ so
that bonds experience negligible damage at low loads and rapid damage near the peak
traction.

### 5.2 Damage Rate Equation

The damage rate $\dot{D}$ used in CoupMPM is

$$
\boxed{
\dot{D} = \mathtt{base\_rate} \cdot \exp\!\left(\alpha \left(\frac{T_{\mathrm{mag}}}{\sigma_{\max}} - 0.5\right)\right),
\quad T_{\mathrm{mag}} / \sigma_{\max} > 0.5,
}
$$

where $\alpha = 2$ is the force-sensitivity parameter (controlling the steepness of
the Bell exponential), $T_{\mathrm{mag}} = \sqrt{T_n^2 + T_t^2}$ is the instantaneous
traction magnitude, and `base_rate` ($= k_{\mathrm{off}}^0$) is the unstressed
off-rate (units: $\mathrm{time}^{-1}$). Damage is integrated forward in time by the
forward-Euler scheme $D^{n+1} = D^n + \dot{D} \, \Delta t$, capped at unity.

In `receptor_ligand()`:
```cpp
double alpha = 2.0;
double T_ratio = T_mag / p.sigma_max;

if (T_ratio > 0.5) {
  double damage_rate = p.base_rate * std::exp(alpha * (T_ratio - 0.5));
  damage += damage_rate * dt;
  if (damage > 1.0) damage = 1.0;
}
```

### 5.3 Traction Law

The elastic traction is a linear spring capped at the peak traction:

$$
T_n = \min\!\left(\frac{\sigma_{\max}}{\delta_n^*} \, \delta_n,\; \sigma_{\max}\right),
\qquad
T_t = \frac{\tau_{\max}}{\delta_t^*} \, \delta_t.
$$

As with the Needleman-Xu law, the effective tractions are scaled by $(1 - D)$.

### 5.4 Biological Parameters

| Parameter | Symbol | Typical value | Physical meaning |
|---|---|---|---|
| `sigma_max` | $\sigma_{\max}$ | $100$–$10{,}000\,\text{Pa}$ | Maximum receptor-ligand traction |
| `delta_n` | $\delta_n^*$ | $100$–$500$ nm | Characteristic receptor reach |
| `base_rate` | $k_{\mathrm{off}}^0$ | $0.1$–$10$ s$^{-1}$ | Thermal off-rate at zero force |
| `alpha` | $\alpha$ | $2$ | Force sensitivity (Bell parameter) |
| `formation_dist` | $d_{\mathrm{form}}$ | $2\delta_n^*$ | Bond-formation cutoff |

---

## 6. Nanson Normal Update

### 6.1 Motivation

In finite-strain MPM the particles deform continuously. The reference normal $\mathbf{n}_0$
stored at bond creation is defined in the *undeformed* frame. To evaluate the traction on
the *current* configuration the normal must be transported to the deformed frame. The
classical result from continuum mechanics (Nanson's formula) states that a material area
element with reference normal $\mathbf{n}_0$ and area $dA_0$ maps to a deformed area
element with normal $\mathbf{n}$ and area $dA$ via the cofactor of the deformation
gradient:

$$
\mathbf{n} \, dA = J \mathbf{F}^{-\mathrm{T}} \mathbf{n}_0 \, dA_0
= \mathrm{cof}(\mathbf{F}) \, \mathbf{n}_0 \, dA_0,
$$

where $J = \det(\mathbf{F})$ and $\mathrm{cof}(\mathbf{F}) = J \mathbf{F}^{-\mathrm{T}}$.

### 6.2 Mass-Weighted Average Cofactor

A cohesive bond connects two particles $i$ and $j$, each with its own deformation
gradient $\mathbf{F}_i$ and $\mathbf{F}_j$. Because the bond spans a material interface,
neither $\mathbf{F}_i$ nor $\mathbf{F}_j$ alone is representative of the interface. To
preserve Newton's Third Law across the interface while accounting for both particles'
deformation states, CoupMPM uses the **mass-weighted average cofactor**:

$$
\boxed{
\hat{\mathbf{n}} = \frac{\tilde{\mathbf{n}}}{\|\tilde{\mathbf{n}}\|},
\qquad
\tilde{\mathbf{n}} =
\left[\frac{m_i \, \mathrm{cof}(\mathbf{F}_i) + m_j \, \mathrm{cof}(\mathbf{F}_j)}{m_i + m_j}\right]
\mathbf{n}_0,
}
$$

where $m_i$ and $m_j$ are the masses of particles $i$ and $j$. The result is normalised
to a unit vector. If either Jacobian is degenerate ($|J_i|$ or $|J_j| < 10^{-20}$), the
algorithm falls back to the undeformed reference normal $\mathbf{n}_0$.

In component form, the $d$-th component of $\tilde{\mathbf{n}}$ is

$$
\tilde{n}_d
= \sum_{e=1}^{3}
  \frac{m_i \, [\mathrm{cof}(\mathbf{F}_i)]_{de} + m_j \, [\mathrm{cof}(\mathbf{F}_j)]_{de}}
       {m_i + m_j}
  \, n_0^e.
$$

In `compute_forces()`:
```cpp
double mi = atom_ptr->mass[atom_ptr->type[li]];
double mj = atom_ptr->mass[atom_ptr->type[lj]];
double wtot = mi + mj;

double nm = 0;
for (int d = 0; d < 3; d++) {
    normal[d] = 0;
    for (int e = 0; e < 3; e++)
        normal[d] += ((mi * cof_i[3*d + e] + mj * cof_j[3*d + e]) / wtot)
                     * bond.n0[e];
    nm += normal[d] * normal[d];
}
nm = std::sqrt(nm);
if (nm > 1e-20) {
    for (int d = 0; d < 3; d++) normal[d] /= nm;
}
```

### 6.3 Cofactor Computation

The cofactor matrix is the transpose of the matrix of minors. For a $3 \times 3$ matrix
$\mathbf{F}$ with entries $F_{ij}$ (stored in row-major order), the nine independent
cofactors are:

$$
\begin{aligned}
[\mathrm{cof}(\mathbf{F})]_{00} &= F_{11}F_{22} - F_{12}F_{21}, \\
[\mathrm{cof}(\mathbf{F})]_{01} &= F_{12}F_{20} - F_{10}F_{22}, \\
[\mathrm{cof}(\mathbf{F})]_{02} &= F_{10}F_{21} - F_{11}F_{20}, \\
[\mathrm{cof}(\mathbf{F})]_{10} &= F_{02}F_{21} - F_{01}F_{22}, \\
[\mathrm{cof}(\mathbf{F})]_{11} &= F_{00}F_{22} - F_{02}F_{20}, \\
[\mathrm{cof}(\mathbf{F})]_{12} &= F_{01}F_{20} - F_{00}F_{21}, \\
[\mathrm{cof}(\mathbf{F})]_{20} &= F_{01}F_{12} - F_{02}F_{11}, \\
[\mathrm{cof}(\mathbf{F})]_{21} &= F_{02}F_{10} - F_{00}F_{12}, \\
[\mathrm{cof}(\mathbf{F})]_{22} &= F_{00}F_{11} - F_{01}F_{10}.
\end{aligned}
$$

Note that $\mathrm{cof}(\mathbf{F}) = J \mathbf{F}^{-\mathrm{T}}$ when $J \neq 0$, which
is the standard Nanson relation, but the cofactor is well defined even for singular
$\mathbf{F}$.

---

## 7. Summary of the Per-Timestep DPBCZ Algorithm

The following steps are performed within the standard CoupMPM MPM timestep.
Steps specific to the cohesive zone module are marked **[CZ]**.

1. **[CZ] `detect_new_bonds`** *(every `bond_check_interval` steps)* — scan neighbor
   list for surface-particle pairs within $d_{\mathrm{form}}$; create bonds and record
   their reference states.

2. **[CZ] `compute_forces`** — for each active bond:
   - Update the deformed normal $\hat{\mathbf{n}}$ via the mass-weighted average
     cofactor (Section 6).
   - Decompose $\boldsymbol{\delta}$ into $\delta_n$ and $\delta_t$.
   - Evaluate $T_n$ and $T_t$ from the selected traction-separation law (Section 4
     or 5).
   - Apply $(1 - D)$ damage scaling.
   - Accumulate $\pm A_{\mathrm{ref}} (T_n \hat{\mathbf{n}} + T_t \hat{\mathbf{t}})$
     into `f[li]` and `f[lj]`.

3. **P2G** — cohesive forces are transferred to the grid as external forces:
   $\mathbf{f}_i^{\mathrm{ext}} \mathrel{+}= w_{ip} \mathbf{f}_p^{\mathrm{coh}}$.

4. **Grid solve and G2P** — standard MPM integration (see `theory_apic_transfer.md`).

5. **[CZ] `update_damage_and_break`** — advance $D$ for each bond; deactivate bonds
   that have failed.

---

## 8. Comparison with Traditional FEM Cohesive Zones

| Feature | FEM cohesive elements | DPBCZ (this work) |
|---|---|---|
| Bond location defined at | Pre-processing (mesh insertion) | Runtime (proximity detection) |
| Reference configuration | Global, fixed at $t = 0$ | Per-bond, fixed at formation time |
| Topology changes | Requires remeshing | Automatic |
| Cell sorting | Not supported | Native |
| Differential adhesion | Requires type-specific mesh zones | Single type-pair parameter table |
| Fracture path | Constrained to mesh interfaces | Arbitrary (follows particle contacts) |
| Nanson transport | Global $\mathbf{F}^{-\mathrm{T}}$ | Per-bond mass-weighted average |

---

## References

- Bell, G. I. (1978). Models for the specific adhesion of cells to cells. *Science*,
  200(4342), 618–627. https://doi.org/10.1126/science.347575
- Graner, F., & Glazier, J. A. (1992). Simulation of biological cell sorting using a
  two-dimensional extended Potts model. *Physical Review Letters*, 69(13), 2013–2016.
  https://doi.org/10.1103/PhysRevLett.69.2013
- Needleman, A. (1987). A continuum model for void nucleation by inclusion debonding.
  *Journal of Applied Mechanics*, 54(3), 525–531. https://doi.org/10.1115/1.3173064
- Steinberg, M. S. (1963). Reconstruction of tissues by dissociated cells. *Science*,
  141(3579), 401–408. https://doi.org/10.1126/science.141.3579.401
- Xu, X.-P., & Needleman, A. (1994). Numerical simulations of fast crack growth in
  brittle solids. *Journal of the Mechanics and Physics of Solids*, 42(9), 1397–1434.
  https://doi.org/10.1016/0022-5096(94)90003-5
- Bardenhagen, S. G., Brackbill, J. U., & Sulsky, D. (2000). The material-point method
  for granular materials. *Computer Methods in Applied Mechanics and Engineering*,
  187(3–4), 529–541. https://doi.org/10.1016/S0045-7825(99)00338-2
- Bonet, J., & Wood, R. D. (2008). *Nonlinear Continuum Mechanics for Finite Element
  Analysis* (2nd ed.). Cambridge University Press.
  https://doi.org/10.1017/CBO9780511755446
