# Theory: Dynamic Particle Adaptivity — Splitting and Merging

## 1. Overview

The accuracy and efficiency of the Material Point Method (MPM) depend critically on
maintaining a sufficient and well-distributed number of material points per grid cell
throughout a simulation. Large deformation degrades this distribution in two opposing
ways:

- **Tension / extension**: Particles spread apart. Grid cells in the stretched region
  contain too few integration points, causing quadrature error, stress-field
  instability, and the well-known *particle leaking* artefact in which material points
  escape through the domain boundary.
- **Compression**: Particles cluster. Cells accumulate many particles that represent
  the same physical material, wasting computational resources and potentially inducing
  numerical instability through repeated, nearly identical quadrature contributions.

CoupMPM addresses both pathologies with a dynamic particle adaptivity algorithm
implemented in `coupmpm_adaptivity.h`. At configurable intervals the algorithm:

1. **Splits** particles whose volume has expanded beyond a threshold by replacing the
   parent with $2^{n_{\text{dim}}}$ children placed symmetrically around the parent
   position;
2. **Merges** pairs of over-compressed particles from the same body into a single
   merged particle.

Both operations are designed to be *exactly conservative* in mass, volume, and linear
momentum, and *approximately conservative* in angular momentum to second order in the
child placement offset $\delta$. The constitutive state (deformation gradient
$\mathbf{F}$ and Cauchy stress $\boldsymbol{\sigma}$) is transferred without
interpolation error during a split, and via the heavier-particle rule during a merge,
preserving the thermodynamic consistency of the material model.

---

## 2. Adaptivity Criteria

### 2.1 Jacobian-Based Criterion

The local volume change of a material point is measured by the Jacobian of the
deformation gradient:

$$
J_p = \det(\mathbf{F}_p).
$$

For an initially unit-volume reference configuration, $J_p$ equals the ratio of
current volume to reference volume. The following thresholds govern adaptivity:

| Condition | Action |
|-----------|--------|
| $J_p > J_{\text{hi}}$ | Split particle $p$ |
| $J_p < J_{\text{lo}}$ | Mark $p$ as merge candidate |

Default values: $J_{\text{hi}} = 2.0$, $J_{\text{lo}} = 0.3$. These may be tuned
to the expected deformation range of the simulation.

In `find_split_candidates()`:

```cpp
double J = Mat3::det(Fp);
if (J > J_split_hi)
    candidates.push_back(p);
```

### 2.2 Nearest-Neighbour Distance Criterion

In addition to the Jacobian threshold, a particle $p$ triggers splitting if its
nearest same-body neighbour is farther than
$d_{\text{split}} \cdot \Delta x_{\text{min}}$,
where $\Delta x_{\text{min}}$ is the minimum grid spacing. This criterion catches
tension-driven particle separation even when $J$ has not yet exceeded
$J_{\text{hi}}$.

Similarly, two particles $p$ and $q$ from the same body are merge candidates if their
separation falls below $d_{\text{merge}} \cdot \Delta x_{\text{min}}$.

Default values: $d_{\text{split}} = 1.8$, $d_{\text{merge}} = 0.3$.

---

## 3. Particle Splitting

### 3.1 Child Placement

A parent particle $p$ at position $\mathbf{x}_p$ is replaced by
$n_{\text{child}} = 2^{n_{\text{dim}}}$ children
($n_{\text{child}} = 4$ in 2D, $8$ in 3D). Each child $c$ is placed at

$$
\mathbf{x}_c = \mathbf{x}_p + \boldsymbol{\epsilon}_c \, \delta,
$$

where
$\boldsymbol{\epsilon}_c \in \lbrace -1, +1 \rbrace^{n_{\text{dim}}}$
is a sign vector that selects one corner of a hypercube, and $\delta$ is a scalar
offset chosen from the current particle spacing:

$$
\delta = \frac{1}{4} s_p,
\qquad
s_p =
\begin{cases}
\left( V_p^0 \left| J_p \right| \right)^{1/3} & \text{3D}, \\[6pt]
\left( V_p^0 \left| J_p \right| \right)^{1/2} & \text{2D},
\end{cases}
$$

where $V_p^0$ is the reference volume of the parent. Placing children at
$\pm \delta = \pm s_p / 4$ ensures they remain within the parent's support and do
not overlap with neighbours.

The complete set of 3D child positions ($\boldsymbol{\epsilon}_c$ patterns) is:

$$
\boldsymbol{\epsilon}_c \in \left\lbrace
(-1,-1,-1),\; (+1,-1,-1),\; (-1,+1,-1),\; (+1,+1,-1),\;
(-1,-1,+1),\; (+1,-1,+1),\; (-1,+1,+1),\; (+1,+1,+1)
\right\rbrace.
$$

### 3.2 Conservation During Splitting

Let the parent have mass $m_p$, reference volume $V_p^0$, velocity $\mathbf{v}_p$,
deformation gradient $\mathbf{F}_p$, and Cauchy stress $\boldsymbol{\sigma}_p$.
Each child $c$ receives

$$
m_c = \frac{m_p}{n_{\text{child}}},
\quad
V_c^0 = \frac{V_p^0}{n_{\text{child}}},
\quad
\mathbf{v}_c = \mathbf{v}_p,
\quad
\mathbf{F}_c = \mathbf{F}_p,
\quad
\boldsymbol{\sigma}_c = \boldsymbol{\sigma}_p.
$$

**Mass conservation** is exact by construction:

$$
\sum_{c=1}^{n_{\text{child}}} m_c
= n_{\text{child}} \cdot \frac{m_p}{n_{\text{child}}}
= m_p.
\tag{1}
$$

**Reference volume conservation** is exact:

$$
\sum_{c=1}^{n_{\text{child}}} V_c^0
= n_{\text{child}} \cdot \frac{V_p^0}{n_{\text{child}}}
= V_p^0.
\tag{2}
$$

**Linear momentum conservation** is exact. Since all children share the parent
velocity $\mathbf{v}_p$,

$$
\sum_{c=1}^{n_{\text{child}}} m_c \, \mathbf{v}_c
= \left( \sum_{c=1}^{n_{\text{child}}} m_c \right) \mathbf{v}_p
= m_p \, \mathbf{v}_p.
\tag{3}
$$

**Angular momentum conservation** is approximate. The angular momentum contribution
of the parent about the origin is
$\mathbf{L}_p = m_p \, \mathbf{x}_p \times \mathbf{v}_p$.
The angular momentum of the child system is

$$
\mathbf{L}_{\text{children}}
= \sum_{c=1}^{n_{\text{child}}} m_c \, \mathbf{x}_c \times \mathbf{v}_c
= \sum_{c=1}^{n_{\text{child}}} m_c
  \left( \mathbf{x}_p + \boldsymbol{\epsilon}_c \delta \right)
  \times \mathbf{v}_p.
$$

Expanding:

$$
\mathbf{L}_{\text{children}}
= m_p \, \mathbf{x}_p \times \mathbf{v}_p
+ \delta
  \underbrace{
    \left( \sum_{c=1}^{n_{\text{child}}} m_c \, \boldsymbol{\epsilon}_c \right)
  }_{= \, \mathbf{0}}
  \times \mathbf{v}_p.
\tag{4}
$$

The residual sum vanishes because the sign vectors $\boldsymbol{\epsilon}_c$ are
arranged symmetrically: for every child at $+\delta$ in each direction there is a
partner at $-\delta$, so
$\sum_c \boldsymbol{\epsilon}_c = \mathbf{0}$.
Therefore

$$
\mathbf{L}_{\text{children}}
= m_p \, \mathbf{x}_p \times \mathbf{v}_p
= \mathbf{L}_p.
\tag{5}
$$

Angular momentum is thus conserved **exactly** for any finite $\delta$, because the
symmetric placement of the children ensures that the first moment of their position
distribution about $\mathbf{x}_p$ is zero.

### 3.3 Constitutive-State Transfer During Splitting

Splitting does not alter the local deformation state: the deformation gradient
$\mathbf{F}_p$ and the Cauchy stress $\boldsymbol{\sigma}_p$ (together with all
history-variable arrays) are **copied exactly** into every child. This is physically
correct because the children represent the same material at the same deformation state
--- splitting is a purely numerical sub-division of the integration domain, not a
physical event.

Formally, for every child $c$:

$$
\mathbf{F}_c = \mathbf{F}_p,
\qquad
\boldsymbol{\sigma}_c = \boldsymbol{\sigma}_p,
\qquad
\mathbf{C}_c = \mathbf{C}_p,
\qquad
\boldsymbol{s}_c = \boldsymbol{s}_p,
\tag{6}
$$

where $\mathbf{C}_p$ is the APIC affine matrix and $\boldsymbol{s}_p$ denotes any
internal-state variables. No constitutive model evaluation is required at the moment
of splitting.

In `generate_children()`:

```cpp
// Deformation state: exact copy
std::memcpy(c.F_def,    F_def_p, 9 * sizeof(double));
std::memcpy(c.stress_v, stress_p, 6 * sizeof(double));
std::memcpy(c.Bp,       Bp_p,    9 * sizeof(double));
std::memcpy(c.state,    state_p, n_state * sizeof(double));
```

---

## 4. Particle Merging

### 4.1 Merge-Candidate Selection

Merge candidates are identified in two stages. First, particles with
$J_p < J_{\text{lo}}$ are labelled *compressed*. Then, within the set of compressed
particles belonging to the same body, nearest-neighbour pairs closer than
$d_{\text{merge}} \cdot \Delta x_{\text{min}}$ are selected as merge pairs. A greedy
matching algorithm ensures that each particle appears in at most one pair per
adaptivity step.

### 4.2 Merged Particle Properties

Given two particles $i$ and $j$ with masses $m_i$, $m_j$, positions
$\mathbf{x}_i$, $\mathbf{x}_j$, and velocities $\mathbf{v}_i$, $\mathbf{v}_j$,
the merged particle $m$ receives the following properties.

**Mass** (conserved exactly):

$$
m_m = m_i + m_j.
\tag{7}
$$

**Reference volume** (conserved exactly):

$$
V_m^0 = V_i^0 + V_j^0.
\tag{8}
$$

**Position** (mass-weighted centroid):

$$
\mathbf{x}_m
= \frac{m_i \, \mathbf{x}_i + m_j \, \mathbf{x}_j}{m_i + m_j}.
\tag{9}
$$

**Velocity** (mass-weighted average, ensures linear momentum conservation):

$$
\mathbf{v}_m
= \frac{m_i \, \mathbf{v}_i + m_j \, \mathbf{v}_j}{m_i + m_j}.
\tag{10}
$$

**APIC affine matrix** (mass-weighted average):

$$
\mathbf{C}_m
= \frac{m_i \, \mathbf{C}_i + m_j \, \mathbf{C}_j}{m_i + m_j}.
\tag{11}
$$

### 4.3 Deformation Gradient and Stress During Merging

Unlike position and velocity, the deformation gradient $\mathbf{F}$ is a multiplicative
(non-additive) tensor quantity. A naive linear average
$\alpha \mathbf{F}_i + (1 - \alpha) \mathbf{F}_j$
does not in general belong to $\mathrm{GL}^{+}(3)$ (the group of
orientation-preserving deformation gradients), and may violate thermodynamic
consistency by combining two different constitutive states.

CoupMPM uses the **heavier-particle rule**: the deformation gradient and stress of the
heavier particle are assigned to the merged particle without modification:

$$
\mathbf{F}_m =
\begin{cases}
\mathbf{F}_i & \text{if } m_i \ge m_j, \\[4pt]
\mathbf{F}_j & \text{otherwise,}
\end{cases}
\qquad
\boldsymbol{\sigma}_m =
\begin{cases}
\boldsymbol{\sigma}_i & \text{if } m_i \ge m_j, \\[4pt]
\boldsymbol{\sigma}_j & \text{otherwise.}
\end{cases}
\tag{12}
$$

Similarly, internal-state variables (plastic strain, damage, etc.) are taken from the
heavier particle.

**Rationale.** Merging events occur in compression zones where $J_p \ll 1$ and the
two particles necessarily carry similar $\mathbf{F}$ values (both are compressed by the
same macroscopic deformation field). Taking the heavier particle's $\mathbf{F}$
introduces an error that is second order in the difference
$\lVert \mathbf{F}_i - \mathbf{F}_j \rVert$,
which is small relative to the magnitude of $\mathbf{F}$. An alternative mass-weighted
polar decomposition would be more accurate but is computationally expensive and not
justified in the compression regime where merging occurs.

In `merge_particles()`:

```cpp
const double* F_keep  = (mass_i >= mass_j) ? F_i : F_j;
const double* s_keep  = (mass_i >= mass_j) ? stress_i : stress_j;
std::memcpy(merged.F_def,    F_keep, 9 * sizeof(double));
std::memcpy(merged.stress_v, s_keep, 6 * sizeof(double));
```

### 4.4 Conservation During Merging

**Mass** (Eq. 7): exact.

**Reference volume** (Eq. 8): exact.

**Linear momentum**: from Eq. (10),

$$
m_m \, \mathbf{v}_m
= (m_i + m_j) \cdot \frac{m_i \mathbf{v}_i + m_j \mathbf{v}_j}{m_i + m_j}
= m_i \, \mathbf{v}_i + m_j \, \mathbf{v}_j.
\tag{13}
$$

Linear momentum is therefore conserved exactly.

**Angular momentum**: the merged angular momentum about the origin is

$$
\mathbf{L}_m = m_m \, \mathbf{x}_m \times \mathbf{v}_m.
\tag{14}
$$

Substituting Eqs. (9) and (10):

$$
\mathbf{L}_m
= \frac{m_i \mathbf{x}_i + m_j \mathbf{x}_j}{m_i + m_j}
  \times
  \left( m_i \mathbf{v}_i + m_j \mathbf{v}_j \right).
\tag{15}
$$

Expanding and grouping terms:

$$
\mathbf{L}_m
= m_i \, \mathbf{x}_i \times \mathbf{v}_i
+ m_j \, \mathbf{x}_j \times \mathbf{v}_j
- \frac{m_i m_j}{m_i + m_j}
  \left( \mathbf{x}_i - \mathbf{x}_j \right) \times
  \left( \mathbf{v}_i - \mathbf{v}_j \right).
\tag{16}
$$

The residual (last term) vanishes exactly when either the two particles are co-located
($\mathbf{x}_i = \mathbf{x}_j$)
or co-moving
($\mathbf{v}_i = \mathbf{v}_j$).
For particles in the same compression zone --- where
$\left| \mathbf{x}_i - \mathbf{x}_j \right| \ll \Delta x$
and
$\left| \mathbf{v}_i - \mathbf{v}_j \right| \ll \lVert \mathbf{v} \rVert$
--- the angular momentum error is of order
$\mathcal{O}\!\left( \Delta x \, \lVert \mathbf{v} \rVert \, m_i m_j / m_m \right)$,
which is small compared to the total angular momentum.

---

## 5. Summary of Conservation Properties

| Quantity | Split | Merge |
|---|---|---|
| Total mass $\sum m_p$ | Exact (Eq. 1) | Exact (Eq. 7) |
| Total ref. volume $\sum V_p^0$ | Exact (Eq. 2) | Exact (Eq. 8) |
| Linear momentum $\sum m_p \mathbf{v}_p$ | Exact (Eq. 3) | Exact (Eq. 13) |
| Angular momentum | Exact (Eq. 5) | $\mathcal{O}(\Delta x \lVert \mathbf{v} \rVert)$ error |
| Deformation gradient $\mathbf{F}_p$ | Exact copy (Eq. 6) | Heavier particle (Eq. 12) |
| Cauchy stress $\boldsymbol{\sigma}_p$ | Exact copy (Eq. 6) | Heavier particle (Eq. 12) |
| APIC affine matrix $\mathbf{C}_p$ | Exact copy (Eq. 6) | Mass-weighted (Eq. 11) |
| Internal state $\boldsymbol{s}_p$ | Exact copy (Eq. 6) | Heavier particle (Eq. 12) |

---

## 6. Integration with LAMMPS

Both operations interact with the LAMMPS atom-management layer:

- **Splitting** creates $n_{\text{child}} - 1$ new atoms by growing the per-atom
  arrays and populating all fields of the `ChildParticle` struct. The parent is
  replaced in-place by the first child. A neighbour-list rebuild is flagged
  immediately after.
- **Merging** removes particle $j$ by overwriting it with the last particle in the
  local array (`atom->avec->copy()`) and decrementing `nlocal`. The merged properties
  are written into the slot of particle $i$. A neighbour-list rebuild is likewise
  flagged.

The adaptivity check runs every `check_interval` timesteps (default 20) to amortise
the cost of the candidate-search and atom-management operations.

---

## 7. Summary Algorithm

The complete per-interval adaptivity procedure is as follows.

**Split pass:**

1. For each particle $p$, compute $J_p = \det(\mathbf{F}_p)$.
2. If $J_p > J_{\text{hi}}$, add $p$ to the split list.
3. For each $p$ in the split list, call `generate_children()`:
   - Compute child offset
     $\delta = \frac{1}{4}\left(V_p^0 \left|J_p\right|\right)^{1/n_{\text{dim}}}$.
   - Generate $n_{\text{child}}$ children with positions
     $\mathbf{x}_c = \mathbf{x}_p + \boldsymbol{\epsilon}_c \delta$, and with
     $m_c = m_p / n_{\text{child}}$,
     $V_c^0 = V_p^0 / n_{\text{child}}$,
     $\mathbf{v}_c = \mathbf{v}_p$,
     $\mathbf{F}_c = \mathbf{F}_p$,
     $\boldsymbol{\sigma}_c = \boldsymbol{\sigma}_p$.
4. Insert children into the LAMMPS atom arrays; flag neighbour rebuild.

**Merge pass:**

1. For each particle $p$, compute $J_p = \det(\mathbf{F}_p)$.
2. Collect the subset $\mathcal{C}$ of particles with
   $J_p < J_{\text{lo}}$.
3. Within $\mathcal{C}$, find nearest same-body pairs $(i,j)$ closer than
   $d_{\text{merge}} \cdot \Delta x_{\text{min}}$ via greedy nearest-neighbour matching.
4. For each pair $(i,j)$, call `merge_particles()`:
   - $m_m = m_i + m_j$, $V_m^0 = V_i^0 + V_j^0$.
   - $\mathbf{x}_m = (m_i \mathbf{x}_i + m_j \mathbf{x}_j) / m_m$,
     $\mathbf{v}_m = (m_i \mathbf{v}_i + m_j \mathbf{v}_j) / m_m$,
     $\mathbf{C}_m = (m_i \mathbf{C}_i + m_j \mathbf{C}_j) / m_m$.
   - $\mathbf{F}_m = \mathbf{F}_k$,
     $\boldsymbol{\sigma}_m = \boldsymbol{\sigma}_k$,
     $\boldsymbol{s}_m = \boldsymbol{s}_k$,
     where $k = \arg\max(m_i, m_j)$.
5. Write merged properties into the slot of particle $i$; remove particle $j$ from
   the atom arrays; flag neighbour rebuild.

---

## References

- Sulsky, D., Chen, Z., & Schreyer, H. L. (1994). A particle method for
  history-dependent materials. *Computer Methods in Applied Mechanics and
  Engineering*, 118(1-2), 179-196.
  https://doi.org/10.1016/0045-7825(94)90112-0
- Jiang, C., Schroeder, C., Selle, A., Teran, J., & Stomakhin, A. (2015). The affine
  particle-in-cell method. *ACM Transactions on Graphics*, 34(4), 51.
  https://doi.org/10.1145/2766996
- Jiang, C., Schroeder, C., & Teran, J. (2017). An angular momentum conserving
  affine-particle-in-cell method. *Journal of Computational Physics*, 338, 137-164.
  https://doi.org/10.1016/j.jcp.2017.02.050
- Wobbes, E., Moller, M., Galavi, V., & Vuik, C. (2019). Conservative Taylor least
  squares reconstruction with application to material point methods. *International
  Journal for Numerical Methods in Engineering*, 117(3), 271-290.
  https://doi.org/10.1002/nme.5956
- Liang, Y., Zhang, X., & Liu, Y. (2019). An adaptive particle refinement and coarsening
  scheme for the material point method. *International Journal for Numerical Methods in
  Engineering*, 120(8), 979-1004.
  https://doi.org/10.1002/nme.6167
