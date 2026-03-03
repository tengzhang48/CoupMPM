# Theory: Bardenhagen Multi-Body Contact in MPM

## 1. Overview

In multi-body Material Point Method (MPM) simulations, two or more deformable or rigid
bodies are simultaneously discretised by material points that share a single background
Eulerian grid. Because all bodies map their mass and momentum onto the same set of grid
nodes during the Particle-to-Grid (P2G) step, any node covered by more than one body
accumulates a single, mixed momentum field. Without a contact treatment this merged
nodal velocity would allow the bodies to freely interpenetrate.

CoupMPM implements the **Bardenhagen multi-velocity contact algorithm** of Bardenhagen,
Brackbill & Sulsky (2000). The key idea is to decompose the mixed nodal state into
per-body contributions, detect nodes where bodies are approaching each other, and apply
contact impulses that enforce a non-penetration constraint while respecting Coulomb
friction. Each body is identified by an integer tag (`tagint body_id`) stored together
with its nodal mass, momentum, and velocity in the `NodeBodyData` structure of the
`MPMGrid` class. Up to `MAX_BODIES_PER_NODE = 4` bodies may coexist at any single
grid node.

The algorithm is activated by setting `grid.contact_bardenhagen = true` during grid
allocation. It is implemented in the `ContactBardenhagen` class in
`coupmpm_contact.h` and executes in two phases:

1. **`pre_p2g`** — zeroes all per-body data so that the subsequent P2G step accumulates
   fresh body-specific mass and momentum.
2. **`post_grid_solve`** — after the standard grid momentum equations have been
   integrated, applies contact impulses to the per-body velocities and reconstructs the
   total nodal velocity.

---

## 2. Per-Body Data on the Grid

For every grid node $i$ and every body $b$ present at that node, the `NodeBodyData`
struct holds:

| Field | Symbol | Description |
|---|---|---|
| `body_id` | — | Integer tag identifying the body |
| `mass` | $m_i^{(b)}$ | Mass of body $b$ at node $i$ |
| `momentum[3]` | $(m\mathbf{v})_i^{(b)}$ | Momentum of body $b$ at node $i$ |
| `velocity[3]` | $\mathbf{v}_i^{(b)}$ | Per-body velocity (derived post-P2G) |
| `velocity_new[3]` | $\mathbf{v}_i^{(b),\mathrm{new}}$ | Per-body velocity after contact correction |

The array `grid.body_data` is laid out as a flat array of size
`ntotal * MAX_BODIES_PER_NODE`. The per-body data for node $i$ starts at
`base = i * MAX_BODIES_PER_NODE`. The number of distinct bodies at node $i$ is stored
in `grid.num_bodies[i]`.

---

## 3. Initialization: Pre-P2G Reset

Before each P2G step, `pre_p2g` zeros all per-body data and resets the body counter:

$$
m_i^{(b)} = 0, \quad (m\mathbf{v})_i^{(b)} = \mathbf{0},
\quad \text{for all } i, b.
$$

During the subsequent P2G loop, each particle $p$ belonging to body $b_p$ accumulates
its weighted mass and momentum into the corresponding `NodeBodyData` entry located via
`grid.find_or_add_body(n, body_id_p)`.

---

## 4. Center-of-Mass Grid Velocity

The standard MPM grid solve integrates the total nodal momentum and produces the
**center-of-mass (momentum-weighted) velocity** at each node:

$$
\boxed{
\mathbf{v}_i^{\mathrm{cm}}
= \frac{(m\mathbf{v})_i}{m_i}
= \frac{\displaystyle\sum_{b=1}^{N_b} (m\mathbf{v})_i^{(b)}}{\displaystyle\sum_{b=1}^{N_b} m_i^{(b)}},
}
$$

where $N_b$ is the number of bodies at node $i$, and the total nodal mass is
$m_i = \sum_b m_i^{(b)}$. This velocity is stored in
`grid.velocity_new_{x,y,z}[n]` and represents the momentum-conserving average motion
of all bodies at the node. In the code it is read directly after `grid_solve()`:

```cpp
const double vcm[3] = {
    grid.velocity_new_x[n],
    grid.velocity_new_y[n],
    grid.velocity_new_z[n]
};
```

---

## 5. Per-Body Velocity

From the per-body momentum accumulated during P2G, the individual body velocity at
node $i$ is recovered by mass normalisation:

$$
\mathbf{v}_i^{(b)} = \frac{(m\mathbf{v})_i^{(b)}}{m_i^{(b)}},
\qquad m_i^{(b)} > 0.
$$

In the code:

```cpp
for (int b = 0; b < nb; b++) {
    NodeBodyData& bd = grid.body_data[base + b];
    if (bd.mass > MASS_TOL)
        for (int d = 0; d < 3; d++)
            bd.velocity[d] = bd.momentum[d] / bd.mass;
}
```

Contact detection and impulse computation then operate on these per-body velocities for
every pair of bodies $(a, b)$ co-located at the same node.

---

## 6. Contact Detection

### 6.1 Relative Velocity and Contact Normal

For each pair of bodies $(a, b)$ at node $i$, the relative velocity is

$$
\Delta \mathbf{v}^{ab} = \mathbf{v}_i^{(a)} - \mathbf{v}_i^{(b)}.
$$

The contact normal $\hat{\mathbf{n}}^{ab}$ is constructed from the difference of the
per-body velocity directions — a proxy for the spatial separation of the two bodies'
mass distributions at this node:

$$
\mathbf{n}^{ab} = \frac{(m\mathbf{v})_i^{(a)}}{m_i^{(a)}} - \frac{(m\mathbf{v})_i^{(b)}}{m_i^{(b)}}
= \mathbf{v}_i^{(a)} - \mathbf{v}_i^{(b)},
\qquad
\hat{\mathbf{n}}^{ab} = \frac{\mathbf{n}^{ab}}{|\mathbf{n}^{ab}|}.
$$

If this vector is degenerate (norm $< 10^{-15}$), the algorithm falls back to the
relative velocity direction $\Delta \mathbf{v}^{ab}$. If both are degenerate, the
node pair is skipped.

### 6.2 Normal Relative Velocity

The signed normal component of the relative velocity is

$$
v_n^{ab} = \Delta \mathbf{v}^{ab} \cdot \hat{\mathbf{n}}^{ab}.
$$

- $v_n^{ab} < 0$: bodies are **approaching** — contact forces are required.
- $v_n^{ab} \geq 0$: bodies are **separating** — no contact (unless adhesion is active).

---

## 7. Contact Impulses

All contact corrections are expressed as **impulses** (force $\times$ timestep), so
that the velocity change for body $a$ is $\Delta \mathbf{v}^{(a)} = \mathbf{J} /
m_i^{(a)}$. The reduced mass for the pair is

$$
m_{\mathrm{red}}^{ab} = \frac{m_i^{(a)} \, m_i^{(b)}}{m_i^{(a)} + m_i^{(b)}}.
$$

### 7.1 Normal (Non-Penetration) Impulse

When the bodies are approaching ($v_n^{ab} < 0$), a repulsive normal impulse is applied
to drive the relative normal velocity to zero:

$$
\boxed{J_n = -m_{\mathrm{red}}^{ab} \, v_n^{ab}.}
$$

This impulse is the minimum required to prevent interpenetration without imparting a
net rebound; it exactly zeros the approaching component of the relative velocity.

### 7.2 Adhesion (Optional)

When an adhesion energy density $\gamma$ (in energy per unit area) is specified and
the bodies are separating ($v_n^{ab} \geq 0$), a cohesive (attractive) normal impulse
resists separation. The effective nodal area is approximated as

$$
A_{\mathrm{eff}} =
\begin{cases}
\Delta x_{\min}^2 & \text{(3-D)}, \\
\Delta x_{\min}   & \text{(2-D)},
\end{cases}
\qquad \Delta x_{\min} = \min(\Delta x, \Delta y, \Delta z).
$$

The adhesive force is $f_{\mathrm{adh}} = \gamma A_{\mathrm{eff}}$, giving an adhesive
impulse capped so that it cannot reverse the separation:

$$
J_n = -\min\!\left(\gamma A_{\mathrm{eff}} \, \Delta t,\;
                    m_{\mathrm{red}}^{ab} \, v_n^{ab}\right).
$$

### 7.3 Tangential (Frictional) Impulse

The tangential component of the relative velocity is

$$
\Delta \mathbf{v}_t^{ab} = \Delta \mathbf{v}^{ab} - v_n^{ab} \, \hat{\mathbf{n}}^{ab},
\qquad
\hat{\mathbf{t}}^{ab} = \frac{\Delta \mathbf{v}_t^{ab}}{|\Delta \mathbf{v}_t^{ab}|}.
$$

A **Coulomb friction** constraint limits the tangential impulse to a fraction $\mu$ of
the normal impulse magnitude. The applied tangential impulse is the smaller of the
full-stop impulse (that would arrest all sliding) and the Coulomb limit:

$$
\boxed{J_t = -\min\!\left(\mu \, |J_n|,\; m_{\mathrm{red}}^{ab} \, |\Delta \mathbf{v}_t^{ab}|\right),}
$$

where $\mu$ is the Coulomb friction coefficient (`friction_mu`, default 0.3). The
friction impulse acts in the direction $\hat{\mathbf{t}}^{ab}$ (opposing sliding).
If $|\Delta \mathbf{v}_t^{ab}| < 10^{-20}$ or $\mu = 0$, no friction impulse is applied.

---

## 8. Velocity Update

The contact impulse vector at node $i$ for the pair $(a, b)$ is

$$
\mathbf{J}^{ab} = J_n \, \hat{\mathbf{n}}^{ab} + J_t \, \hat{\mathbf{t}}^{ab}.
$$

The per-body velocities are updated by Newton's third law: body $a$ receives impulse
$+\mathbf{J}^{ab}$ and body $b$ receives $-\mathbf{J}^{ab}$:

$$
\mathbf{v}_i^{(a),\mathrm{new}} = \mathbf{v}_i^{(a)} + \frac{\mathbf{J}^{ab}}{m_i^{(a)}},
\qquad
\mathbf{v}_i^{(b),\mathrm{new}} = \mathbf{v}_i^{(b)} - \frac{\mathbf{J}^{ab}}{m_i^{(b)}}.
$$

In the code:

```cpp
for (int d = 0; d < 3; d++) {
    double imp_d = impulse_n * normal[d] + impulse_t * tangent[d];
    ba.velocity_new[d] = ba.velocity[d] + imp_d / ba.mass;
    bb.velocity_new[d] = bb.velocity[d] - imp_d / bb.mass;
}
```

---

## 9. Reconstruction of the Total Nodal Velocity

After all body pairs have been processed, the corrected total nodal velocity is
reconstructed as the mass-weighted average of the updated per-body velocities:

$$
\mathbf{v}_i^{n+1}
= \frac{\displaystyle\sum_{b=1}^{N_b} m_i^{(b)} \, \mathbf{v}_i^{(b),\mathrm{new}}}{m_i}.
$$

For bodies at the node that were not involved in any contact interaction,
$\mathbf{v}_i^{(b),\mathrm{new}} = \mathbf{v}_i^{(b)}$ (the original per-body velocity
is used unmodified). This reconstructed velocity replaces
`grid.velocity_new_{x,y,z}[n]` and is subsequently used for the G2P transfer.

In the code:

```cpp
if (any_contact && grid.mass[n] > MASS_TOL) {
    grid.velocity_new_x[n] = vn_x / grid.mass[n];
    grid.velocity_new_y[n] = vn_y / grid.mass[n];
    grid.velocity_new_z[n] = vn_z / grid.mass[n];
}
```

---

## 10. Summary of the Contact Algorithm per Timestep

The following steps are performed by `ContactBardenhagen` within the standard MPM
timestep. Steps specific to the contact algorithm are marked **[C]**.

1. **[C] `pre_p2g`** — zero all per-body mass, momentum, and velocity arrays.

2. **P2G** — for each particle $p$ (body $b_p$) and each node $i$ in its support:
   - Standard accumulation of total mass, APIC momentum, internal and external forces.
   - **[C]** Accumulate per-body mass:
     $$m_i^{(b_p)} \mathrel{+}= w_{ip} \, m_p.$$
   - **[C]** Accumulate per-body momentum:
     $$(mv_\alpha)_i^{(b_p)} \mathrel{+}= w_{ip} \, m_p
       \left(v_p^\alpha + \sum_\beta C_p^{\alpha\beta}(x_i^\beta - x_p^\beta)\right).$$

3. **Reverse MPI exchange** — sum ghost contributions onto owner nodes (including
   per-body fields via `sync_body_data_mpi`).

4. **Grid solve** — compute center-of-mass velocity:
   $$\mathbf{v}_i^{\mathrm{cm}} = \frac{(m\mathbf{v})_i}{m_i}
     + \frac{\Delta t}{m_i}\!\left(\mathbf{f}_i^{\mathrm{int}} + \mathbf{f}_i^{\mathrm{ext}}\right).$$

5. **[C] `post_grid_solve`** — at every node $i$ with $N_b \geq 2$:
   - Compute per-body velocities: $\mathbf{v}_i^{(b)} = (m\mathbf{v})_i^{(b)} / m_i^{(b)}$.
   - For each pair $(a,b)$: detect approach; compute $J_n$, $J_t$; update
     $\mathbf{v}_i^{(a),\mathrm{new}}$ and $\mathbf{v}_i^{(b),\mathrm{new}}$.
   - Reconstruct total velocity:
     $\mathbf{v}_i^{n+1} = \sum_b m_i^{(b)} \mathbf{v}_i^{(b),\mathrm{new}} / m_i$.

6. **Forward MPI exchange** — broadcast $\mathbf{v}_i^{n+1}$ to ghost nodes.

7. **G2P** — interpolate updated nodal velocities back to particles.

---

## 11. Conservation Properties

**Linear momentum.** The contact impulses satisfy Newton's third law: the impulse
applied to body $a$ is equal and opposite to that applied to body $b$. Summing over
all pairs at all nodes, the total nodal momentum $\sum_b m_i^{(b)} \mathbf{v}_i^{(b)}$
is unchanged by the contact step, so the reconstructed nodal velocity satisfies

$$
m_i \, \mathbf{v}_i^{n+1} = \sum_b m_i^{(b)} \mathbf{v}_i^{(b),\mathrm{new}}
= \sum_b m_i^{(b)} \mathbf{v}_i^{(b)} = (m\mathbf{v})_i.
$$

The contact correction therefore preserves the center-of-mass momentum at every node.

**Non-penetration.** After the contact step the relative normal velocity satisfies

$$
(\mathbf{v}_i^{(a),\mathrm{new}} - \mathbf{v}_i^{(b),\mathrm{new}}) \cdot \hat{\mathbf{n}}^{ab}
= v_n^{ab} + J_n / m_{\mathrm{red}}^{ab} \cdot (1) = v_n^{ab} - v_n^{ab} = 0,
$$

ensuring the approaching component of relative motion is eliminated.

---

## References

- Bardenhagen, S. G., Brackbill, J. U., & Sulsky, D. (2000). The material-point method
  for granular materials. *Computer Methods in Applied Mechanics and Engineering*,
  187(3–4), 529–541. https://doi.org/10.1016/S0045-7825(99)00338-2
- Bardenhagen, S. G., Guilkey, J. E., Roessig, K. M., Brackbill, J. U., Witzel, W. M.,
  & Foster, J. C. (2001). An improved contact algorithm for the material point method
  and application to stress propagation in granular material. *Computer Modeling in
  Engineering & Sciences*, 2(4), 509–522.
- Sulsky, D., Chen, Z., & Schreyer, H. L. (1994). A particle method for
  history-dependent materials. *Computer Methods in Applied Mechanics and Engineering*,
  118(1–2), 179–196. https://doi.org/10.1016/0045-7825(94)90112-0
- Jiang, C., Schroeder, C., Selle, A., Teran, J., & Stomakhin, A. (2015). The affine
  particle-in-cell method. *ACM Transactions on Graphics*, 34(4), 51.
  https://doi.org/10.1145/2766996
