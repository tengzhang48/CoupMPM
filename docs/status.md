# Implementation Status

## Known Limitations

The following issues are known and deferred for future development:

| Issue | Impact | Recommended Fix |
|---|---|---|
| `grad-rho` field not forward-communicated | Ghost nodes carry zero density gradient; particles near subdomain boundaries may be misclassified as interior | Add a `forward_comm` pass for `grad_rho_x/y/z` |
| No reference-configuration cohesive zones | Cohesive adhesion is range-limited to within one grid cell | Implement the Crook–Homel reference-configuration method (Phase 2) |

## Complete

- **`atom_vec_mpm`**: Full MPI pack/unpack for exchange, border, restart, forward, and reverse communication (16 methods). Data file format: `atom-ID mol-ID atom-type vol0 x y z`.
- **`coupmpm_grid`**: SoA layout, variable ghost width, grid solve, B-bar normalization, per-body Bardenhagen node tracking.
- **`coupmpm_kernel`**: Linear, quadratic B-spline, and cubic B-spline kernels with gradients and D_inv; support-range computation.
- **`coupmpm_transfer`**: P2G (APIC, stress divergence, B-bar, per-body momentum, P2GRecord). G2P (velocity, affine matrix **B**_p, velocity gradient **L**, div_v). Anti-P2G migration protocol. MPI ghost exchange.
- **`coupmpm_stress`**: Neo-Hookean with Jacobian clamping. Acoustic wave speed for CFL estimation.
- **`coupmpm_contact`**: Bardenhagen multi-velocity algorithm with geometric contact normal, Coulomb friction, and area-scaled adhesion.
- **`coupmpm_surface`**: Density-gradient surface detection with global `MPI_Allreduce` threshold. Nanson area scaling for adhesion.
- **`coupmpm_adaptivity`**: Jacobian-based particle splitting and nearest-neighbour merging with LAMMPS atom-management integration.
- **`coupmpm_cohesive`**: Dynamic Cohesive Zone module with three traction-separation laws and runtime bond formation/rupture.
- **`coupmpm_io`**: Grid VTK, particle VTK, and PVD time-series output.
- **`fix_coupmpm`** (parent): Keyword parsing, grid/MPI setup, complete Verlet timestep loop (P2G → grid solve → G2P), CFL control. Exposes public state for companion fixes.
- **`fix_coupmpm_contact`**: Companion fix wrapping `MPMContact`; called by parent at pre-P2G and post-grid-solve.
- **`fix_coupmpm_cohesive`**: Companion fix for dynamic cohesive zones; force injection via parent callback; bond detection and damage update in `end_of_step`; `pack/unpack_exchange` for bond migration.
- **`fix_coupmpm_adaptivity`**: Companion fix for particle splitting and merging in `end_of_step`.
- **`fix_coupmpm_output`**: Companion fix for VTK output and surface detection in `end_of_step`.

## Stubs and Future Work

- Mooney-Rivlin constitutive model (stub only)
- Generalized Maxwell viscoelastic model
- AFLIP blended transfer scheme
- Reference-configuration cohesive zones (Crook–Homel)
- `pair_style mpm/penalty` for penalty contact
- Reaction-diffusion chemistry coupling
- Checkpoint/restart support

## Validation Roadmap

### Phase 0 — Single-rank benchmarks
1. Patch test: deformation gradient **F** remains identity under zero applied velocity.
2. 1D elastic wave: wave speed = √((κ + 4μ/3) / ρ₀).
3. Particle adaptivity conservation check: total mass and momentum invariant through split/merge cycles.

### Phase 1 — Multi-rank benchmarks
4. 1D wave on 4 MPI ranks: momentum drift < 10⁻¹⁰ after 1000 steps.
5. Bardenhagen contact across MPI subdomain boundaries.
6. Surface detection consistency: surface flags identical regardless of domain decomposition.

### Phase 2 — Contact benchmarks
7. Hertz contact: contact area *a*³ ∝ applied load *P*.
8. JKR adhesion: pull-off force = (3/2)π γ R.
9. Sliding block on inclined plane: friction coefficient recovery.

### Phase 3 — Biological applications
10. Cell sorting via differential adhesion (Steinberg model).
11. Cytokinetic ring contraction during cell division.
12. Coupled LBM-MPM simulation of a cell in shear flow.
