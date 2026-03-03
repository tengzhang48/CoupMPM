# Architecture

## Source Layout

```
CoupMPM/
├── fix_coupmpm.h/.cpp              Parent fix: grid/MPI setup, P2G→grid solve→G2P timestep loop
├── fix_coupmpm_contact.h/.cpp      Companion fix: multi-body contact (Bardenhagen / penalty)
├── fix_coupmpm_cohesive.h/.cpp     Companion fix: dynamic cohesive zones and bond management
├── fix_coupmpm_adaptivity.h/.cpp   Companion fix: particle splitting and merging
├── fix_coupmpm_output.h/.cpp       Companion fix: VTK output and surface detection
├── atom_vec_mpm.h/.cpp             Custom atom_style mpm: MPI pack/unpack for all MPM fields
├── coupmpm_grid.h                  Background grid: SoA layout, ghost-inclusive indexing
├── coupmpm_kernel.h                Shape functions: linear, quadratic B-spline, cubic B-spline
├── coupmpm_transfer.h              P2G, G2P (APIC + B-bar), anti-P2G, MPI ghost exchange
├── coupmpm_stress.h                Constitutive models: Neo-Hookean, Mooney-Rivlin
├── coupmpm_contact.h               Multi-body contact algorithms: Bardenhagen, penalty
├── coupmpm_surface.h               Surface detection via density gradient with MPI reduction
├── coupmpm_adaptivity.h            Particle splitting and merging for resolution control
├── coupmpm_cohesive.h              Dynamic Cohesive Zone module for adhesion/cell sorting
└── coupmpm_io.h                    VTK grid/particle output with PVD time-series index
```

## Companion Fix Architecture

`fix coupmpm` is the **parent** fix and manages the core MPM timestep loop (P2G → grid solve → G2P). Four optional **companion fixes** extend its behaviour; each locates the parent fix in its `init()` method and registers a pointer with it:

| Companion fix | Style string | Functionality |
|---|---|---|
| `FixCoupMPMContact` | `coupmpm/contact` | Bardenhagen / penalty contact on the grid |
| `FixCoupMPMCohesive` | `coupmpm/cohesive` | Dynamic cohesive zone bonds |
| `FixCoupMPMAdaptivity` | `coupmpm/adaptivity` | Particle splitting and merging |
| `FixCoupMPMOutput` | `coupmpm/output` | VTK file output and surface detection |

Companion fixes are **entirely optional**. Any subset may be used and the parent fix works standalone. Each companion:

1. Finds the parent in `init()` via a loop over `modify->fix[]` and stores a back-pointer.
2. Registers itself with the parent (`parent->fix_contact = this`, etc.).
3. The parent calls contact callbacks (`pre_p2g`, `post_grid_solve`) directly inside `initial_integrate`.
4. Cohesive force injection (`compute_forces_before_p2g`) is similarly driven by the parent before P2G.
5. `end_of_step` callbacks (surface detection, VTK output, adaptivity, bond updates) are handled by LAMMPS calling each companion's own `end_of_step` method.

## Key Physics Notes

1. **B-bar two-pass**: `raw_div_v` is mass-weighted during P2G. It must be normalized by the nodal mass **after** `reverse_comm`, not before.

2. **APIC affine matrix**: `Bp` stores **C**_p. The APIC velocity contribution at a node is `v_p + C_p · (x_node − x_p)`. `D_inv` is applied during G2P, not during P2G.

3. **Ghost exchange completeness**: The reverse communication includes all P2G accumulation fields: mass, momentum, `force_int`, `force_ext`, `raw_div_v`, and — when Bardenhagen contact is active — the per-body `NodeBodyData` (mass, momentum, and centre-of-mass per body per node).

4. **Contact normal must be geometric**: Using Δv / |Δv| as the contact normal collapses the normal/tangential decomposition. The momentum-direction difference between bodies provides a kinematics-independent geometric normal.

5. **Anti-P2G timing**: LAMMPS Verlet order is `initial_integrate` → forces → `final_integrate` → `comm::exchange`. Executing anti-P2G in `final_integrate` is correct because it runs before particle migration.

6. **Adaptive-particle mass**: When adaptivity is enabled, particle mass is computed from `vol0 × rho0` rather than the per-type LAMMPS mass table to ensure conservation through split/merge cycles.

7. **Surface detection threshold**: `rho_max` must be globally reduced via `MPI_Allreduce(MPI_MAX)` before computing the surface threshold; otherwise each rank applies a different cut-off.

## Design Correspondence with CoupLB

CoupMPM shares architectural patterns with the companion CoupLB Lattice-Boltzmann package:

| CoupLB | CoupMPM | Notes |
|---|---|---|
| Grid SoA with `f[Q * ntotal]` | `MPMGrid` SoA with mass / momentum / force arrays | Same ghost-inclusive linear indexing |
| `Streaming::exchange_dim` | `MPMGhostExchange::reverse_dim` / `forward_dim` | Same six-face neighbour topology |
| `IBM::spread` / `interpolate` | `p2g` / `g2p` (B-spline kernel loops) | Structurally identical loop patterns |
| `IO::write_vtk` (MPI `Gatherv`) | `MPMIO::write_grid_vtk` | Nearly identical gather-and-write logic |
| `exchange_forces` (reverse comm) | `reverse_comm` | Same ghost-to-owner accumulation protocol |
