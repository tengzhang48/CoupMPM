# CoupMPM — Material Point Method Package for LAMMPS

## Overview

CoupMPM is a highly scalable Eulerian-Lagrangian Material Point Method (MPM) solver implemented as a native LAMMPS package. It is specifically designed for extreme-deformation, multi-physics simulations including soft-matter mechanics, bio-inspired materials, and cell-aggregate dynamics. The package couples a background Eulerian grid with Lagrangian material points, enabling robust handling of large strains, topological changes, and multi-body contact without mesh distortion.

The solver integrates directly with the LAMMPS parallelism infrastructure. All grid operations and particle-to-grid (P2G) / grid-to-particle (G2P) transfers are fully MPI-parallelized through a ghost-exchange protocol, making CoupMPM suitable for high-performance computing (HPC) deployments at arbitrary processor counts.

### Source Layout

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

### Companion Fix Architecture

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

---

## Key Features

### Affine Particle-In-Cell (APIC) Transfers
P2G and G2P transfers use the APIC formulation (Jiang et al. 2015, 2017), in which each particle carries an affine velocity matrix **C**_p in addition to its translational velocity. This eliminates the angular-momentum dissipation inherent in PIC while retaining numerical stability superior to FLIP, and avoids the cell-crossing instability of pure FLIP.

### B-Bar Volumetric Anti-Locking
Volumetric locking in nearly incompressible materials is suppressed through the B-bar projection method. The smoothed volumetric strain rate is accumulated on the grid during P2G and projected back during G2P, ensuring that the deviatoric and volumetric parts of the deformation gradient are treated consistently without spurious pressure oscillations.

### Bardenhagen Multi-Body Contact
Inter-body contact is resolved by the Bardenhagen–Kober algorithm (Bardenhagen et al. 2000). At nodes where mass from multiple distinct bodies overlaps, per-body velocities are decomposed into normal and tangential components relative to a geometrically consistent contact normal. Impenetrability is enforced via a momentum-exchange impulse, and Coulomb friction and area-scaled adhesion (J m⁻²) are applied in the tangential direction.

### Dynamic Cohesive Zone Module
A novel Dynamic Cohesive Zone (DCZ) module enables runtime formation and rupture of adhesive bonds between material-point pairs. Three traction-separation laws are available: the Needleman-Xu exponential law, a linear-elastic softening law, and a receptor-ligand kinetic law suited to cell-membrane adhesion modelling. The DCZ module is designed for cell-sorting and dynamic-adhesion applications in which bonds form and break stochastically during the simulation.

### MPI Parallelism and HPC Readiness
- Ghost-node exchange (forward and reverse) covers all P2G accumulation fields: mass, momentum, internal force, external force, and volumetric strain rate.
- Global `MPI_Allreduce` operations enforce consistent surface-detection thresholds and CFL timestep constraints across all ranks.
- Particle migration uses LAMMPS's native `comm->exchange` mechanism, with an anti-P2G protocol that removes departing particles from the grid before they leave the subdomain.
- The package builds with both the traditional LAMMPS `make` system and CMake.

---

## Build Instructions

### Traditional Make (in-tree)
```bash
cp -r CoupMPM/ /path/to/lammps/src/COUPMPM/
cd /path/to/lammps/src
make yes-coupmpm
make -j8 mpi
```

### CMake
```bash
cd /path/to/lammps/build
cmake -D PKG_COUPMPM=on ../cmake
make -j8
```

---

## Usage and Syntax

### Fix Command Syntax

The package is now split into a parent fix and four optional companion fixes.
The parent fix **must** be defined before any companion fix.

#### Parent fix (`fix coupmpm`)

```
fix ID group-ID coupmpm              \
    grid <dx> <dy> <dz>              \
    kernel <linear|bspline2|bspline3> \
    bbar <yes|no>                    \
    constitutive <neohookean mu V kappa V \
                 |mooneyrivlin C1 V C2 V kappa V> \
    dt_auto <yes|no>                 \
    energy_check <yes|no>            \
    cfl <val>                        \
    rho0 <val>
```

#### Companion fix: contact (`fix coupmpm/contact`)

```
fix ID group-ID coupmpm/contact \
    method <none|bardenhagen [mu V] [adhesion V]|penalty>
```

#### Companion fix: cohesive zones (`fix coupmpm/cohesive`)

```
fix ID group-ID coupmpm/cohesive    \
    law <needleman|linear|receptor> \
    sigma <val>                     \
    delta <val>                     \
    delta_max <val>                 \
    form_dist <val>                 \
    interval <N>
```

#### Companion fix: adaptivity (`fix coupmpm/adaptivity`)

```
fix ID group-ID coupmpm/adaptivity \
    J_split <val>                  \
    J_merge <val>                  \
    interval <N>
```

#### Companion fix: output (`fix coupmpm/output`)

```
fix ID group-ID coupmpm/output    \
    vtk_interval <N>              \
    vtk_prefix <prefix>           \
    surface_interval <N>          \
    surface_alpha <val>
```

> **Prerequisites:** The simulation must use `atom_style mpm`. All keywords are optional; defaults are listed in the table below. Each companion fix must appear **after** `fix coupmpm` in the input script.

### Keyword Reference

#### `fix coupmpm` keywords

| Keyword | Arguments | Default | Description |
|---|---|---|---|
| `grid` | `dx dy dz` | `0.1 0.1 0.1` | Background grid cell spacings in the *x*, *y*, and *z* directions (simulation-length units). |
| `kernel` | `linear` \| `bspline2` \| `bspline3` | `linear` | Shape-function kernel. `linear` uses tent functions (support ±1 cell); `bspline2` and `bspline3` use quadratic and cubic B-spline kernels (support ±1.5 and ±2 cells, respectively), which increase smoothness and reduce quadrature noise. |
| `bbar` | `yes` \| `no` | `yes` | Enable the B-bar volumetric anti-locking correction. Recommended for nearly incompressible materials (Poisson's ratio > 0.45). |
| `constitutive` | `neohookean mu V kappa V` \| `mooneyrivlin C1 V C2 V kappa V` | Neo-Hookean with μ = 1 × 10³, κ = 1 × 10⁴ | Hyperelastic constitutive model. `neohookean` requires the shear modulus `mu` and bulk modulus `kappa` (both > 0). `mooneyrivlin` (stub) requires the two deviatoric coefficients and the bulk modulus. |
| `dt_auto` | `yes` \| `no` | `yes` | Automatically adjust the LAMMPS timestep at each step to satisfy the CFL criterion `dt = cfl * dx / c_wave`. When `no`, the timestep specified in the input script is used unchanged. |
| `energy_check` | `yes` \| `no` | `no` | Print a kinetic + strain energy balance to the log at every step. Useful for validating conservation in single-rank benchmarks. |
| `cfl` | `val` | `0.3` | CFL safety factor applied to the acoustic wave-speed estimate when `dt_auto yes` is active. Typical values are 0.2–0.5. |
| `rho0` | `val` | `1000.0` | Reference (initial) mass density in simulation units. Used for CFL wave-speed estimation and adaptive-particle mass initialisation. |

#### `fix coupmpm/contact` keywords

| Keyword | Arguments | Default | Description |
|---|---|---|---|
| `method` | `none` \| `bardenhagen [mu V] [adhesion V]` \| `penalty` | `none` | Multi-body contact algorithm. `bardenhagen` activates the Bardenhagen-Kober impenetrability algorithm; optional sub-keywords set the Coulomb friction coefficient `mu` (dimensionless, default 0.3) and the surface-energy adhesion parameter `adhesion` (J m⁻², default 0). `penalty` activates a spring-dashpot penalty force (parameters set internally). |

#### `fix coupmpm/cohesive` keywords

| Keyword | Arguments | Default | Description |
|---|---|---|---|
| `law` | `needleman` \| `linear` \| `receptor` | `needleman` | Traction-separation law for cohesive bonds. |
| `sigma` | `val` | `100.0` | Peak cohesive traction σ_max (stress units). |
| `delta` | `val` | `1 × 10⁻⁴` | Characteristic separation length δ_0 at peak traction (length units). |
| `delta_max` | `val` | `2 × 10⁻⁴` | Failure separation δ_max beyond which a bond is irreversibly ruptured (length units). |
| `form_dist` | `val` | `5 × 10⁻⁴` | Maximum particle-pair distance at which a new cohesive bond may form (length units). |
| `interval` | `N` | `10` | Check for new bond-formation candidates and rupture events every *N* timesteps. |

#### `fix coupmpm/adaptivity` keywords

| Keyword | Arguments | Default | Description |
|---|---|---|---|
| `J_split` | `val` | `2.0` | Jacobian upper threshold. A particle with *J* > `J_split` is split into child particles to restore resolution in highly stretched regions. |
| `J_merge` | `val` | `0.3` | Jacobian lower threshold. Particles with *J* < `J_merge` are candidates for merging with a nearest neighbour to remove over-compressed particles. |
| `interval` | `N` | `20` | Check and apply splitting/merging every *N* timesteps. |

#### `fix coupmpm/output` keywords

| Keyword | Arguments | Default | Description |
|---|---|---|---|
| `vtk_interval` | `N` | `0` (disabled) | Write VTK files (grid and particles) every *N* timesteps. A PVD time-series index file is maintained automatically. |
| `vtk_prefix` | `prefix` | `coupmpm` | Filename prefix for all VTK output files. |
| `surface_interval` | `N` | `10` | Recompute the particle surface-normal field every *N* timesteps using the density-gradient method. |
| `surface_alpha` | `val` | `0.1` | Fractional threshold for surface classification: nodes with ρ < `surface_alpha` × ρ_max are flagged as surface nodes. |

---

## Minimal Example

The following LAMMPS input script illustrates a three-dimensional compression test of a soft Neo-Hookean block using the Bardenhagen contact algorithm, APIC transfers with B-bar anti-locking, and VTK output:

```lammps
# ---------------------------------------------------------------
# CoupMPM minimal example: uniaxial compression of a soft block
# ---------------------------------------------------------------

units           si
dimension       3
boundary        f f f

atom_style      mpm
atom_modify     map array

read_data       block.data          # atom-ID mol-ID type vol0 x y z

# Parent fix: grid, kernel, constitutive model
fix mpm all coupmpm                 \
    grid 0.05 0.05 0.05             \
    kernel bspline2                 \
    bbar yes                        \
    constitutive neohookean mu 1e4 kappa 1e5 \
    dt_auto yes                     \
    cfl 0.3                         \
    rho0 1200.0

# Contact companion
fix contact all coupmpm/contact     \
    method bardenhagen mu 0.2

# Output companion
fix viz all coupmpm/output          \
    vtk_interval 50                 \
    vtk_prefix compress

# Apply a constant compressive velocity to the top surface group
fix wall_top top_atoms setforce 0.0 0.0 0.0
velocity top_atoms set 0.0 0.0 -0.01

thermo          100
thermo_style    custom step time ke pe etotal

timestep        1e-5
run             5000
```

### Cohesive Zone Example (Cell Sorting)

To activate the Dynamic Cohesive Zone module for a cell-aggregate simulation:

```lammps
fix mpm all coupmpm                         \
    grid 2.0e-6 2.0e-6 2.0e-6              \
    kernel bspline3                         \
    bbar yes                                \
    constitutive neohookean mu 500.0 kappa 2000.0 \
    dt_auto yes cfl 0.25 rho0 1050.0

fix contact all coupmpm/contact             \
    method bardenhagen mu 0.0 adhesion 5e-4

fix cz all coupmpm/cohesive                 \
    law receptor                            \
    sigma 200.0                             \
    delta 5e-7                              \
    delta_max 1.2e-6                        \
    form_dist 3e-6                          \
    interval 5

fix viz all coupmpm/output                  \
    vtk_interval 100 vtk_prefix cells
```

---

## Known Limitations

The following issues are known and deferred for future development:

| Issue | Impact | Recommended Fix |
|---|---|---|
| `grad-rho` field not forward-communicated | Ghost nodes carry zero density gradient; particles near subdomain boundaries may be misclassified as interior | Add a `forward_comm` pass for `grad_rho_x/y/z` |
| No reference-configuration cohesive zones | Cohesive adhesion is range-limited to within one grid cell | Implement the Crook–Homel reference-configuration method (Phase 2) |

---

## Implementation Status

### Complete
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
- **`fix_coupmpm`** (parent): Keyword parsing, grid/MPI setup, complete Verlet timestep loop (P2G → grid solve → G2P), CFL control.  Exposes public state for companion fixes.
- **`fix_coupmpm_contact`**: Companion fix wrapping `MPMContact`; called by parent at pre-P2G and post-grid-solve.
- **`fix_coupmpm_cohesive`**: Companion fix for dynamic cohesive zones; force injection via parent callback; bond detection and damage update in `end_of_step`; `pack/unpack_exchange` for bond migration.
- **`fix_coupmpm_adaptivity`**: Companion fix for particle splitting and merging in `end_of_step`.
- **`fix_coupmpm_output`**: Companion fix for VTK output and surface detection in `end_of_step`.

### Stubs and Future Work
- Mooney-Rivlin constitutive model (stub only)
- Generalized Maxwell viscoelastic model
- AFLIP blended transfer scheme
- Reference-configuration cohesive zones (Crook–Homel)
- `pair_style mpm/penalty` for penalty contact
- Reaction-diffusion chemistry coupling
- Checkpoint/restart support

---

## Key Physics Notes

1. **B-bar two-pass**: `raw_div_v` is mass-weighted during P2G. It must be normalized by the nodal mass **after** `reverse_comm`, not before.

2. **APIC affine matrix**: `Bp` stores **C**_p. The APIC velocity contribution at a node is `v_p + C_p · (x_node − x_p)`. `D_inv` is applied during G2P, not during P2G.

3. **Ghost exchange completeness**: The reverse communication includes all P2G accumulation fields: mass, momentum, `force_int`, `force_ext`, `raw_div_v`, and — when Bardenhagen contact is active — the per-body `NodeBodyData` (mass, momentum, and centre-of-mass per body per node).

4. **Contact normal must be geometric**: Using Δv / |Δv| as the contact normal collapses the normal/tangential decomposition. The momentum-direction difference between bodies provides a kinematics-independent geometric normal.

5. **Anti-P2G timing**: LAMMPS Verlet order is `initial_integrate` → forces → `final_integrate` → `comm::exchange`. Executing anti-P2G in `final_integrate` is correct because it runs before particle migration.

6. **Adaptive-particle mass**: When adaptivity is enabled, particle mass is computed from `vol0 × rho0` rather than the per-type LAMMPS mass table to ensure conservation through split/merge cycles.

7. **Surface detection threshold**: `rho_max` must be globally reduced via `MPI_Allreduce(MPI_MAX)` before computing the surface threshold; otherwise each rank applies a different cut-off.

---

## Design Correspondence with CoupLB

CoupMPM shares architectural patterns with the companion CoupLB Lattice-Boltzmann package:

| CoupLB | CoupMPM | Notes |
|---|---|---|
| Grid SoA with `f[Q * ntotal]` | `MPMGrid` SoA with mass / momentum / force arrays | Same ghost-inclusive linear indexing |
| `Streaming::exchange_dim` | `MPMGhostExchange::reverse_dim` / `forward_dim` | Same six-face neighbour topology |
| `IBM::spread` / `interpolate` | `p2g` / `g2p` (B-spline kernel loops) | Structurally identical loop patterns |
| `IO::write_vtk` (MPI `Gatherv`) | `MPMIO::write_grid_vtk` | Nearly identical gather-and-write logic |
| `exchange_forces` (reverse comm) | `reverse_comm` | Same ghost-to-owner accumulation protocol |

---

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

---

## References

- Jiang, C., Schroeder, C., Selle, A., Teran, J., & Stomakhin, A. (2015). The affine particle-in-cell method. *ACM Transactions on Graphics*, 34(4), 51.
- Jiang, C., Schroeder, C., & Teran, J. (2017). An angular momentum conserving affine-particle-in-cell method. *Journal of Computational Physics*, 338, 137–164.
- Bardenhagen, S. G., Brackbill, J. U., & Sulsky, D. (2000). The material-point method for granular materials. *Computer Methods in Applied Mechanics and Engineering*, 187(3–4), 529–541.
- Crook, A., & Homel, M. (2026). Reference-configuration cohesive zones for MPM. *Computer Methods in Applied Mechanics and Engineering* (in press).
- Kakouris, E. G., & Triantafyllou, S. P. (2019). Phase-field material point method for brittle fracture. *International Journal for Numerical Methods in Engineering*, 120(3), 257–279.
- Fei, Y., Gao, M., Li, M., & Jiang, C. (2021). A massively parallel and scalable multi-GPU material point method. *ACM Transactions on Graphics*, 40(4), 30.
- Blatny, L., & Gaume, J. (2025). A versatile snow and ice rheology for the material point method. *Geoscientific Model Development*, 18, 389–413.
