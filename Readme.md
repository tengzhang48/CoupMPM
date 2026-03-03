# CoupMPM — Material Point Method Package for LAMMPS

## Overview

CoupMPM is a highly scalable Eulerian-Lagrangian Material Point Method (MPM) solver implemented as a native LAMMPS package. It is specifically designed for extreme-deformation, multi-physics simulations including soft-matter mechanics, bio-inspired materials, and cell-aggregate dynamics. The package couples a background Eulerian grid with Lagrangian material points, enabling robust handling of large strains, topological changes, and multi-body contact without mesh distortion.

The solver integrates directly with the LAMMPS parallelism infrastructure. All grid operations and particle-to-grid (P2G) / grid-to-particle (G2P) transfers are fully MPI-parallelized through a ghost-exchange protocol, making CoupMPM suitable for high-performance computing (HPC) deployments at arbitrary processor counts.

### Source Layout

```
CoupMPM/
├── fix_coupmpm.h/.cpp        Main fix: timestep loop, keyword parsing, grid/MPI setup
├── atom_vec_mpm.h/.cpp       Custom atom_style mpm: MPI pack/unpack for all MPM fields
├── coupmpm_grid.h            Background grid: SoA layout, ghost-inclusive indexing
├── coupmpm_kernel.h          Shape functions: linear, quadratic B-spline, cubic B-spline
├── coupmpm_transfer.h        P2G, G2P (APIC + B-bar), anti-P2G, MPI ghost exchange
├── coupmpm_stress.h          Constitutive models: Neo-Hookean, Mooney-Rivlin
├── coupmpm_contact.h         Multi-body contact: Bardenhagen, penalty
├── coupmpm_surface.h         Surface detection via density gradient with MPI reduction
├── coupmpm_adaptivity.h      Particle splitting and merging for resolution control
├── coupmpm_cohesive.h        Dynamic Cohesive Zone module for adhesion/cell sorting
└── coupmpm_io.h              VTK grid/particle output with PVD time-series index
```

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

```
fix ID group-ID coupmpm                              \
    grid <dx> <dy> <dz>                              \
    kernel <linear|bspline2|bspline3>                \
    bbar <yes|no>                                    \
    contact <none|bardenhagen [mu V] [adhesion V]|penalty> \
    constitutive <neohookean mu V kappa V            \
                 |mooneyrivlin C1 C2 kappa>          \
    dt_auto <yes|no>                                 \
    energy_check <yes|no>                            \
    vtk_interval <N>                                 \
    vtk_prefix <prefix>                              \
    surface_interval <N>                             \
    surface_alpha <val>                              \
    cfl <val>                                        \
    rho0 <val>                                       \
    adaptivity <yes|no>                              \
    J_split <val>                                    \
    J_merge <val>                                    \
    adapt_interval <N>                               \
    cohesive <yes|no>                                \
    cz_law <needleman|linear|receptor>               \
    cz_sigma <val>                                   \
    cz_delta <val>                                   \
    cz_delta_max <val>                               \
    cz_form_dist <val>                               \
    cz_interval <N>
```

> **Prerequisites:** The simulation must use `atom_style mpm`. All keywords are optional; defaults are listed in the table below.

### Keyword Reference

| Keyword | Arguments | Default | Description |
|---|---|---|---|
| `grid` | `dx dy dz` | `0.1 0.1 0.1` | Background grid cell spacings in the *x*, *y*, and *z* directions (simulation-length units). |
| `kernel` | `linear` \| `bspline2` \| `bspline3` | `linear` | Shape-function kernel. `linear` uses tent functions (support ±1 cell); `bspline2` and `bspline3` use quadratic and cubic B-spline kernels (support ±1.5 and ±2 cells, respectively), which increase smoothness and reduce quadrature noise. |
| `bbar` | `yes` \| `no` | `yes` | Enable the B-bar volumetric anti-locking correction. Recommended for nearly incompressible materials (Poisson's ratio > 0.45). |
| `contact` | `none` \| `bardenhagen [mu V] [adhesion V]` \| `penalty` | `none` | Multi-body contact algorithm. `bardenhagen` activates the Bardenhagen-Kober impenetrability algorithm; optional sub-keywords set the Coulomb friction coefficient `mu` (dimensionless, default 0.3) and the surface-energy adhesion parameter `adhesion` (J m⁻², default 0). `penalty` activates a spring-dashpot penalty force (parameters set internally). |
| `constitutive` | `neohookean mu V kappa V` \| `mooneyrivlin C1 V C2 V kappa V` | Neo-Hookean with μ = 1 × 10³, κ = 1 × 10⁴ | Hyperelastic constitutive model. `neohookean` requires the shear modulus `mu` and bulk modulus `kappa` (both > 0). `mooneyrivlin` (stub) requires the two deviatoric coefficients and the bulk modulus. |
| `dt_auto` | `yes` \| `no` | `yes` | Automatically adjust the LAMMPS timestep at each step to satisfy the CFL criterion `dt = cfl * dx / c_wave`. When `no`, the timestep specified in the input script is used unchanged. |
| `energy_check` | `yes` \| `no` | `no` | Print a kinetic + strain energy balance to the log at every step. Useful for validating conservation in single-rank benchmarks. |
| `vtk_interval` | `N` | `0` (disabled) | Write VTK files (grid and particles) every *N* timesteps. A PVD time-series index file is maintained automatically. |
| `vtk_prefix` | `prefix` | `coupmpm` | Filename prefix for all VTK output files. |
| `surface_interval` | `N` | `10` | Recompute the particle surface-normal field every *N* timesteps using the density-gradient method. |
| `surface_alpha` | `val` | `0.1` | Fractional threshold for surface classification: nodes with ρ < `surface_alpha` × ρ_max are flagged as surface nodes. |
| `cfl` | `val` | `0.3` | CFL safety factor applied to the acoustic wave-speed estimate when `dt_auto yes` is active. Typical values are 0.2–0.5. |
| `rho0` | `val` | `1000.0` | Reference (initial) mass density in simulation units. Used for CFL wave-speed estimation and adaptive-particle mass initialisation. |
| `adaptivity` | `yes` \| `no` | `no` | Enable dynamic particle splitting and merging based on the local Jacobian *J* = det(**F**). |
| `J_split` | `val` | `2.0` | Jacobian upper threshold. A particle with *J* > `J_split` is split into child particles to restore resolution in highly stretched regions. |
| `J_merge` | `val` | `0.3` | Jacobian lower threshold. Particles with *J* < `J_merge` are candidates for merging with a nearest neighbour to remove over-compressed particles. |
| `adapt_interval` | `N` | `20` | Check and apply splitting/merging every *N* timesteps. |
| `cohesive` | `yes` \| `no` | `no` | Enable the Dynamic Cohesive Zone (DCZ) module for runtime bond formation and rupture. Requires a LAMMPS neighbour list; activated automatically when `cohesive yes` is set. |
| `cz_law` | `needleman` \| `linear` \| `receptor` | `needleman` | Traction-separation law for cohesive bonds. `needleman` uses the Needleman-Xu exponential potential. `linear` uses a linearly softening law. `receptor` uses a receptor-ligand kinetic binding model suitable for cell-membrane adhesion. |
| `cz_sigma` | `val` | `100.0` | Peak cohesive traction σ_max (stress units). |
| `cz_delta` | `val` | `1 × 10⁻⁴` | Characteristic separation length δ_0 at peak traction (length units). |
| `cz_delta_max` | `val` | `2 × 10⁻⁴` | Failure separation δ_max beyond which a bond is irreversibly ruptured (length units). |
| `cz_form_dist` | `val` | `5 × 10⁻⁴` | Maximum particle-pair distance at which a new cohesive bond may form (length units). |
| `cz_interval` | `N` | `10` | Check for new bond-formation candidates and rupture events every *N* timesteps. |

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

read_data       block.data          # atom-ID mol-ID type x y z vol0

# Material point method fix
fix mpm all coupmpm                 \
    grid 0.05 0.05 0.05             \
    kernel bspline2                 \
    bbar yes                        \
    contact bardenhagen mu 0.2      \
    constitutive neohookean mu 1e4 kappa 1e5 \
    dt_auto yes                     \
    cfl 0.3                         \
    rho0 1200.0                     \
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
    contact bardenhagen mu 0.0 adhesion 5e-4 \
    constitutive neohookean mu 500.0 kappa 2000.0 \
    dt_auto yes cfl 0.25 rho0 1050.0        \
    cohesive yes                            \
    cz_law receptor                         \
    cz_sigma 200.0                          \
    cz_delta 5e-7                           \
    cz_delta_max 1.2e-6                     \
    cz_form_dist 3e-6                       \
    cz_interval 5                           \
    vtk_interval 100 vtk_prefix cells
```

---

## References

- Jiang, C., Schroeder, C., Selle, A., Teran, J., & Stomakhin, A. (2015). The affine particle-in-cell method. *ACM Transactions on Graphics*, 34(4), 51.
- Jiang, C., Schroeder, C., & Teran, J. (2017). An angular momentum conserving affine-particle-in-cell method. *Journal of Computational Physics*, 338, 137–164.
- Bardenhagen, S. G., Brackbill, J. U., & Sulsky, D. (2000). The material-point method for granular materials. *Computer Methods in Applied Mechanics and Engineering*, 187(3–4), 529–541.
- Crook, A., & Homel, M. (2026). Reference-configuration cohesive zones for MPM. *Computer Methods in Applied Mechanics and Engineering* (in press).
- Kakouris, E. G., & Triantafyllou, S. P. (2019). Phase-field material point method for brittle fracture. *International Journal for Numerical Methods in Engineering*, 120(3), 257–279.
- Fei, Y., Gao, M., Li, M., & Jiang, C. (2021). A massively parallel and scalable multi-GPU material point method. *ACM Transactions on Graphics*, 40(4), 30.
- Blatny, L., & Gaume, J. (2025). A versatile snow and ice rheology for the material point method. *Geoscientific Model Development*, 18, 389–413.
