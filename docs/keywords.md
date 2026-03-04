# Keywords & Syntax

## Fix Command Syntax

The package is split into a parent fix and four optional companion fixes.
The parent fix **must** be defined before any companion fix.

### Parent fix (`fix coupmpm`)

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

### Companion fix: contact (`fix coupmpm/contact`)

```
fix ID group-ID coupmpm/contact \
    method <none|bardenhagen [mu V] [adhesion V]|penalty>
```

### Companion fix: cohesive zones (`fix coupmpm/cohesive`)

```
fix ID group-ID coupmpm/cohesive    \
    law <needleman|linear|receptor> \
    sigma <val>                     \
    delta <val>                     \
    delta_max <val>                 \
    form_dist <val>                 \
    interval <N>
```

### Companion fix: adaptivity (`fix coupmpm/adaptivity`)

```
fix ID group-ID coupmpm/adaptivity \
    J_split <val>                  \
    J_merge <val>                  \
    interval <N>
```

### Companion fix: output (`fix coupmpm/output`)

```
fix ID group-ID coupmpm/output    \
    vtk_interval <N>              \
    vtk_prefix <prefix>           \
    surface_interval <N>          \
    surface_alpha <val>
```

> **Prerequisites:** The simulation must use `atom_style mpm`. All keywords are optional; defaults are listed in the tables below. Each companion fix must appear **after** `fix coupmpm` in the input script.

---

## Keyword Reference

### `fix coupmpm` keywords

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

### `fix coupmpm/contact` keywords

| Keyword | Arguments | Default | Description |
|---|---|---|---|
| `method` | `none` \| `bardenhagen [mu V] [adhesion V]` \| `penalty` | `none` | Multi-body contact algorithm. `bardenhagen` activates the Bardenhagen-Kober impenetrability algorithm; optional sub-keywords set the Coulomb friction coefficient `mu` (dimensionless, default 0.3) and the surface-energy adhesion parameter `adhesion` (J m⁻², default 0). `penalty` activates a spring-dashpot penalty force (parameters set internally). |

### `fix coupmpm/cohesive` keywords

| Keyword | Arguments | Default | Description |
|---|---|---|---|
| `law` | `needleman` \| `linear` \| `receptor` | `needleman` | Traction-separation law for cohesive bonds. |
| `sigma` | `val` | `100.0` | Peak cohesive traction σ_max (stress units). |
| `delta` | `val` | `1 × 10⁻⁴` | Characteristic separation length δ_0 at peak traction (length units). |
| `delta_max` | `val` | `2 × 10⁻⁴` | Failure separation δ_max beyond which a bond is irreversibly ruptured (length units). |
| `form_dist` | `val` | `5 × 10⁻⁴` | Maximum particle-pair distance at which a new cohesive bond may form (length units). |
| `interval` | `N` | `10` | Check for new bond-formation candidates and rupture events every *N* timesteps. |

### `fix coupmpm/adaptivity` keywords

| Keyword | Arguments | Default | Description |
|---|---|---|---|
| `J_split` | `val` | `2.0` | Jacobian upper threshold. A particle with *J* > `J_split` is split into child particles to restore resolution in highly stretched regions. |
| `J_merge` | `val` | `0.3` | Jacobian lower threshold. Particles with *J* < `J_merge` are candidates for merging with a nearest neighbour to remove over-compressed particles. |
| `interval` | `N` | `20` | Check and apply splitting/merging every *N* timesteps. |

### `fix coupmpm/output` keywords

| Keyword | Arguments | Default | Description |
|---|---|---|---|
| `vtk_interval` | `N` | `0` (disabled) | Write VTK files (grid and particles) every *N* timesteps. A PVD time-series index file is maintained automatically. |
| `vtk_prefix` | `prefix` | `coupmpm` | Filename prefix for all VTK output files. |
| `surface_interval` | `N` | `10` | Recompute the particle surface-normal field every *N* timesteps using the density-gradient method. |
| `surface_alpha` | `val` | `0.1` | Fractional threshold for surface classification: nodes with ρ < `surface_alpha` × ρ_max are flagged as surface nodes. |

---

## Extended Examples

### Uniaxial Compression

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
