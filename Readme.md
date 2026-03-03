# CoupMPM — Material Point Method Package for LAMMPS

A native LAMMPS package implementing the Material Point Method for large-deformation solid mechanics with modular contact, APIC transfer, B-bar anti-locking, and MPI parallelism.

**Status**: Post-audit, pre-compilation. Six bugs fixed from three independent LLM audits. Ready to compile and validate.

## File Overview

```
src/COUPMPM/
├── fix_coupmpm.h/.cpp        Main fix: timestep loop, keyword parsing, grid/MPI setup
├── atom_vec_mpm.h/.cpp       Custom atom_style: full MPI pack/unpack for all MPM fields
├── coupmpm_grid.h            Grid: SoA layout, ghost-inclusive indexing, grid solve, B-bar
├── coupmpm_kernel.h          Shape functions: linear, B-spline quadratic/cubic, gradients
├── coupmpm_transfer.h        P2G, G2P (APIC + B-bar), anti-P2G migration, MPI ghost exchange
├── coupmpm_stress.h          Constitutive: Neo-Hookean (complete), Mooney-Rivlin (stub)
├── coupmpm_contact.h         Contact: Bardenhagen multi-velocity, penalty (stub), none
├── coupmpm_surface.h         Surface detection: grad-rho with MPI-safe thresholding
├── coupmpm_adaptivity.h      Particle splitting/merging for density control
├── coupmpm_io.h              VTK grid/particle output, PVD time series
├── CMakeLists.txt            LAMMPS CMake integration
└── examples/
    └── patch_test.lmp        First validation benchmark
```

## Audit History

Three independent LLM audits identified 17 issues total. After triage:

| Category | Count | Action |
|---|---|---|
| Real bugs, fixed | 6 | See "Bugs Fixed" below |
| False alarms, rejected | 4 | See "Rejected Issues" below |
| Design issues, deferred | 4 | See "Known Limitations" below |
| Defensive improvements | 2 | Added (P2GRecord tag, energy J clamp) |

### Bugs Fixed (v1.1)

1. **Bardenhagen contact normal** (`coupmpm_contact.h`): Using relative velocity as normal forced v_n > 0 always — contact never engaged. Replaced with per-body momentum-direction difference, a geometric normal independent of kinematics.

2. **Adhesion units** (`coupmpm_contact.h`): `adhesion_gamma * m_red` mixed J/m2 with kg. Replaced with traction-based adhesion: force = gamma * A_eff. Impulse capped to prevent direction reversal.

3. **Contact impulse double-counting** (`coupmpm_contact.h`): Old code applied `dt * f / mass` but impulse was already force x dt. Fixed to `impulse / mass`.

4. **Surface detection rho_max local** (`coupmpm_surface.h`): Added `MPI_Allreduce(MPI_MAX)` for global rho_max. Without this, each rank uses a different threshold.

5. **Periodic ghost y-direction** (`coupmpm_transfer.h`): Removed `j_unused` dummy loop in `iter_ghost_and_interior_periodic` for d==1.

6. **Energy fabs(J)** (`coupmpm_stress.h`): `std::fabs(J)` in log masked inverted elements. Replaced with positive clamp.

### Defensive Improvements

7. **P2GRecord global tag** (`coupmpm_transfer.h`): Added `global_tag` from `atom->tag` for robustness against atom sorting.

8. **Mass from vol0*rho0 for adaptivity** (`fix_coupmpm.cpp`): When adaptivity enabled, mass_p computed from `vol0 * rho0` instead of per-type mass, ensuring conservation through split/merge.

### Rejected Issues (False Alarms from Audits)

- **"Anti-P2G runs too late"**: False. LAMMPS Verlet order is initial_integrate -> forces -> final_integrate -> comm->exchange. Anti-P2G in final_integrate is correctly before exchange.
- **"Move anti-P2G to pre_exchange()"**: Unnecessary given correct Verlet ordering.
- **"Atom sorting breaks anti-P2G"**: Low risk (sorting off by default). Mitigated by global tag.
- **"Cubic B-spline needs ghost=2 not 3"**: False. Cubic support [-2,2] spans 3 cells. Ghost=3 correct.

## Known Limitations (Must Fix Before Production)

| Issue | Impact | Fix |
|---|---|---|
| Bardenhagen NodeBodyData not in reverse_comm | Multi-rank contact misses ghost contributions | Defer per-body velocity to owned nodes after reverse_comm |
| grad-rho not forward-communicated | Ghost nodes have zero gradient, particles near boundaries misclassified | Add forward_comm for grad_rho_x/y/z |
| Per-type mass limits adaptivity | Split children revert to full mass from type table | Switch to per-atom rmass (mass_type=0) for heavy splitting |
| No reference-config cohesive zones | Adhesion only works within grid cell range | Implement Crook & Homel method (Phase 2) |

## Implementation Status

### Complete and Audited
- **atom_vec_mpm**: Full MPI pack/unpack (exchange, border, restart, forward, reverse). 16 methods. Data file: `atom-ID mol-ID atom-type x y z vol0`.
- **coupmpm_grid**: SoA layout, variable ghost width, grid solve, B-bar normalization, per-body Bardenhagen tracking.
- **coupmpm_kernel**: Three kernels with gradients and D_inv. Support range computation.
- **coupmpm_transfer**: P2G (APIC, stress divergence, B-bar, per-body, P2GRecord). G2P (velocity, Bp, L, div_v). Anti-P2G. MPI ghost exchange.
- **coupmpm_stress**: Neo-Hookean with safe J handling. Wave speed for CFL.
- **coupmpm_contact**: Bardenhagen with geometric normal, Coulomb friction, area-scaled adhesion.
- **coupmpm_surface**: grad-rho with global MPI threshold. Nanson area scaling.
- **coupmpm_adaptivity**: J-based splitting, nearest-neighbor merging, LAMMPS atom management.
- **coupmpm_io**: Grid VTK, particle VTK, PVD time series.
- **fix_coupmpm**: Keywords, grid/MPI setup, complete timestep, CFL, VTK, surface detection, adaptivity.

### Stubs / Future
- Mooney-Rivlin (stub), Generalized Maxwell, AFLIP transfer, reference-config cohesive zones, pair_style mpm/penalty, reaction-diffusion chemistry, checkpoint/restart.

## Build Instructions

### In-tree (traditional make)
```bash
cp -r COUPMPM/ /path/to/lammps/src/COUPMPM/
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

## Keywords

```
fix ID group-ID coupmpm &
    grid dx dy dz &
    kernel {linear | bspline2 | bspline3} &
    bbar {yes | no} &
    contact {none | bardenhagen [mu val] [adhesion val] | penalty} &
    constitutive neohookean mu val kappa val &
    dt_auto {yes | no} &
    energy_check {yes | no} &
    vtk_interval N &
    vtk_prefix prefix &
    surface_interval N &
    surface_alpha val &
    cfl val &
    rho0 val &
    adaptivity {yes | no} &
    J_split val &
    J_merge val &
    adapt_interval N
```

## Key Physics Notes

1. **B-bar two-pass**: `raw_div_v` is mass-weighted during P2G. Normalize by mass AFTER reverse comm.

2. **APIC affine matrix**: `Bp` stores C_p. APIC velocity at node: `v_p + C_p * (x_node - x_p)`. D_inv applied during G2P, not P2G.

3. **Ghost exchange completeness**: Reverse comm must include ALL P2G fields (mass, momentum, force_int, force_ext, raw_div_v).

4. **Contact normal must be geometric**: Using dv/|dv| collapses normal/tangential decomposition. Use mass gradient or momentum direction instead. This was the most critical bug found.

5. **Anti-P2G timing**: LAMMPS Verlet order is initial_integrate -> forces -> final_integrate -> comm->exchange. Anti-P2G in final_integrate is correctly before migration.

6. **Splitting mass**: Per-type mass means mass_p must derive from `vol0 * rho0` when adaptivity is on.

7. **Surface threshold**: rho_max must be globally reduced via MPI_Allreduce before computing threshold.

## Design Decisions from CoupLB

| CoupLB | CoupMPM | Notes |
|---|---|---|
| Grid SoA with f[Q*ntotal] | MPMGrid SoA with mass/momentum/force | Same ghost-inclusive indexing |
| Streaming::exchange_dim | MPMGhostExchange::reverse/forward_dim | Same neighbor topology |
| IBM::spread/interpolate | p2g/g2p (B-spline kernel) | Same loop structure |
| IO::write_vtk (Gatherv) | MPMIO::write_grid_vtk | Nearly identical |
| exchange_forces (reverse) | reverse_comm | Same ghost->owner accumulation |

## Validation Roadmap

### Phase 0: Single-rank
1. Patch test (F stays identity at zero velocity)
2. 1D elastic wave (speed = sqrt((kappa+4mu/3)/rho))
3. Splitting/merging conservation check

### Phase 1: Multi-rank
4. 1D wave on 4 ranks (drift < 1e-10)
5. Bardenhagen contact across MPI boundaries
6. Surface detection consistency across ranks

### Phase 2: Contact benchmarks
7. Hertz contact (a^3 proportional to P)
8. JKR adhesion (pull-off force)
9. Sliding block on incline

### Phase 3: Biology
10. Cell sorting (differential adhesion)
11. Cell division with contractile ring
12. LBM-MPM coupling

## References

- APIC: Jiang et al., ACM Trans. Graph. 2015, 2017
- Bardenhagen contact: Bardenhagen et al., CMAME 2000
- Cohesive zones: Crook & Homel, CMAME 2026 (GEOS-MPM)
- Penalty contact physics: Kakouris et al. (spring + dashpot)
- AFLIP: Fei et al. 2021
- Matter solver: Blatny & Gaume, Geosci. Model Dev. 2025
