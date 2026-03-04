# CoupMPM — Material Point Method Package for LAMMPS

## Overview

CoupMPM is a highly scalable Eulerian-Lagrangian Material Point Method (MPM) solver implemented as a native LAMMPS package. Designed for extreme-deformation, multi-physics simulations — including soft-matter mechanics, bio-inspired materials, and cell-aggregate dynamics — it couples a background Eulerian grid with Lagrangian material points, enabling robust handling of large strains, topological changes, and multi-body contact without mesh distortion.

## Key Features

- **APIC Transfers** — Affine Particle-In-Cell (Jiang et al. 2015/2017) eliminates angular-momentum dissipation while retaining numerical stability.
- **B-Bar Anti-Locking** — Volumetric locking in nearly incompressible materials is suppressed via the B-bar projection method.
- **Bardenhagen Contact** — Multi-body impenetrability with Coulomb friction and area-scaled adhesion (Bardenhagen–Kober 2000).
- **Dynamic Cohesive Zones** — Runtime bond formation/rupture with Needleman-Xu, linear-softening, and receptor-ligand traction-separation laws.
- **MPI Parallelism** — Full ghost-exchange protocol for P2G/G2P; builds with both LAMMPS `make` and CMake.

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

## Minimal Example

```lammps
units           si
dimension       3
boundary        f f f
atom_style      mpm
atom_modify     map array
read_data       block.data

fix mpm all coupmpm                 \
    grid 0.05 0.05 0.05             \
    kernel bspline2                 \
    bbar yes                        \
    constitutive neohookean mu 1e4 kappa 1e5 \
    dt_auto yes cfl 0.3 rho0 1200.0

fix contact all coupmpm/contact method bardenhagen mu 0.2
fix viz all coupmpm/output vtk_interval 50 vtk_prefix compress

thermo 100
timestep 1e-5
run 5000
```

## 📖 Full Documentation

For complete **Syntax & Keyword Reference**, **Theory**, and **Extended Examples**, visit the MkDocs site:

> **[https://tengzhang48.github.io/CoupMPM](https://tengzhang48.github.io/CoupMPM)**

The documentation includes:
- [User Guide](https://tengzhang48.github.io/CoupMPM/user_guide/) — overview of the user guide section
- [Keywords & Syntax](https://tengzhang48.github.io/CoupMPM/keywords/) — all fix commands, keyword tables, and extended examples
- [Theory Manual](https://tengzhang48.github.io/CoupMPM/theory/) — APIC, B-bar, Bardenhagen contact, and dynamic cohesive zones
- [Architecture](https://tengzhang48.github.io/CoupMPM/architecture/) — source layout, companion fix design, and key physics notes
- [Implementation Status](https://tengzhang48.github.io/CoupMPM/status/) — known limitations, completion status, and validation roadmap

## ⚠️ Breaking Change: Data File Column Order

The `Atoms` section column order for `atom_style mpm` has been corrected to follow the universal LAMMPS convention where coordinates (`x y z`) are always the **last three columns**.

**Old (incorrect) format:**
```
atom-ID  mol-ID  atom-type  x  y  z  vol0
```

**New (correct) format:**
```
atom-ID  mol-ID  atom-type  vol0  x  y  z
```

**Why this matters:** LAMMPS's `read_data.cpp` unconditionally extracts the last 3 columns of `size_data_atom` as the physical x, y, z coordinates before passing them to `data_atom()`. With the old format (`id mol type x y z vol0`), LAMMPS was extracting columns 4, 5, 6 as coordinates — yielding `coord = (y, z, vol0)` instead of `(x, y, z)`. Every particle loaded via `read_data` had corrupted positions.

Any existing data files using the old format must be updated: move the `vol0` field from the last column to the fourth column (before `x y z`).
