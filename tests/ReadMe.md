# CoupMPM Test Suite

## Quick Start

```bash
# Set LAMMPS binary path
export COUPMPM_LMP=/path/to/lmp

# Run all tests (single rank)
python3 run_tests.py

# Run all tests on 4 MPI ranks
python3 run_tests.py --np 4

# Run a single test
python3 run_tests.py --test t02

# Keep output files for debugging
python3 run_tests.py --keep
```

## Test Hierarchy

Tests are ordered by complexity. Each test isolates one feature and depends
only on lower-numbered tests passing.

### Level 0: Compilation
| Test | What it verifies |
|------|-----------------|
| t00_compile | `atom_style mpm` is recognized (package compiled into LAMMPS) |

### Level 1: Basic Initialization
| Test | What it verifies |
|------|-----------------|
| t01_startup | `read_data` with mpm format, `fix coupmpm` grid setup, 2 steps without crash |

### Level 2: Single-Body Physics (Zero External Force)
| Test | What it verifies | Analytical check |
|------|-----------------|-----------------|
| t02_patch_zero | Zero velocity: KE stays 0, positions don't drift | KE = 0 exactly |
| t03_patch_uniform | Uniform translation: velocity preserved, F stays I | v_cm = const, KE = const |
| t04_freefall | Single particle ballistic: x = x0 + v*t | Position within 1% |
| t05_conservation | Random velocities: momentum conserved | vcm drift < 1e-6 relative |

### Level 3: Wave Mechanics
| Test | What it verifies | Analytical check |
|------|-----------------|-----------------|
| t06_wave1d | 1D elastic wave in column | Stability + momentum conservation |
| t14_bbar | B-bar with nearly incompressible material (kappa/mu=1000) | Stable for 100 steps |
| t15_inversion | Extreme compression: J approaches 0 | No NaN, no crash |

### Level 4: Multi-Body Contact
| Test | What it verifies | Analytical check |
|------|-----------------|-----------------|
| t07_contact_2body | Bardenhagen contact: symmetric collision | vcm stays ~0 |

### Level 5: Cohesive Zones
| Test | What it verifies | Check |
|------|-----------------|-------|
| t08_cohesive_form | Bond formation between close bodies | "new bonds" in log |
| t09_cohesive_break | Bond failure under separation | "bonds broken" in log |

### Level 6: Adaptivity
| Test | What it verifies | Check |
|------|-----------------|-------|
| t10_adapt_split | Particle splitting under expansion (J > threshold) | atom count increases |
| t11_adapt_merge | Particle merging under compression (J < threshold) | atom count decreases |

### Level 7: Surface Detection
| Test | What it verifies | Check |
|------|-----------------|-------|
| t12_surface | Density gradient surface detection | surface particles > 0 |

### Level 8: Parallel Correctness
| Test | What it verifies | Check |
|------|-----------------|-------|
| t13_mpi_conserve | Multi-rank (4 procs): atom count + momentum conservation | drift < 1% |

## Architecture

Each test directory contains:
- `setup.py` — generates the LAMMPS data file (run before LAMMPS)
- `in.test` — LAMMPS input script
- `check.py` — Python validation (parses `log.test`, returns exit code 0/1)

The master runner (`run_tests.py`) executes: `setup.py` -> LAMMPS -> `check.py`

### Data File Format

All data files use `atom_style mpm` format:
```
atom-ID  mol-ID  atom-type  vol0  x  y  z
```
Note: `vol0` precedes coordinates because LAMMPS `read_data.cpp` unconditionally
extracts the last 3 columns as x, y, z.

### Validation Strategy

| Method | Used for |
|--------|---------|
| Analytical solution | Patch tests, freefall, wave speed |
| Conservation laws | Every test checks mass/momentum |
| Stability | Inversion, B-bar, extreme deformation |
| Feature detection | Surface flags, bond counts, split/merge counts |
| Cross-rank comparison | MPI tests compare to single-rank baseline |

### Tolerance Philosophy

- **Exact**: mass conservation, atom count (integer quantities)
- **Machine epsilon** (1e-10): zero-velocity patch test, momentum in periodic box
- **Small** (1e-6): momentum in non-trivial flows (floating-point accumulation)
- **Moderate** (1%): single-particle trajectory (APIC round-trip dissipation)
- **Stability only**: inversion, B-bar locking (just verify no crash/NaN)

## Adding New Tests

1. Create `tests/tNN_name/` directory
2. Add `setup.py` (generates data file), `in.test` (LAMMPS input), `check.py` (validator)
3. Register in `run_tests.py` TESTS list with `(dirname, description, min_mpi_ranks)`
4. `check.py` must exit 0 on pass, 1 on fail, and print a one-line summary

## Known Limitations

- Tests assume SI units throughout
- Single-rank tests don't catch MPI-specific bugs (use t13 for that)
- No performance benchmarks (these are correctness tests only)
- VTK output is not validated (visual inspection in ParaView recommended)
- Cohesive tests depend on surface detection working first
