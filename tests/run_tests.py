#!/usr/bin/env python3
"""
CoupMPM Test Suite — Master Runner
===================================
Runs all tests sequentially, reports pass/fail/skip.

Usage:
    python3 run_tests.py                      # all tests, 1 rank
    python3 run_tests.py --np 4               # all tests, 4 ranks
    python3 run_tests.py --test t02           # single test
    python3 run_tests.py --lmp /path/to/lmp   # custom LAMMPS binary
    python3 run_tests.py --keep               # don't clean up outputs

Environment:
    Set COUPMPM_LMP to the LAMMPS binary path, or pass --lmp.
    The binary must be compiled with the COUPMPM package.
"""

import os
import sys
import glob
import time
import shutil
import argparse
import subprocess

# ============================================================
# Test registry — order matters (dependencies)
# ============================================================
TESTS = [
    ("t00_compile",      "Compilation smoke test",           1),
    ("t01_startup",      "atom_style mpm + fix coupmpm init",1),
    ("t02_patch_zero",   "Patch test: zero velocity",        1),
    ("t03_patch_uniform","Patch test: uniform translation",   1),
    ("t04_freefall",     "Gravity freefall (x += v*dt)",     1),
    ("t05_conservation", "Mass + momentum conservation",      1),
    ("t06_wave1d",       "1D elastic wave speed",            1),
    ("t07_contact_2body","Bardenhagen 2-body collision",      1),
    ("t08_cohesive_form","Cohesive bond formation",           1),
    ("t09_cohesive_break","Cohesive bond failure",            1),
    ("t10_adapt_split",  "Particle splitting (J > 2)",       1),
    ("t11_adapt_merge",  "Particle merging (J < 0.3)",       1),
    ("t12_surface",      "Surface detection via grad-rho",   1),
    ("t13_mpi_conserve", "Multi-rank conservation",          4),
    ("t14_bbar",         "B-bar locking test",               1),
    ("t15_inversion",    "Element inversion recovery",       1),
]

def find_lmp(args):
    """Find the LAMMPS binary."""
    if args.lmp:
        return args.lmp
    env = os.environ.get("COUPMPM_LMP")
    if env:
        return env
    # Try common paths
    for path in ["lmp", "lmp_mpi", "lmp_serial", "../lmp", "../../build/lmp"]:
        if shutil.which(path):
            return path
    return None

def run_test(test_dir, lmp_bin, np, keep):
    """Run a single test. Returns (passed, message)."""
    test_path = os.path.join(os.path.dirname(__file__), test_dir)

    if not os.path.isdir(test_path):
        return None, "SKIP: directory not found"

    in_file = os.path.join(test_path, "in.test")
    if not os.path.isfile(in_file):
        return None, "SKIP: no in.test"

    check_file = os.path.join(test_path, "check.py")

    # Run setup script if present (generates data files, etc.)
    setup_file = os.path.join(test_path, "setup.py")
    if os.path.isfile(setup_file):
        try:
            setup_result = subprocess.run(
                [sys.executable, "setup.py"],
                cwd=test_path,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=30, text=True)
            if setup_result.returncode != 0:
                return False, f"FAIL: setup.py failed: {setup_result.stderr.strip()}"
        except subprocess.TimeoutExpired:
            return False, "FAIL: setup.py timeout"

    # Build command
    if np > 1:
        cmd = ["mpirun", "-np", str(np), "--oversubscribe", lmp_bin, "-in", "in.test"]
    else:
        cmd = [lmp_bin, "-in", "in.test"]

    # Run LAMMPS
    try:
        result = subprocess.run(
            cmd, cwd=test_path,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=120, text=True)
    except subprocess.TimeoutExpired:
        return False, "FAIL: timeout (120s)"
    except FileNotFoundError as e:
        return None, f"SKIP: {e}"

    # Check LAMMPS exit code
    if result.returncode != 0:
        # Extract last few lines of error
        err_lines = result.stderr.strip().split('\n')[-5:]
        err_msg = ' | '.join(err_lines)
        return False, f"FAIL: LAMMPS exit code {result.returncode}: {err_msg}"

    # Save log for debugging
    log_file = os.path.join(test_path, "log.test")
    with open(log_file, 'w') as f:
        f.write(result.stdout)

    # Run validation script if present
    if os.path.isfile(check_file):
        try:
            check_result = subprocess.run(
                [sys.executable, "check.py"],
                cwd=test_path,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=30, text=True)
        except subprocess.TimeoutExpired:
            return False, "FAIL: check.py timeout"

        if check_result.returncode != 0:
            msg = check_result.stdout.strip().split('\n')[-1] if check_result.stdout else "unknown"
            return False, f"FAIL: {msg}"

        return True, check_result.stdout.strip().split('\n')[-1]

    # No check script — just verify LAMMPS completed
    if "Total wall time" in result.stdout:
        return True, "PASS (LAMMPS completed, no validator)"
    return False, "FAIL: LAMMPS did not complete"

def main():
    parser = argparse.ArgumentParser(description="CoupMPM Test Suite")
    parser.add_argument("--np", type=int, default=1, help="MPI ranks (default 1)")
    parser.add_argument("--lmp", type=str, default=None, help="LAMMPS binary path")
    parser.add_argument("--test", type=str, default=None, help="Run single test (e.g., t02)")
    parser.add_argument("--keep", action="store_true", help="Keep output files")
    args = parser.parse_args()

    lmp_bin = find_lmp(args)
    if not lmp_bin:
        print("ERROR: Cannot find LAMMPS binary. Set COUPMPM_LMP or use --lmp.")
        sys.exit(1)
    print(f"LAMMPS binary: {lmp_bin}")
    print(f"Default MPI ranks: {args.np}")
    print("=" * 70)

    results = []
    t0 = time.time()

    for test_dir, desc, min_np in TESTS:
        if args.test and args.test not in test_dir:
            continue

        np = max(args.np, min_np)
        label = f"[{test_dir}] {desc}"

        passed, msg = run_test(test_dir, lmp_bin, np, args.keep)

        if passed is None:
            status = "SKIP"
        elif passed:
            status = "PASS"
        else:
            status = "FAIL"

        results.append((test_dir, status, msg))
        print(f"  {status:4s}  {label:50s}  {msg}")

    elapsed = time.time() - t0
    print("=" * 70)

    n_pass = sum(1 for _, s, _ in results if s == "PASS")
    n_fail = sum(1 for _, s, _ in results if s == "FAIL")
    n_skip = sum(1 for _, s, _ in results if s == "SKIP")
    total = len(results)

    print(f"Results: {n_pass}/{total} passed, {n_fail} failed, {n_skip} skipped  [{elapsed:.1f}s]")

    if n_fail > 0:
        print("\nFailed tests:")
        for d, s, m in results:
            if s == "FAIL":
                print(f"  {d}: {m}")
        sys.exit(1)
    else:
        print("\nAll tests passed!")
        sys.exit(0)

if __name__ == "__main__":
    main()
