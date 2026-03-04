#!/usr/bin/env python3
"""
Generate LAMMPS data files for atom_style mpm.

Data file format:
  atom-ID  mol-ID  atom-type  vol0  x  y  z

(vol0 precedes x y z because LAMMPS read_data.cpp extracts the last
3 columns as coordinates unconditionally.)
"""

import numpy as np


def write_data_file(filename, box, atoms, ntypes=1, masses=None):
    """
    Write a LAMMPS data file for atom_style mpm.

    Parameters
    ----------
    filename : str
    box : dict with keys 'xlo','xhi','ylo','yhi','zlo','zhi'
    atoms : list of dicts, each with:
        'id': int, 'mol': int, 'type': int,
        'x': float, 'y': float, 'z': float,
        'vol0': float
        Optionally: 'vx','vy','vz' for Velocities section
    ntypes : int
    masses : dict {type_id: mass} or None (default: all types mass 1.0)
    """
    has_vel = any('vx' in a for a in atoms)

    with open(filename, 'w') as f:
        f.write("# CoupMPM test data file\n\n")
        f.write(f"{len(atoms)} atoms\n")
        f.write(f"{ntypes} atom types\n\n")
        f.write(f"{box['xlo']:.8e} {box['xhi']:.8e} xlo xhi\n")
        f.write(f"{box['ylo']:.8e} {box['yhi']:.8e} ylo yhi\n")
        f.write(f"{box['zlo']:.8e} {box['zhi']:.8e} zlo zhi\n\n")

        # Masses
        f.write("Masses\n\n")
        if masses is None:
            masses = {t: 1.0 for t in range(1, ntypes + 1)}
        for t in sorted(masses.keys()):
            f.write(f"{t} {masses[t]:.8e}\n")
        f.write("\n")

        # Atoms section: id mol type vol0 x y z
        f.write("Atoms # mpm\n\n")
        for a in atoms:
            f.write(f"{a['id']} {a['mol']} {a['type']} "
                    f"{a['vol0']:.8e} "
                    f"{a['x']:.8e} {a['y']:.8e} {a['z']:.8e}\n")
        f.write("\n")

        # Velocities section (optional)
        if has_vel:
            f.write("Velocities\n\n")
            for a in atoms:
                vx = a.get('vx', 0.0)
                vy = a.get('vy', 0.0)
                vz = a.get('vz', 0.0)
                f.write(f"{a['id']} {vx:.8e} {vy:.8e} {vz:.8e}\n")
            f.write("\n")


def make_grid_block(xlo, xhi, ylo, yhi, zlo, zhi, dx, mol=1, atype=1,
                    start_id=1, vx=0.0, vy=0.0, vz=0.0):
    """
    Create a uniform grid of MPM particles filling a box.
    One particle per cell, centered.

    Returns list of atom dicts.
    """
    vol0 = dx * dx * dx
    atoms = []
    aid = start_id
    nx = int(round((xhi - xlo) / dx))
    ny = int(round((yhi - ylo) / dx))
    nz = int(round((zhi - zlo) / dx))

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                x = xlo + (i + 0.5) * dx
                y = ylo + (j + 0.5) * dx
                z = zlo + (k + 0.5) * dx
                atoms.append({
                    'id': aid, 'mol': mol, 'type': atype,
                    'x': x, 'y': y, 'z': z,
                    'vol0': vol0,
                    'vx': vx, 'vy': vy, 'vz': vz
                })
                aid += 1
    return atoms


def parse_lammps_log(logfile, keywords=None):
    """
    Parse a LAMMPS log file, extracting thermo output.

    Returns a dict of arrays keyed by thermo keyword.
    Example: data['KinEng'][step_index]
    """
    data = {}
    headers = None
    in_thermo = False

    with open(logfile, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("Step "):
                headers = line.split()
                for h in headers:
                    if h not in data:
                        data[h] = []
                in_thermo = True
                continue
            if in_thermo:
                if line.startswith("Loop time") or line.startswith("ERROR"):
                    in_thermo = False
                    continue
                parts = line.split()
                if len(parts) == len(headers):
                    try:
                        vals = [float(v) for v in parts]
                        for h, v in zip(headers, vals):
                            data[h].append(v)
                    except ValueError:
                        in_thermo = False

    # Convert to numpy
    for k in data:
        data[k] = np.array(data[k])

    return data


def parse_coupmpm_output(logfile, pattern):
    """
    Extract CoupMPM diagnostic lines matching a pattern.
    Returns list of matching lines.
    """
    matches = []
    with open(logfile, 'r') as f:
        for line in f:
            if pattern in line:
                matches.append(line.strip())
    return matches
