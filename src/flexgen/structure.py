"""Single-chain PDB input and residue selections."""

from pathlib import Path
import numpy as np


def read_structure(path):
    residues, atoms = [], []
    keys = {}
    models = 0
    for line in Path(path).read_text().splitlines():
        if line.startswith("MODEL"):
            models += 1
            if models > 1:
                raise ValueError("Input must contain one model")
        if not line.startswith("ATOM  "):
            continue
        if line[16] not in (" ", "A"):
            continue
        key = (line[21], int(line[22:26]), line[26])
        name = line[12:16].strip()
        if name.startswith("H") or (name[0].isdigit() and name[1:2] == "H"):
            continue
        if key not in keys:
            keys[key] = len(residues)
            residues.append((*key, line[17:20]))
            atoms.append({})
        atoms[keys[key]].setdefault(
            name, np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        )
    if len(residues) < 4:
        raise ValueError("At least four protein residues are required")
    if len({r[0] for r in residues}) != 1:
        raise ValueError("Provide a single protein chain")
    standard = set(
        "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split()
    )
    if any(r[3] not in standard for r in residues):
        raise ValueError("Only standard amino acids are supported")
    if any(not all(n in a for n in ("N", "CA", "C")) for a in atoms):
        raise ValueError("Complete N, CA and C atoms are required")
    if any(np.linalg.norm(a["C"] - b["N"]) > 2.0 for a, b in zip(atoms, atoms[1:])):
        raise ValueError(
            "Input contains a backbone chain break; provide a continuous chain"
        )
    return residues, atoms


def select_residues(spec, residues):
    """Comma-separated PDB identifiers: A:12,A:20-25,A:30A; '_' denotes blank chain."""
    import re

    selected = set()
    for token in filter(None, spec.split(",")):
        m = re.fullmatch(r"([^:]):(-?\d+)([A-Za-z]?)(?:-(-?\d+))?", token.strip())
        if not m:
            raise ValueError(f"Invalid residue selection: {token}")
        chain, start, icode, stop = m.groups()
        chain = " " if chain == "_" else chain
        if stop and icode:
            raise ValueError("Insertion codes cannot be combined with ranges")
        start = int(start)
        end = int(stop) if stop else start
        if end < start:
            raise ValueError("Reversed residue range")
        matches = {
            i
            for i, r in enumerate(residues)
            if r[0] == chain
            and start <= r[1] <= end
            and (bool(stop) or r[2].strip() == icode)
        }
        if not matches:
            raise ValueError(f"No input residues match {token}")
        selected.update(matches)
    return sorted(selected)
