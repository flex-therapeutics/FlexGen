from pathlib import Path
from flexgen.chemistry import RESTYPES, RESTYPE_1TO3, ATOM14_NAMES


def write_pdb(path, atoms, aa, mask, residues):
    if len(residues) != len(aa):
        raise ValueError("input residue keys and output sequence length differ")
    with Path(path).open("w") as handle:
        for model_index, sample in enumerate(atoms, 1):
            handle.write(f"MODEL     {model_index:4d}\n")
            serial = 1
            for i, (chain, number, insertion, name) in enumerate(residues):
                expected = RESTYPE_1TO3[RESTYPES[int(aa[i])]]
                if expected != name:
                    raise ValueError(f"residue {i}: input {name}, output {expected}")
                for slot, atom in enumerate(ATOM14_NAMES[name]):
                    if not atom or not mask[i, slot]:
                        continue
                    x, y, z = sample[i, slot]
                    handle.write(
                        f"ATOM  {serial:5d} {atom:>4s} {name:3s} {chain:1s}{number:4d}{insertion:1s}   {x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}          {atom[0]:>2s}\n"
                    )
                    serial += 1
            handle.write("ENDMDL\n")
        handle.write("END\n")
