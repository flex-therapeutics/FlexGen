# FlexGen

All-atom protein conformational sampling with joint backbone-frame and side-chain latent flow, using a frozen Proteina-Complexa encoder–decoder.

## Install

Use Python 3.12. Install PyTorch and a compatible `torch-scatter` build, then:

```bash
git clone https://github.com/flex-therapeutics/FlexGen.git
cd FlexGen
pip install .
git clone https://github.com/NVIDIA-BioNeMo/Proteina-Complexa.git
git -C Proteina-Complexa checkout acf55a89a9a7206d5ce9abc0bf2246c3b3829f92
curl -fL https://github.com/flex-therapeutics/FlexGen/releases/download/v1.0.0/FlexGen.pt -o FlexGen.pt
curl -fL https://api.ngc.nvidia.com/v2/models/nvidia/clara/proteina_complexa/versions/1.0/files/complexa_ae.ckpt -o complexa_ae.ckpt
```

## Sample

```bash
flexgen --input protein.pdb --output ensemble.pdb \
  --weights FlexGen.pt --codec-weights complexa_ae.ckpt \
  --complexa-root Proteina-Complexa --samples 60 --device cuda:0
```

Provide one continuous protein chain with standard residues and complete N/CA/C atoms. Output is a multi-model all-atom PDB in the input coordinate frame, with unchanged amino-acid identities. Missing side-chain atoms are generated. Defaults: 32 steps, batch size 1, noise scale 1. Use `--batch-size` to increase throughput and `--noise-scale` in [0,1] to reduce motion; noise scale is not a calibrated RMSD. Keep seed and batch size fixed for reproducibility.

## Fix residues or individual atoms

**Whole residues:** add `--fix A:20-25,A:40` to the command above. This fixes all available input atoms in chain A, residues 20–25 and 40.

**Specific atoms:** create `jobs.json`:

```json
{
  "active_site": {
    "input": "protein.pdb",
    "select_fixed_atoms": {
      "A20-25": "ALL",
      "A40": "BKBN",
      "A53": "OG",
      "A60": "CG,OD1,OD2"
    }
  }
}
```

Selections use **PDB chain and residue numbers**, not sequence positions:

| Selection | Atoms fixed |
|---|---|
| `"A20-25": "ALL"` | All available atoms in residues A20–A25 |
| `"A40": "BKBN"` | N, CA, C and O of residue A40 |
| `"A53": "OG"` | Only OG of residue A53 (e.g. serine) |
| `"A60": "CG,OD1,OD2"` | These three atoms of residue A60 (e.g. aspartate) |

Replace the example residues with those in your protein. Selected atoms must exist in the input; invalid selections raise errors. `A:20-25` is also accepted; `A:30A` specifies insertion code A. Set `select_fixed_atoms` to `false` for no fixing or `true` for all available protein atoms.

```bash
flexgen --inputs jobs.json --out-dir outputs \
  --weights FlexGen.pt --codec-weights complexa_ae.ckpt \
  --complexa-root Proteina-Complexa --samples 60 --device cuda:0
```

Input paths are relative to the JSON file. Each named job produces a PDB and a JSON record of its settings and fixed atoms. Fixing conditions each integration step, with exact selected coordinates restored at output. Whole-residue fixing follows the paper procedure; partial-residue fixing additionally decodes, replaces selected coordinates and re-encodes at each step. This slower extension has not been benchmarked in the paper.

Inference code only. See [NOTICE.md](NOTICE.md) for third-party software and model terms.
