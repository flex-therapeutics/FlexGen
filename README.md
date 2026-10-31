# FlexGen

All-atom protein conformation sampling with joint backbone-frame and side-chain latent flow. The frozen Proteina-Complexa encoder–decoder supplies the atomistic representation. Input amino-acid identities are retained.

## Install

Use Python 3.12 and install PyTorch and a matching `torch-scatter` build, then:

```bash
pip install .
git clone https://github.com/NVIDIA-BioNeMo/Proteina-Complexa.git
git -C Proteina-Complexa checkout acf55a89a9a7206d5ce9abc0bf2246c3b3829f92
```

The inference code loads the external codec source directly; the complete Complexa design pipeline is not required. Obtain the FlexGen inference weights separately. Download the released codec:

```bash
curl -L 'https://api.ngc.nvidia.com/v2/models/nvidia/clara/proteina_complexa/versions/1.0/files/complexa_ae.ckpt' -o complexa_ae.ckpt
```

## Sample

```bash
flexgen --input input.pdb --output outputs/ensemble.pdb \
  --weights FlexGen.pt --codec-weights complexa_ae.ckpt \
  --complexa-root Proteina-Complexa --samples 60 --device cuda:0
```

Provide one continuous protein chain with standard residues and complete N/CA/C atoms. Output is a multi-model PDB in the input coordinate frame. Existing side-chain atoms condition sampling; missing side-chain atoms are generated.

Use `--fix A:20-25,A:40` to preserve all available input atoms in selected residues. Fixed endpoints condition every integration step; input coordinates are restored exactly at output. `--noise-scale` scales prior noise in [0,1]; it is not a calibrated RMSD. Defaults are 32 integration steps and one sample per batch. Increase `--batch-size` as memory permits; retain the same seed and batch size for reproducibility.

This repository contains inference code only. Third-party source and model terms are described in [NOTICE.md](NOTICE.md).

## JSON inputs

Use `--inputs jobs.json --out-dir outputs` instead of `--input` and `--output`:

```json
{
  "active_site": {
    "input": "protein.pdb",
    "select_fixed_atoms": {"A20-25": "ALL", "A40": "N,CA,C,O", "A53": "OG"},
    "noise_scale": 1.0
  }
}
```

Input paths are relative to the JSON file. Selections use input PDB chain/residue identifiers. `select_fixed_atoms` accepts `true` (all known protein atoms), `false`, a residue string such as `"A20-25,A40"`, or a mapping to `ALL`, `BKBN`, or explicit heavy-atom names. The colon form `A:20-25` and insertion codes such as `A:30A` are also accepted. Unknown atoms, absent input atoms, unsupported fields, and duplicate JSON keys raise errors. Ligands, new residues, and sequence redesign are not supported.

Whole-residue fixing uses the paper sampling procedure. For partial-residue selections, every predicted endpoint is decoded, selected coordinates are replaced with input coordinates, and the result is re-encoded into frames and latents. This optional extension costs additional encoder/decoder evaluations and has not been benchmarked in the paper. Re-encoding approximates the constraints in latent space; the final coordinate replacement makes the selected output atoms exact. Unselected atoms are not explicitly fixed. Each output includes a JSON record of the selected atoms and sampling settings.
