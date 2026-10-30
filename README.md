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
