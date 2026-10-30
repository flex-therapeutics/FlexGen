"""Generate all-atom ensembles with the paper FlexGen model."""

import argparse
import hashlib
import sys
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from .structure import read_structure, select_residues


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--codec-weights", required=True)
    p.add_argument(
        "--complexa-root", required=True, help="Released Proteina-Complexa checkout"
    )
    p.add_argument("--samples", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--fix",
        default="",
        help="Fix all available input atoms in PDB residues, e.g. A:20-25,A:40",
    )
    p.add_argument(
        "--noise-scale",
        type=float,
        default=1.0,
        help="Prior noise multiplier in [0,1]; not calibrated RMSD",
    )
    a = p.parse_args()
    if min(a.samples, a.batch_size, a.steps) < 1:
        p.error("samples, batch-size and steps must be positive")
    if not 0 <= a.noise_scale <= 1:
        p.error("noise-scale must lie in [0,1]")
    residues, atoms = read_structure(a.input)
    fixed = np.zeros(len(residues), dtype=bool)
    fixed[select_residues(a.fix, residues)] = True
    root = Path(a.complexa_root).resolve()
    if not (root / "src/proteinfoundation/partial_autoencoder/encoder.py").is_file():
        p.error("Invalid Complexa source directory")
    sys.path[:0] = [str(root / "src"), str(root / "community_models")]
    from .latent_model import JointLatentModel
    from .latent_codec import FrozenComplexaCodec
    from .sampling import sample_fixed
    from .pdb_writer import write_pdb

    ckpt = torch.load(a.weights, map_location="cpu", weights_only=True)
    cfg = OmegaConf.create(ckpt["config"])
    if ckpt["architecture_version"] != JointLatentModel.architecture_version:
        raise ValueError("Incompatible model weights")
    with open(a.codec_weights, "rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    if digest != ckpt["codec_sha256"]:
        raise ValueError("Incorrect frozen codec weights")
    model = JointLatentModel(cfg.model).to(a.device).eval()
    model.load_state_dict(ckpt["state_dict"], strict=True)
    codec = FrozenComplexaCodec(a.codec_weights).to(a.device)
    center = np.mean([x["CA"] for x in atoms], axis=0)
    results = []
    with torch.inference_mode():
        for start in range(0, a.samples, a.batch_size):
            out = sample_fixed(
                model,
                codec,
                cfg,
                a.input,
                fixed,
                min(a.batch_size, a.samples - start),
                a.seed + start,
                a.steps,
                a.device,
                prior_scale=a.noise_scale,
            )
            results.append(out["atom14"].cpu().numpy() + center)
    coords = np.concatenate(results)
    # Preserve fixed input coordinates exactly at PDB precision.
    from .chemistry import ATOM14_NAMES

    for i in np.flatnonzero(fixed):
        for j, name in enumerate(ATOM14_NAMES[residues[i][3]]):
            if name in atoms[i]:
                coords[:, i, j] = atoms[i][name]
    path = Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_pdb(
        path,
        coords,
        out["aatype"][0].cpu().numpy(),
        out["atom14_mask"][0].cpu().numpy(),
        residues,
    )
    print(path.resolve())


if __name__ == "__main__":
    main()
