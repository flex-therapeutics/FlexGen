"""Generate all-atom ensembles with the paper FlexGen model."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from .structure import read_structure


def main():
    p = argparse.ArgumentParser(description=__doc__)
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input", help="Single input PDB")
    inputs.add_argument("--inputs", help="Foundry-style JSON with named input specifications")
    p.add_argument("--output", help="Output multi-model PDB for --input")
    p.add_argument("--out-dir", help="Output directory for named JSON jobs")
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
    from .input_spec import load_jobs, atom_selection
    if a.inputs:
        if not a.out_dir or a.output or a.fix or a.noise_scale != 1.0:
            p.error("--inputs requires --out-dir; specify fixing and noise_scale inside each JSON job")
        jobs = load_jobs(a.inputs)
        for job in jobs:
            job["output"] = str(Path(a.out_dir) / (job["name"] + ".pdb"))
    else:
        if not a.output or a.out_dir:
            p.error("--input requires --output")
        residues, atoms = read_structure(a.input)
        jobs = [dict(name=Path(a.output).stem, input=a.input, output=a.output,
                     residues=residues, atoms=atoms, mask=atom_selection(a.fix or False, residues, atoms),
                     noise_scale=a.noise_scale)]
    for job in jobs:
        if Path(job["input"]).resolve() == Path(job["output"]).resolve():
            p.error("Output must not overwrite input")
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
    from .chemistry import ATOM14_NAMES
    for job in jobs:
        residues, atoms = job["residues"], job["atoms"]
        center = np.mean([x["CA"] for x in atoms], axis=0)
        fixed = np.zeros(len(residues), dtype=bool)
        results = []
        with torch.inference_mode():
            for start in range(0, a.samples, a.batch_size):
                out = sample_fixed(
                    model, codec, cfg, job["input"], fixed,
                    min(a.batch_size, a.samples - start), a.seed + start,
                    a.steps, a.device, prior_scale=job["noise_scale"],
                    fixed_atom_mask=job["mask"],
                )
                results.append(out["atom14"].cpu().numpy() + center)
        coords = np.concatenate(results)
        # Preserve only selected known atoms exactly at input PDB precision.
        for i, j in np.argwhere(job["mask"]):
            coords[:, i, j] = atoms[i][ATOM14_NAMES[residues[i][3]][j]]
        path = Path(job["output"])
        path.parent.mkdir(parents=True, exist_ok=True)
        write_pdb(path, coords, out["aatype"][0].cpu().numpy(),
                  out["atom14_mask"][0].cpu().numpy(), residues)
        metadata = dict(input=str(Path(job["input"]).resolve()), samples=a.samples,
                        seed=a.seed, batch_size=a.batch_size, steps=a.steps,
                        noise_scale=job["noise_scale"], fixed_atoms=[
                            dict(chain=residues[i][0], residue=residues[i][1],
                                 insertion_code=residues[i][2].strip(), atom=ATOM14_NAMES[residues[i][3]][j])
                            for i,j in np.argwhere(job["mask"])],
                        atom_subset_projection_calls=out["atom_subset_projection_calls"])
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(path.resolve())


if __name__ == "__main__":
    main()
