import torch
from flexgen.latent_data import input_features
from flexgen.conditioning import prepare_latents
from flexgen.rng import sampling_rng
from flexgen.latent_geometry import restore_atoms
from flexgen.flow import endpoint_euler_step
from flexgen._vendor.gafl.data import so3_utils
from flexgen._vendor.bbflow.data.interpolant_bbflow import InterpolantBBFlow


@torch.no_grad()
def sample_fixed(
    model,
    codec,
    cfg,
    pdb,
    fixed,
    samples=1,
    seed=2026,
    steps=32,
    device="cuda:0",
    mode="allatom",
    prior_scale=1.0,
):
    if not 0.0 <= prior_scale <= 1.0:
        raise ValueError("prior_scale must be in [0,1]")
    if mode not in ["allatom", "backbone"]:
        raise ValueError(mode)
    raw = input_features(pdb)
    fixed = torch.as_tensor(fixed, dtype=torch.bool, device=device)
    assert fixed.shape == raw["res_mask"].shape and steps > 0
    batch = {k: v.to(device)[None].expand(samples, *v.shape) for k, v in raw.items()}
    batch = prepare_latents(batch, codec, align_target=False)
    ip = InterpolantBBFlow(cfg.interpolant)
    ip.set_device(device)
    B, N = batch["res_mask"].shape
    fm = fixed[None, :, None]
    with sampling_rng(
        seed,
        [torch.device(device).index or 0] if str(device).startswith("cuda") else [],
    ):
        trans, rot = ip.sample_from_prior(
            B, N, batch["trans_equilibrium"], batch["rotmats_equilibrium"]
        )
        z = batch["latent_equilibrium"] + cfg.training.latent_sigma * torch.randn(
            batch["latent_equilibrium"].shape,
            device=device,
            generator=torch.Generator(device=device).manual_seed(seed + 1),
        )
        cond = dict(batch)
        if prior_scale != 1.0:
            trans = batch["trans_equilibrium"] + prior_scale * (
                trans - batch["trans_equilibrium"]
            )
            rot = so3_utils.geodesic_t(prior_scale, rot, batch["rotmats_equilibrium"])
            z = batch["latent_equilibrium"] + prior_scale * (
                z - batch["latent_equilibrium"]
            )

        def project(x, r, z):
            return (
                torch.where(fm, batch["trans_equilibrium"], x),
                torch.where(fm[..., None], batch["rotmats_equilibrium"], r),
                torch.where(fm, batch["latent_equilibrium"], z)
                if mode == "allatom"
                else z,
            )

        maximum_fixed_frame_drift = 0.0
        for k in range(steps):
            t, nt = (k / steps, (k + 1) / steps)
            pred = model(
                dict(
                    cond,
                    trans_t=trans,
                    rotmats_t=rot,
                    latent_t=z,
                    t=trans.new_full((B, 1), t),
                )
            )
            px, pr, pz = project(
                pred["pred_trans"], pred["pred_rotmats"], pred["pred_latent"]
            )
            trans = endpoint_euler_step(trans, px, t, nt)
            rot = so3_utils.geodesic_t((nt - t) / (1 - t), pr, rot)
            z = endpoint_euler_step(z, pz, t, nt)
            if k == steps - 1:
                trans, rot, z = project(trans, rot, z)
            if fixed.any():
                maximum_fixed_frame_drift = max(
                    maximum_fixed_frame_drift,
                    float(
                        (trans[:, fixed] - batch["trans_equilibrium"][:, fixed])
                        .abs()
                        .max()
                    ),
                )
            if ip._cfg.self_condition:
                cond["trans_sc"] = px
    out = codec.decode(
        z, trans, rot, batch["aatype"], batch["res_mask"], batch["chain_ids"]
    )
    out["atom14"] = restore_atoms(out["atom14"], batch) * out["atom14_mask"][..., None]
    target = raw["atom14_equilibrium"].to(device)[None].expand_as(out["atom14"])
    known = raw["atom14_equilibrium_mask"].to(device).bool()[None]
    constrained = fixed[None, :, None] & known
    if mode == "backbone":
        constrained = constrained & (torch.arange(14, device=device)[None, None, :] < 4)
    out["atom14"] = torch.where(constrained[..., None], target, out["atom14"])
    out["aatype"] = batch["aatype"]
    out["fixed_atom_mask"] = constrained
    out["maximum_fixed_frame_drift"] = maximum_fixed_frame_drift
    return out
