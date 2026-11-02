import torch
from flexgen.latent_data import input_features
from flexgen.conditioning import prepare_latents
from flexgen.rng import sampling_rng
from flexgen.latent_geometry import restore_atoms
from flexgen.flow import endpoint_euler_step
from flexgen._vendor.gafl.data import so3_utils
from flexgen._vendor.frame_flow.data.frame_interpolant import FrameInterpolant


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
    fixed_atom_mask=None,
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
    ip = FrameInterpolant(cfg.interpolant)
    ip.set_device(device)
    B, N = batch["res_mask"].shape
    known_atoms = batch["atom14_equilibrium_mask"].bool()
    selected = fixed[None, :, None] & known_atoms
    if fixed_atom_mask is not None:
        requested = torch.as_tensor(fixed_atom_mask, dtype=torch.bool, device=device)
        if requested.shape != (N, 14):
            raise ValueError("fixed_atom_mask must have shape (residues, 14)")
        if (requested[None] & ~known_atoms).any():
            raise ValueError("Cannot fix atoms absent from the input")
        selected = selected | requested[None]
        # Keep the original path for complete residue selections.
        fixed = fixed | (requested.any(-1) & (requested == known_atoms[0]).all(-1))
    fm = fixed[None, :, None]
    partial = selected & ~fm
    if mode != "allatom" and partial.any():
        raise ValueError("Atom-subset constraints require allatom mode")
    partial_residues = partial.any(-1)
    projection_calls = 0
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
            nonlocal projection_calls
            if partial.any():
                # Project each predicted endpoint through coordinates, then encode
                # the mixed constrained/free geometry back into the flow state.
                # No entire residue is frozen for a partial atom selection.
                decoded = codec.decode(z, x, r, batch["aatype"], batch["res_mask"], batch["chain_ids"])
                coords = torch.where(partial[..., None], batch["atom14_equilibrium"], decoded["atom14"])
                from flexgen.latent_geometry import frames_from_atom14
                new_rot = frames_from_atom14(coords)
                new_z = codec.encode(coords, batch["aatype"], decoded["atom14_mask"], batch["res_mask"], batch["chain_ids"])
                x = torch.where(partial_residues[..., None], coords[:, :, 1], x)
                r = torch.where(partial_residues[..., None, None], new_rot, r)
                z = torch.where(partial_residues[..., None], new_z, z)
                projection_calls += 1
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
    constrained = selected & known
    if mode == "backbone":
        constrained = constrained & (torch.arange(14, device=device)[None, None, :] < 4)
    out["atom14"] = torch.where(constrained[..., None], target, out["atom14"])
    out["aatype"] = batch["aatype"]
    out["fixed_atom_mask"] = constrained
    out["maximum_fixed_frame_drift"] = maximum_fixed_frame_drift
    out["atom_subset_projection_calls"] = projection_calls
    return out
