"""Input-defined coordinate gauge and rigid alignment of training endpoints."""

import torch
from torch.nn.functional import normalize


def frames_from_atom14(x):
    ex = normalize(x[..., 2, :] - x[..., 1, :], dim=-1)
    ey = x[..., 0, :] - x[..., 1, :]
    ey = normalize(ey - (ey * ex).sum(-1, keepdim=True) * ex, dim=-1)
    return torch.stack((ex, ey, torch.linalg.cross(ex, ey)), dim=-1)


def apply_rotation(x, q):
    return torch.einsum("bij,b...j->b...i", q, x)


@torch.no_grad()
def canonicalize(batch, align_target=False):
    """Use the first valid input frame as axes; no target defines the gauge.

    Endpoint Kabsch alignment removes only target rigid-body motion, not internal
    deformation. It never changes the prior based on a target.
    """
    out = dict(batch)
    mask = batch["res_mask"]
    w = mask / mask.sum(-1, keepdim=True).clamp_min(1)
    eq = batch["trans_equilibrium"]
    center = (eq * w[..., None]).sum(1)
    eq_centered = eq - center[:, None]
    if align_target:
        target = batch["trans_1"]
        tc = (target * w[..., None]).sum(1)
        target = target - tc[:, None]
        cov = torch.einsum("bni,bnj,bn->bij", target, eq_centered, w)
        u, _, vh = torch.linalg.svd(cov)
        v = vh.transpose(-1, -2)
        correction = torch.eye(3, device=eq.device).expand(len(eq), 3, 3).clone()
        correction[:, -1, -1] = torch.linalg.det(v @ u.transpose(-1, -2))
        q = v @ correction @ u.transpose(-1, -2)
        out["trans_1"] = apply_rotation(target, q) + center[:, None]
        out["rotmats_1"] = q[:, None] @ batch["rotmats_1"]
        out["atom14_1"] = (
            apply_rotation(batch["atom14_1"] - tc[:, None, None], q)
            + center[:, None, None]
        )
    anchor = mask.argmax(-1)
    q = batch["rotmats_equilibrium"][
        torch.arange(len(eq), device=eq.device), anchor
    ].transpose(-1, -2)
    for key in ["trans_equilibrium", "trans_1"]:
        if key in out:
            out[key] = apply_rotation(out[key] - center[:, None], q) * mask[..., None]
    for key in ["rotmats_equilibrium", "rotmats_1"]:
        if key in out:
            rotated = q[:, None] @ out[key]
            out[key] = torch.where(
                mask[..., None, None].bool(), rotated, torch.eye(3, device=eq.device)
            )
    for key in ["atom14_equilibrium", "atom14_1"]:
        if key in out:
            am = (
                out["atom14_equilibrium_mask"]
                if key.endswith("equilibrium")
                else out["atom14_mask"]
            )
            out[key] = (
                apply_rotation(out[key] - center[:, None, None], q) * am[..., None]
            )
    out["gauge_rotation"] = q
    out["gauge_center"] = center
    return out


def restore_atoms(x, batch):
    return (
        apply_rotation(x, batch["gauge_rotation"].transpose(-1, -2))
        + batch["gauge_center"][:, None, None]
    )
