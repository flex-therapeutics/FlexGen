"""Simultaneous conditional frame + pretrained latent flow in an input-defined gauge."""

import torch
from torch import nn
from flexgen._vendor.frame_flow.models.frame_network import FrameNetwork
from flexgen._vendor.gafl.data import so3_utils
from flexgen.flow import endpoint_euler_step, validate_joint_interpolant


class JointLatentModel(nn.Module):
    architecture_version = "joint_frame_complexa_latent_v1"

    def __init__(self, config):
        super().__init__()
        self.backbone = FrameNetwork(config)
        width = config.gfa.c_s
        self.state_embed = nn.Sequential(
            nn.Linear(61, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.latent_to_frame = nn.Linear(width, width)
        nn.init.zeros_(self.latent_to_frame.weight)
        nn.init.zeros_(self.latent_to_frame.bias)
        self.latent_head = nn.Sequential(
            nn.LayerNorm(2 * width),
            nn.Linear(2 * width, width),
            nn.SiLU(),
            nn.Linear(width, 8),
        )
        nn.init.zeros_(self.latent_head[-1].weight)
        nn.init.zeros_(self.latent_head[-1].bias)

    def forward(self, batch):
        B, N = batch["trans_t"].shape[:2]
        features = torch.cat(
            (
                batch["seq"][..., :20],
                batch["latent_t"],
                batch["latent_equilibrium"],
                batch["trans_t"] / 10,
                batch["rotmats_t"].flatten(-2),
                batch["trans_equilibrium"] / 10,
                batch["rotmats_equilibrium"].flatten(-2),
                batch["t"][:, None].expand(B, N, 1),
            ),
            dim=-1,
        )
        state = self.state_embed(features) * batch["res_mask"][..., None]
        result = self.backbone(
            dict(
                batch,
                atom_node_context=self.latent_to_frame(state)
                * batch["res_mask"][..., None],
            )
        )
        result["pred_latent"] = batch["latent_equilibrium"] + self.latent_head(
            torch.cat((result["node_embed"], state), dim=-1)
        )
        result["pred_latent"] *= batch["res_mask"][..., None]
        return result

    @torch.no_grad()
    def sample_joint(
        self, batch, interpolant, steps=32, latent_sigma=0.5, generator=None
    ):
        validate_joint_interpolant(interpolant._cfg)
        if steps < 1:
            raise ValueError("positive step count required")
        B, N = batch["trans_equilibrium"].shape[:2]
        trans, rot = interpolant.sample_from_prior(
            B, N, batch["trans_equilibrium"], batch["rotmats_equilibrium"]
        )
        z = batch["latent_equilibrium"] + latent_sigma * torch.randn(
            batch["latent_equilibrium"].shape, device=trans.device, generator=generator
        )
        conditioning = dict(batch)
        for k in range(steps):
            t, nt = (k / steps, (k + 1) / steps)
            pred = self(
                dict(
                    conditioning,
                    trans_t=trans,
                    rotmats_t=rot,
                    latent_t=z,
                    t=trans.new_full((B, 1), t),
                )
            )
            trans = endpoint_euler_step(trans, pred["pred_trans"], t, nt)
            rot = so3_utils.geodesic_t((nt - t) / (1 - t), pred["pred_rotmats"], rot)
            z = endpoint_euler_step(z, pred["pred_latent"], t, nt)
            if interpolant._cfg.self_condition:
                conditioning["trans_sc"] = pred["pred_trans"]
        return dict(trans=trans, rotmats=rot, latent=z)
