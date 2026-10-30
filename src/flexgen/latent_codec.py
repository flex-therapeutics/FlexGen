"""Frozen published Complexa AE."""

import torch
from torch import nn
from flexgen.chemistry import ATOM14_NAMES, RESTYPES, RESTYPE_1TO3
from flexgen._vendor.openfold.np import residue_constants as rc
from flexgen.latent_geometry import frames_from_atom14
from flexgen.fixed_atoms import fixed_atoms
from flexgen.flow import to_local, to_global


def validate_codec_input(res_mask, chain_ids):
    if not res_mask.bool().all():
        raise ValueError("Complexa requires unpadded inputs")
    if chain_ids is not None and (chain_ids != chain_ids[:, :1]).any():
        raise ValueError("Complexa requires single-chain inputs")


class FrozenComplexaCodec(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        from proteinfoundation.partial_autoencoder.encoder import EncoderTransformer
        from proteinfoundation.partial_autoencoder.decoder import DecoderTransformer

        c = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        cfg = c["hyper_parameters"]["cfg_ae"].nn_ae
        if cfg.latent_z_dim != 8:
            raise ValueError("Expected released 8-dimensional latent space")
        self.encoder = EncoderTransformer(**cfg)
        self.decoder = DecoderTransformer(**cfg)
        for name, model in [("encoder", self.encoder), ("decoder", self.decoder)]:
            model.load_state_dict(
                {
                    k[len(name) + 1 :]: v
                    for k, v in c["state_dict"].items()
                    if k.startswith(name + ".")
                },
                strict=True,
            )
        idx = torch.zeros(21, 14, dtype=torch.long)
        present = torch.zeros(21, 14)
        for i, letter in enumerate(RESTYPES):
            for j, name in enumerate(ATOM14_NAMES[RESTYPE_1TO3[letter]]):
                if name:
                    idx[i, j] = rc.atom_order[name]
                    present[i, j] = 1
        self.register_buffer("idx37", idx)
        self.register_buffer("present14", present)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    def atom37(self, x, aa, mask):
        B, N = x.shape[:2]
        idx = self.idx37[aa]
        coords = x.new_zeros(B, N, 37, 3)
        m = x.new_zeros(B, N, 37)
        coords.scatter_add_(
            2, idx[..., None].expand(-1, -1, -1, 3), x * mask[..., None]
        )
        m.scatter_add_(2, idx, mask)
        return (coords, m.clamp_max(1))

    @torch.no_grad()
    def encode(self, x, aa, atom_mask, res_mask, chain_ids=None):
        validate_codec_input(res_mask, chain_ids)
        coords, mask = self.atom37(x, aa, atom_mask * res_mask[..., None])
        B, N = x.shape[:2]
        residue = res_mask.bool()
        breaks = torch.zeros_like(residue)
        breaks[:, :-1] = (coords[:, 1:, 1] - coords[:, :-1, 1]).norm(dim=-1) > 4
        if chain_ids is not None:
            breaks[:, :-1] |= chain_ids[:, 1:] != chain_ids[:, :-1]
        batch = dict(
            coords=coords,
            coords_nm=coords / 10,
            coord_mask=mask.bool(),
            residue_type=aa,
            mask=residue,
            mask_dict={
                "coords": residue[..., None, None].expand(-1, -1, 37, 3),
                "residue_type": residue,
            },
            residue_pdb_idx=torch.arange(N, device=x.device)[None].expand(B, -1),
            chain_breaks_per_residue=breaks,
        )
        return self.encoder.ln_z(self.encoder(batch)["mean"]) * res_mask[..., None]

    def decode(self, z, trans, rot, aa, res_mask, chain_ids=None):
        pred = self.decoder(
            dict(
                z_latent=z,
                ca_coors_nm=trans / 10,
                residue_mask=res_mask.bool(),
                mask=res_mask.bool(),
            )
        )
        atoms = pred["coors_nm"] * 10
        x = atoms.gather(2, self.idx37[aa][..., None].expand(-1, -1, -1, 3))
        dec_rot = frames_from_atom14(x)
        local = to_local(x, trans, dec_rot)
        fixed = fixed_atoms(trans, rot, res_mask, chain_ids)
        local = torch.cat((fixed[:, :, :4], local[:, :, 4:]), dim=2)
        mask = self.present14[aa] * res_mask[..., None]
        return dict(
            atom14=to_global(local, trans, rot) * mask[..., None],
            atom14_mask=mask,
            seq_logits=pred["seq_logits"],
            raw_atom14=x,
        )
