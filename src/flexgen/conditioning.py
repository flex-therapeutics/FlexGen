from flexgen.latent_geometry import canonicalize


def prepare_latents(batch, codec, align_target=True):
    batch = canonicalize(batch, align_target=align_target)
    aa, mask = (batch["aatype"], batch["res_mask"])
    batch["latent_equilibrium"] = codec.encode(
        batch["atom14_equilibrium"],
        aa,
        batch["atom14_equilibrium_mask"],
        mask,
        batch.get("chain_ids"),
    )
    if align_target:
        batch["latent_1"] = codec.encode(
            batch["atom14_1"], aa, batch["atom14_mask"], mask, batch.get("chain_ids")
        )
    return batch
