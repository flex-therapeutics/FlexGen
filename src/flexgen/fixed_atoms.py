import torch
from flexgen._vendor.gafl.data import all_atom
from flexgen.flow import to_local


def fixed_atoms(trans, rot, res_mask, chain_ids=None):
    """Use the inherited N/CA/C construction and neighbor-aware oxygen placement.

    Split known chain boundaries before oxygen placement. Geometry is rebuilt from the CURRENT evolving frames. Backbone atoms are
    constrained by frame geometry, while sidechain coordinates are a joint flow state.
    """
    if chain_ids is None:
        atom37 = all_atom.atom37_from_trans_rot(trans, rot, res_mask.detach().cpu()).to(
            trans
        )
    else:
        atom37 = trans.new_zeros((*trans.shape[:2], 37, 3))
        for b in range(len(trans)):
            ids = chain_ids[b]
            starts = (
                [0] + (torch.where(ids[1:] != ids[:-1])[0] + 1).tolist() + [len(ids)]
            )
            for start, stop in zip(starts[:-1], starts[1:]):
                atom37[b : b + 1, start:stop] = all_atom.atom37_from_trans_rot(
                    trans[b : b + 1, start:stop],
                    rot[b : b + 1, start:stop],
                    res_mask[b : b + 1, start:stop].detach().cpu(),
                ).to(trans)
    fixed = trans.new_zeros((*trans.shape[:2], 14, 3))
    fixed[..., :4, :] = atom37[..., [0, 1, 2, 4], :]
    return to_local(fixed, trans, rot)
