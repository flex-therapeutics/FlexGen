import numpy as np
import torch
from flexgen._vendor.openfold.np import residue_constants as rc


def atom14_layout(topology, seq_letters):
    res = [r for r in topology.residues if r.name in rc.restype_3to1]
    if len(res) != len(seq_letters):
        raise ValueError(
            f"{len(res)} standard residues vs {len(seq_letters)} sequence letters -- the loader and the topology disagree about the chain"
        )
    n = len(res)
    idx = np.zeros((n, 14), dtype=np.int64)
    msk = np.zeros((n, 14), dtype=np.float32)
    aat = np.zeros(n, dtype=np.int64)
    for i, r in enumerate(res):
        letter = rc.restype_3to1[r.name]
        if letter != seq_letters[i]:
            raise ValueError(
                f"residue {i}: topology says {letter}, sequence says {seq_letters[i]} -- refusing to build a misaligned target"
            )
        aat[i] = rc.restype_order[letter]
        have = {a.name: a.index for a in r.atoms if a.element.symbol != "H"}
        for j, an in enumerate(rc.restype_name_to_atom14_names[r.name]):
            if an and an in have:
                idx[i, j] = have[an]
                msk[i, j] = 1.0
    return (torch.from_numpy(idx), torch.from_numpy(msk), torch.from_numpy(aat))


def atom14_from_frame(xyz_nm, idx, mask, ca_centroid):
    x = torch.as_tensor(np.asarray(xyz_nm), dtype=torch.float32) * 10.0
    out = x[idx]
    out = out - torch.as_tensor(ca_centroid, dtype=torch.float32)
    return out * mask[..., None]
