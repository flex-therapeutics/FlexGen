import torch, mdtraj
from flexgen.data import atom14_layout, atom14_from_frame
from flexgen._vendor.bbflow.data.utils import frames_from_pdb


def input_features(path):
    traj = mdtraj.load(str(path))
    trans, rot, seq, chains = frames_from_pdb(str(path))
    letters = "".join(traj.topology.to_fasta())
    idx, mask, aa = atom14_layout(traj.topology, letters)
    if (aa >= 20).any():
        raise ValueError("Input requires standard amino acids")
    if not mask[:, :3].bool().all():
        raise ValueError("Complete N/CA/C atoms required for input frames")
    center = traj.xyz[0, traj.topology.select("name CA")].mean(0) * 10
    atoms = atom14_from_frame(traj.xyz[0], idx, mask, center)
    if not torch.allclose(atoms[:, 1], trans, atol=0.001):
        raise ValueError("input atom/frame mismatch")
    return dict(
        trans_equilibrium=trans,
        rotmats_equilibrium=rot,
        seq=seq,
        aatype=aa,
        res_mask=torch.ones(len(aa)),
        chain_ids=torch.as_tensor(chains),
        atom14_equilibrium=atoms,
        atom14_equilibrium_mask=mask,
    )
