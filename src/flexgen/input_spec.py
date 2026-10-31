"""Named JSON jobs and protein atom selections inspired by Foundry/RFD3."""
import json
import re
from pathlib import Path
import numpy as np
from .chemistry import ATOM14_NAMES
from .structure import read_structure, select_residues


def atom_selection(spec, residues, atoms):
    """Select known input atoms; never silently expand an atom subset to a residue."""
    mask = np.zeros((len(residues), 14), dtype=bool)
    if spec is None or spec is False:
        return mask
    if spec is True:
        selections = [(list(range(len(residues))), 'ALL')]
    else:
        if isinstance(spec, str):
            spec = {spec: 'ALL'}
        if not isinstance(spec, dict):
            raise ValueError('select_fixed_atoms must be a boolean, residue string, or mapping')
        selections = []
        for selection, names in spec.items():
            if not isinstance(selection, str) or not selection.strip():
                raise ValueError('Empty or invalid residue selection')
            tokens = []
            for token in selection.split(','):
                token = token.strip()
                if ':' not in token:
                    match = re.fullmatch(r'([A-Za-z_])(-?\d+[A-Za-z]?(?:--?\d+)?)', token)
                    if not match:
                        raise ValueError(f'Invalid PDB residue selection: {token}')
                    token = match[1] + ':' + match[2]
                tokens.append(token)
            selections.append((select_residues(','.join(tokens), residues), names))
    for indices, names in selections:
        if not isinstance(names, str) or not names.strip():
            raise ValueError('Atom selection must be ALL, BKBN, or comma-separated atom names')
        names = names.strip().upper()
        for i in indices:
            layout = ATOM14_NAMES[residues[i][3]]
            if names == 'ALL':
                chosen = [n for n in layout if n and n in atoms[i]]
            elif names == 'BKBN':
                chosen = [n for n in ('N','CA','C','O') if n in atoms[i]]
            else:
                chosen = [n.strip() for n in names.split(',')]
                missing = [n for n in chosen if not n or n not in layout or n not in atoms[i]]
                if missing:
                    raise ValueError(f'Unavailable atoms {missing} in residue {residues[i][:3]}')
            if not chosen:
                raise ValueError(f'No known atoms selected in residue {residues[i][:3]}')
            for n in chosen:
                mask[i, layout.index(n)] = True
    return mask


def load_jobs(path):
    """Read named jobs; resolve input paths relative to the JSON file."""
    path = Path(path).resolve()
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    data = json.loads(path.read_text(), object_pairs_hook=unique)
    if not isinstance(data, dict) or not data:
        raise ValueError('Expected a nonempty mapping of job names to input specifications')
    jobs = []
    for name, spec in data.items():
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name):
            raise ValueError(f'Invalid job name: {name}')
        if not isinstance(spec, dict) or set(spec) - {'input','select_fixed_atoms','noise_scale'}:
            raise ValueError(f'Unsupported fields for job {name}')
        if not isinstance(spec.get('input'), str) or not spec['input']:
            raise ValueError(f'Job {name} requires an input PDB path')
        source = (path.parent / spec['input']).resolve()
        residues, atoms = read_structure(source)
        mask = atom_selection(spec.get('select_fixed_atoms', False), residues, atoms)
        scale = spec.get('noise_scale', 1.0)
        if isinstance(scale, bool) or not isinstance(scale, (int,float)) or not 0 <= scale <= 1:
            raise ValueError(f'Invalid noise_scale for job {name}')
        jobs.append(dict(name=name, input=str(source), mask=mask, noise_scale=scale,
                         residues=residues, atoms=atoms, specification=spec))
    return jobs
