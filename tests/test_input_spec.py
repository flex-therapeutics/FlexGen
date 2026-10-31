import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from flexgen.input_spec import atom_selection, load_jobs


class Selections(unittest.TestCase):
    def setUp(self):
        self.residues = [('A', 10, ' ', 'SER'), ('A', 11, 'A', 'GLY')]
        self.atoms = [{n: np.zeros(3) for n in ('N','CA','C','O','CB','OG')},
                      {n: np.zeros(3) for n in ('N','CA','C','O')}]

    def test_partial_does_not_expand_to_residue(self):
        mask = atom_selection({'A10':'OG','A:11A':'CA'}, self.residues, self.atoms)
        self.assertEqual(mask.sum(), 2)
        self.assertTrue(mask[0,5])
        self.assertTrue(mask[1,1])

    def test_all_and_backbone(self):
        self.assertEqual(atom_selection(True,self.residues,self.atoms).sum(),10)
        self.assertEqual(atom_selection('A10',self.residues,self.atoms).sum(),6)
        self.assertEqual(atom_selection({'A10':'BKBN'},self.residues,self.atoms).sum(),4)
        self.assertFalse(atom_selection(False,self.residues,self.atoms).any())

    def test_invalid_selections(self):
        for selection in [{'A10':'ZZ'},{'A11A':'CB'},{'Z10':'ALL'},{'A10':'TIP'},'garbage']:
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                atom_selection(selection,self.residues,self.atoms)

    def test_json_validation_and_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'jobs.json'
            p.write_text(json.dumps({'test':{'input':'protein.pdb','select_fixed_atoms':{'A10':'OG'}}}))
            with patch('flexgen.input_spec.read_structure',return_value=(self.residues,self.atoms)):
                job=load_jobs(p)[0]
                self.assertEqual(job['input'],str(Path(tmp)/'protein.pdb'))
                self.assertEqual(job['mask'].sum(),1)
                for data in [{'test':{'input':'p.pdb','noise_scale':2}}, {'../bad':{'input':'p.pdb'}},
                             {'test':{'input':'p.pdb','select_fixed_atom':True}}]:
                    p.write_text(json.dumps(data))
                    with self.assertRaises(ValueError):load_jobs(p)
            p.write_text('{"x": {}, "x": {}}')
            with self.assertRaises(ValueError):load_jobs(p)

if __name__ == '__main__':
    unittest.main()
