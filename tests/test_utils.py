#!/usr/bin/env python3
"""
Author:        David Meijer
Licence:       MIT License
Description:   Tests for the utils module.
Dependencies:  python>=3.10
               rdkit>=2023.03.1
"""
import unittest

import numpy as np
from rdkit import Chem, RDLogger

from retromol_core.chem_utils import (
    smiles_to_mol, 
    mol_to_smiles, 
    mol_to_morgan_fingerprint
)

# Turn off RDKit warnings for testing.
RDLogger.DisableLog("rdApp.*")

class TestSmilesToMol(unittest.TestCase): 
    """
    Tests for the function 'smiles_to_mol'.
    """
    def test_smiles_to_mol_for_valid_smiles(self) -> None:
        """
        Test the function 'smiles_to_mol' with a valid SMILES string.
        """
        mol = smiles_to_mol(r"CC")
        self.assertIsInstance(mol, Chem.Mol)

    def test_smiles_to_mol_for_invalid_smiles(self) -> None:
        """
        Test the function 'smiles_to_mol' with an invalid SMILES string.
        """
        mol = smiles_to_mol(r"CC(")
        self.assertIsNone(mol)

    def test_smiles_to_mol_for_non_string_smiles(self) -> None:
        """
        Test the function 'smiles_to_mol' with a non-string SMILES.
        """
        smiles = None

        with self.assertRaises(TypeError):
            smiles_to_mol(smiles)

    def test_smiles_to_mol_with_atom_mapping(self) -> None:
        """
        Test the function 'smiles_to_mol' with a valid SMILES string and atom
        mapping.
        """
        mol = smiles_to_mol(r"CC", add_atom_mapping=True)
        self.assertIsInstance(mol, Chem.Mol)

        # Check if the atom mapping numbers are added to the atoms in the molecule.
        for atom in mol.GetAtoms():
            self.assertNotEqual(atom.GetAtomMapNum(), 0)

class TestMolToSmiles(unittest.TestCase):
    """
    Tests for the function 'mol_to_smiles'.
    """
    def test_mol_to_smiles_for_valid_mol(self) -> None:
        """
        Test the function 'mol_to_smiles' with a valid RDKit molecule.
        """
        mol = Chem.MolFromSmiles(r"CC")
        smiles = mol_to_smiles(mol)
        self.assertIsInstance(smiles, str)

    def test_mol_to_smiles_for_invalid_mol(self) -> None:
        """
        Test the function 'mol_to_smiles' with an invalid RDKit molecule.
        """
        mol = Chem.MolFromSmiles(r"CC(") # Invalid SMILES will return None.

        with self.assertRaises(TypeError):
            mol_to_smiles(mol)

class TestMolToFingerprint(unittest.TestCase):
    """
    Tests for the function 'mol_to_fingerprint'.
    """
    def test_mol_to_fingerprint_for_valid_mol(self) -> None:
        """
        Test the function 'mol_to_fingerprint' with a valid RDKit molecule.
        """
        mol = Chem.MolFromSmiles(r"CC")

        radius = 2
        num_bits = 1024

        fingerprint = mol_to_morgan_fingerprint(mol, radius, num_bits)

        self.assertIsInstance(fingerprint, np.ndarray)
        self.assertEqual(fingerprint.shape, (num_bits,))

    def test_mol_to_fingerprint_for_invalid_mol(self) -> None:
        """
        Test the function 'mol_to_fingerprint' with an invalid RDKit molecule.
        """
        mol = None 

        with self.assertRaises(TypeError):
            mol_to_morgan_fingerprint(mol)

if __name__ == "__main__":
    unittest.main()