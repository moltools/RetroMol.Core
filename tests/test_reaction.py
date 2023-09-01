#!/usr/bin/env python3
"""
Author:        David Meijer
Licence:       MIT License
Description:   Tests for the reaction module.
Dependencies:  python>=3.10
               rdkit>=2023.03.1
"""
import unittest 
import typing as ty

from rdkit import Chem, RDLogger

from retromol_core.reaction import (
    mol_to_encoding,
    reaction_rule
)

# Turn off RDKit warnings for testing.
RDLogger.DisableLog("rdApp.*")

class TestMolToEncoding(unittest.TestCase):
    """
    Tests for the function 'mol_to_encoding'.
    """
    def test_mol_to_encoding_for_identical_molecules_with_identical_mappings(self) -> None:
        """
        Test the function 'mol_to_encoding' with two identical molecules that
        have identical atom mappings.
        """
        mol_1 = Chem.MolFromSmiles(r"[C:1][C:2][C:3]")
        mol_2 = Chem.MolFromSmiles(r"[C:3][C:2][C:1]")

        radius = 2
        num_bits = 1024
        N = 3 # Number of atoms in either molecule.

        # Convert the molecules to encodings.
        encoding_1 = mol_to_encoding(mol_1, radius, num_bits, N)
        encoding_2 = mol_to_encoding(mol_2, radius, num_bits, N)

        # Check if the encodings are the same.
        self.assertEqual(encoding_1, encoding_2)

    def test_mol_to_encoding_for_identical_molecules_with_different_mappings(self) -> None:
        """
        Test the function 'mol_to_encoding' with two identical molecules that
        have different atom mappings.

        NOTE: The encoding only takes into account if atom numbers are present
        in a molecule and not how these are assigned.
        """
        mol_1 = Chem.MolFromSmiles(r"[C:1][C:2][C:3]")
        mol_2 = Chem.MolFromSmiles(r"[C:1][C:3][C:2]")

        radius = 2
        num_bits = 1024
        N = 3 # Number of atoms in either molecule.

        # Convert the molecules to encodings.
        encoding_1 = mol_to_encoding(mol_1, radius, num_bits, N)
        encoding_2 = mol_to_encoding(mol_2, radius, num_bits, N)

        # Check if the encodings are the same.
        self.assertEqual(encoding_1, encoding_2)

    def test_mol_to_encoding_for_identical_molecules_with_unique_mappings(self) -> None:
        """
        Test the function 'mol_to_encoding' with two identical molecules that
        have unique atom mappings.
        """
        mol_1 = Chem.MolFromSmiles(r"[C:1][C:2][C:3]")
        mol_2 = Chem.MolFromSmiles(r"[C:4][C:5][C:6]")

        radius = 2
        num_bits = 1024
        N = 3 # Number of atoms in either molecule.

        # Convert the molecules to encodings.
        encoding_1 = mol_to_encoding(mol_1, radius, num_bits, N)
        encoding_2 = mol_to_encoding(mol_2, radius, num_bits, N)

        # Check if the encodings are the same.
        self.assertNotEqual(encoding_1, encoding_2)

class TestReactionRuleDecorator(unittest.TestCase): 
    """
    Tests for the reaction rule decorator 'reaction_rule'.
    """
    def test_reaction_rule_as_decorator(self) -> None:
        """
        Test the reaction rule decorator functionality with a reaction that 
        breaks a disulfide bridge. After applying the reaction rule, there 
        should be two new molecules, each with a single sulfur atom. 
        """
        mol = Chem.MolFromSmiles(r"CSSC")

        @reaction_rule(r"[*:1]-[S:2]-[S:3]-[*:4]")
        def break_disulfide_bond(em: Chem.RWMol, m: ty.List[int]) -> Chem.RWMol:
            _, m1, m2, _ = m 
            em.RemoveBond(m1, m2)
            return em 
        
        results = [result for result in break_disulfide_bond(mol)]
        
        # Check if the reaction only matched once.
        self.assertEqual(len(results), 1)

        # Check if the only result returned two molecules.
        self.assertEqual(len(results[0]), 2)
    
        atoms_mol_1 = [atom.GetAtomicNum() for atom in results[0][0].GetAtoms()]
        atoms_mol_2 = [atom.GetAtomicNum() for atom in results[0][1].GetAtoms()]

        # Count number of sulfur atoms in each resulting molecule.
        self.assertEqual(atoms_mol_1.count(16), 1)
        self.assertEqual(atoms_mol_2.count(16), 1)

if __name__ == "__main__":
    unittest.main()