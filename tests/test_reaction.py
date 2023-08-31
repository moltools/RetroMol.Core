"""
Author:        David Meijer
Licence:       MIT License
Description:   Tests for the reaction module.
Dependencies:  python>=3.10
               rdkit>=2023.03.1
"""
import unittest 
import typing as ty

from rdkit import Chem

from moltools.reaction import reaction_rule

class TestReactionRuleDecorator(unittest.TestCase): 
    """
    Tests for the reaction rule decorator 'reaction_rule'.
    """
    def rest_reaction_rule_as_decorator(self) -> None:
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