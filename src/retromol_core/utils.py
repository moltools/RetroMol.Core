"""
Author:         David Meijer
Licence:        MIT License
Description:    Wrapper functions around RDKit for reaction chemistry.
Dependencies:   python>=3.10
                RDKit>=2023.03.1
                numpy>=1.21.2
"""
import typing as ty

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

def smiles_to_mol(smiles: str, sanitize: bool = True) -> ty.Optional[Chem.Mol]:
    """
    Convert a SMILES string to an RDKit molecule.
    
    Parameters
    ----------
    smiles : str
        SMILES string of the molecule.
    sanitize : bool, optional
        Whether to sanitize the molecule, by default True, by default True.
    
    Returns
    -------
    ty.Optional[Chem.Mol]
        RDKit molecule. None if the SMILES could not be converted.
    """
    return Chem.MolFromSmiles(smiles, sanitize=sanitize)

def mol_to_smiles(
    mol: Chem.Mol, 
    isomeric_smiles: bool = True,
    kekule_smiles: bool = False,
    rooted_at_atom: int = -1,
    canonical: bool = True,
    all_bonds_explicit: bool = False,
    all_hs_explicit: bool = False,
) -> str:
    """
    Convert an RDKit molecule to a SMILES string.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule.
    isomeric_smiles : bool, optional
        Whether to include stereochemistry in the SMILES, by default True.
    kekule_smiles : bool, optional
        Whether to kekulize SMILES, by default False. 
    rooted_at_atom : int, optional
        If non-negative, this forces the SMILES to start at a particular atom.
        The atom is specified by its index in the molecule, by default -1.
    canonical : bool, optional
        Whether to canonicalize the SMILES, by default True.
    all_bonds_explicit : bool, optional
        Whether to include all bonds in the SMILES, by default False.
    all_hs_explicit : bool, optional
        Whether to include all hydrogens in the SMILES, by default False.
    
    Returns
    -------
    str
        SMILES string.
    """
    return Chem.MolToSmiles(
        mol, 
        isomericSmiles=isomeric_smiles,
        kekuleSmiles=kekule_smiles,
        rootedAtAtom=rooted_at_atom,
        canonical=canonical,
        allBondsExplicit=all_bonds_explicit,
        allHsExplicit=all_hs_explicit,
    )

def mol_to_fingerprint(
    mol: Chem.Mol, 
    radius: int = 2, 
    num_bits: int = 2048
) -> np.array:
    """
    Convert an RDKit molecule to a Morgan fingerprint.
    
    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule.
    radius : int, optional
        Radius of the fingerprint, by default 2.
    num_bits : int, optional
        Number of bits in the fingerprint, by default 2048.
    
    Returns
    -------
    np.array
        Morgan fingerprint.
    """
    # Create an empty array to store the fingerprint.
    fp_arr = np.zeros((0,), dtype=np.int8)

    # Generate the fingerprint.
    fp_vect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=num_bits)

    # Convert the fingerprint to the empty numpy array.
    DataStructs.ConvertToNumpyArray(fp_vect, fp_arr)
    
    return fp_arr
