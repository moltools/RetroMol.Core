"""
Author:         David Meijer
Licence:        MIT License
Description:    Wrapper functions around RDKit for reaction chemistry.
Dependencies:   python>=3.10
                RDKit>=2023.03.1
                numpy>=1.21.2
"""
import typing as ty
from logging import Logger

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

def smiles_to_mol(
    smiles: str, 
    add_atom_mapping: bool = False,
    sanitize: bool = True,
    logger: ty.Optional[Logger] = None,
) -> ty.Optional[Chem.Mol]:
    """
    Convert a SMILES string to an RDKit molecule.
    
    Parameters
    ----------
    smiles : str
        SMILES string of the molecule.
    add_atom_mapping : bool, optional
        Whether to add atom mapping numbers to the atoms in the molecule after
        converting from SMILES, by default False. Atoms will be mapped as 
        its atom index in the molecule plus one. I.e., the first atom in the
        molecule (with index 0) will be mapped as 1, the second atom as 2, etc.
    sanitize : bool, optional
        Whether to sanitize the molecule, by default True, by default True.
    logger : ty.Optional[Logger], optional
        Logger for logging, by default None.
    
    Returns
    -------
    ty.Optional[Chem.Mol]
        RDKit molecule. None if the SMILES could not be converted.

    Raises
    ------
    TypeError
        If the SMILES is not a string.
    """
    if logger is not None:
        msg = (
            f"Converting SMILES '{smiles}' to a molecule:"
            f"\n\tadd_atom_mapping: {add_atom_mapping}"
            f"\n\tsanitize: {sanitize}"
        )
        logger.debug(msg)

    if not isinstance(smiles, str):
        msg = f"SMILES must be a string, not {type(smiles)}."

        if logger is not None:
            logger.error(msg)

        raise TypeError(msg)

    mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)

    if mol is None:
        
        if logger is not None:
            msg = f"Could not convert SMILES '{smiles}' to a molecule."
            logger.error(msg)

        return None
    
    if add_atom_mapping:
        # Add atom mapping numbers to the atoms in the molecule.
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx() + 1)

    return mol

def mol_to_smiles(
    mol: Chem.Mol, 
    remove_atom_mapping: bool = False,
    isomeric_smiles: bool = True,
    kekule_smiles: bool = False,
    rooted_at_atom: int = -1,
    canonical: bool = True,
    all_bonds_explicit: bool = False,
    all_hs_explicit: bool = False,
    logger: ty.Optional[Logger] = None,
) -> str:
    """
    Convert an RDKit molecule to a SMILES string.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule.
    remove_atom_mapping : bool, optional
        Whether to remove atom mapping numbers from the atoms in the molecule 
        before converting to SMILES, by default False.
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
    logger : ty.Optional[Logger], optional
        Logger for logging, by default None.
    
    Returns
    -------
    str
        SMILES string.

    Raises
    ------
    TypeError
        If the molecule is not an RDKit molecule.
    """
    if logger is not None:
        msg = (
            f"Converting molecule to SMILES:",
            f"\n\tremove_atom_mapping: {remove_atom_mapping}",
            f"\n\tisomeric_smiles: {isomeric_smiles}",
            f"\n\tkekule_smiles: {kekule_smiles}",
            f"\n\trooted_at_atom: {rooted_at_atom}",
            f"\n\tcanonical: {canonical}",
            f"\n\tall_bonds_explicit: {all_bonds_explicit}",
            f"\n\tall_hs_explicit: {all_hs_explicit}",
        )
        logger.debug(msg)

    if not isinstance(mol, Chem.Mol):
        msg = f"mol must be an RDKit molecule, not {type(mol)}."

        if logger is not None:
            logger.error(msg)

        raise TypeError(msg)
    
    if remove_atom_mapping:
        # Remove atom mapping numbers from the atoms in the molecule.
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)

    smiles = Chem.MolToSmiles(
        mol, 
        isomericSmiles=isomeric_smiles,
        kekuleSmiles=kekule_smiles,
        rootedAtAtom=rooted_at_atom,
        canonical=canonical,
        allBondsExplicit=all_bonds_explicit,
        allHsExplicit=all_hs_explicit,
    )

    return smiles 

def mol_to_fingerprint(
    mol: Chem.Mol, 
    radius: int = 2, 
    num_bits: int = 2048,
    logger: ty.Optional[Logger] = None,
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
    logger : ty.Optional[Logger], optional
        Logger for logging, by default None.
    
    Returns
    -------
    np.array
        Morgan fingerprint.

    Raises
    ------
    TypeError
        If the molecule is not an RDKit molecule.
    """
    if logger is not None:
        msg = (
            f"Converting molecule to Morgan fingerprint:"
            f"\n\tradius: {radius}"
            f"\n\tnum_bits: {num_bits}"
        )
        logger.debug(msg)

    if not isinstance(mol, Chem.Mol):
        msg = f"mol must be an RDKit molecule, not {type(mol)}."

        if logger is not None:
            logger.error(msg)

        raise TypeError(msg)

    # Create an empty array to store the fingerprint.
    fp_arr = np.zeros((0,), dtype=np.int8)

    # Generate the fingerprint.
    fp_vect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=num_bits)

    # Convert the fingerprint to the empty numpy array.
    DataStructs.ConvertToNumpyArray(fp_vect, fp_arr)
    
    return fp_arr
