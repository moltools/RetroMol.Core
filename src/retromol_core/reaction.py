"""
Author:         David Meijer
Licence:        MIT License
Description:    Wrapper functions around RDKit for reaction chemistry.
Dependencies:   python>=3.10
                RDKit>=2023.03.1
"""
import typing as ty 
from functools import wraps
from logging import Logger

import numpy as np
from rdkit import Chem 

from .utils import mol_to_fingerprint

class SanitizationError(Exception):
    """
    Exception raised when a molecule cannot be sanitized.
    """
    def __init__(self, mol: Chem.Mol) -> None:
        """
        Parameters
        ----------
        mol : Chem.Mol
            RDKit molecule that cannot be sanitized.

        Raises
        ------
        Exception
            Exception with error message.
        """
        msg = f"Failed to sanitize molecule: '{Chem.MolToSmiles(mol)}'"
        super().__init__(msg)

def mol_to_encoding(
    mol: Chem.Mol, 
    radius: int, 
    num_bits: int, 
    N: int,
    logger: ty.Optional[Logger] = None,
) -> int:
    """
    Convert an RDKit molecule to a binary fingerprint encoding. The encoding
    includes the atom map numbers of the molecule. The atom map numbers are
    appended to the fingerprint encoding as a binary vector of length `N`, where
    `N` is the number of atoms in the (parent) molecule.

    This encoding is used to create unique encodings for repetitive sub-
    structures from the same molecule. This is useful for reaction rules that
    are applied to molecules with repetitive substructures, such as
    polymerization reactions.

    NOTE: Make sure to add atom mapping numbers to the atoms in the molecule
    before converting to an encoding.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule.
    radius : int
        Radius of the Morgan fingerprint.
    num_bits : int
        Number of bits of the Morgan fingerprint.
    N : int
        Number of atoms in the (parent) molecule.
    logger : ty.Optional[Logger], optional
        Logger for logging, by default None.

    Returns
    -------
    int
        Binary fingerprint encoding of the molecule.
    """
    if logger is not None:
        msg = (
            f"Converting molecule '{Chem.MolToSmiles(mol)}' to binary fingerprint encoding.",
            f"\n\tradius: {radius}",
            f"\n\tnum_bits: {num_bits}",
            f"\n\tN: {N}",
        )
        logger.debug(msg)

    # Get all asigned atom map numbers in the molecule.
    amns = [
        atom.GetAtomMapNum()            # Atom map number.
        for atom in mol.GetAtoms()      # Atoms in molecule.
        if atom.GetAtomMapNum() > 0     # Only track atoms with assigned atom atom map numbers.
    ]

    # Create binary vector of length `N` with 1's at the indices of the atom map
    # numbers in `amns`.
    amns = np.array([1 if x in amns else 0 for x in np.arange(N)])

    # Convert molecule to fingerprint.
    fp = mol_to_fingerprint(mol, radius, num_bits)

    # Concatenate fingerprint and atom map number vector.
    conc_fp = np.hstack([fp, amns])

    # Convert fingerprint to integer.
    encoding = hash(conc_fp.data.tobytes())

    return encoding

def reaction_rule(smarts: str, logger: ty.Optional[Logger] = None) -> ty.Callable:
    """
    Decorator for defining pattern-based reaction rules.

    Parameters
    ----------
    smarts : str
        SMARTS pattern of the reaction rule.
    logger : ty.Optional[Logger], optional
        Logger for logging, by default None.
    
    Returns
    -------
    ty.Callable
        Decorator function.
    """
    pattern = Chem.MolFromSmarts(smarts)

    def decorator(func: ty.Callable) -> ty.Callable:
        """
        Decorator for applying reaction rules to molecules.

        Parameters
        ----------
        func : ty.Callable
            Reaction rule function that is applied to molecules that match the
            SMARTS pattern.
        
        Returns
        -------
        ty.Callable
            Wrapped reaction rule function.
        """

        @wraps(func) # Preserve function metadata.
        def wrapped(mol: Chem.Mol, **kwargs) -> ty.Generator[ty.List[Chem.Mol], None, None]:
            """
            Wrapped reaction rule function.

            Parameters
            ----------
            mol : Chem.Mol
                Molecule to apply the reaction rule to.
            **kwargs
                Keyword arguments for the reaction rule function.
            
            Yields
            ------
            ty.Generator[ty.List[Chem.Mol], None, None]
                List of products of the reaction rule. Each product list is a 
                list of molecules as a result of the reaction rule on a specific 
                match of the SMARTS pattern.

            Raises
            ------
            SanitizationError
                Raised when the result of the reaction rule cannot be sanitized.
            """
            if logger is not None:
                msg = (
                    f"Applying reaction rule '{func.__name__}' "
                    f"to molecule '{Chem.MolToSmiles(mol)}'."
                    f"\n\tsmarts: {smarts}"
                )
                logger.debug(msg)

            # Check if `mol` matches `pattern`.
            match list(mol.GetSubstructMatches(pattern)):
                case []: 
                    
                    if logger is not None:
                        msg = (
                            f"Reaction rule '{func.__name__}' "
                            f"did not match molecule '{Chem.MolToSmiles(mol)}'."
                            f"\n\tsmarts: {smarts}",
                        )
                        logger.debug(msg)

                    # Molecule does not match `pattern`, yield empty list.
                    yield []
                
                case matches:
                    # Apply reaction rule to each match of `pattern` on `mol`.
                    for match in matches:
                        
                        # Wrapped reaction rule is called here and is passed the
                        # molecule, the match of the SMARTS pattern, and any
                        # keyword arguments (e.g., Logger).
                        if result := func(Chem.RWMol(mol), list(match), **kwargs):

                            if logger is not None:
                                msg = (
                                    f"Successfully applied reaction rule '{func.__name__}' "
                                    f"to molecule '{Chem.MolToSmiles(mol)}'."
                                    f"\n\tsmarts: {smarts}"
                                )
                                logger.debug(msg)        

                            # Successfully applied reaction rule.
                            env = result.GetMol()

                            # Sanitize environment to update flags, properties,
                            # and hydrogen counts based on the current state of
                            # the molecule (i.e., covalent bonds).
                            try:
                                Chem.SanitizeMol(env)
                            except Exception as _:
                                # Failed to sanitize environment, raise custom
                                # sanitization exception.
                                raise SanitizationError(env)

                            # The result environment may contain multiple
                            # disconnected fragments. Yield each fragment as a
                            # product of the reaction rule.
                            products = list(Chem.GetMolFrags(env, asMols=True))

                            # Yield products of reaction rule as a result of a
                            # match of `pattern` on `mol`.
                            yield products

                        else: 
                            
                            if logger is not None:
                                msg = (
                                    f"Failed to apply reaction rule '{func.__name__}' "
                                    f"to molecule '{Chem.MolToSmiles(mol)}'."
                                    f"\n\tsmarts: {smarts}"
                                )
                                logger.error(msg)

                            # Failed to apply reaction rule, yield empty list.
                            yield []

        return wrapped
    
    return decorator