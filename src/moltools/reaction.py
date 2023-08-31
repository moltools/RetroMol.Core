"""
Author:         David Meijer
Licence:        MIT License
Description:    Wrapper functions around RDKit for reaction chemistry.
Dependencies:   python>=3.10
                RDKit>=2023.03.1
"""
import typing as ty 
from functools import wraps

from rdkit import Chem 

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

def reaction_rule(smarts: str) -> ty.Callable:
    """
    Decorator for defining pattern-based reaction rules.

    Parameters
    ----------
    smarts : str
        SMARTS pattern of the reaction rule.
    
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
        def wrapped(mol: Chem.Mol) -> ty.Generator[ty.List[Chem.Mol], None, None]:
            """
            Wrapped reaction rule function.

            Parameters
            ----------
            mol : Chem.Mol
                Molecule to apply the reaction rule to.
            
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
            # Check if `mol` matches `pattern`.
            match list(mol.GetSubstructMatches(pattern)):
                case []:
                    # No matches found, yield empty list.
                    yield []
                
                case matches:
                    # Apply reaction rule to each match of `pattern` on `mol`.
                    for match in matches:
                        
                        if result := func(Chem.RWMol(mol), match):
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
                            # Failed to apply reaction rule, yield empty list.
                            yield []

        return wrapped
    
    return decorator