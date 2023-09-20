"""
geom.py
========
Author:         David Meijer
Licence:        MIT License
Description:    Wrapper functions around RDKit for shape cheminformatics.
"""
import typing as ty

import numpy as np
from rdkit import Chem 
from rdkit.Chem import AllChem
from scipy.linalg import eig

def get_conformers(
    mol: Chem.Mol, 
    num_confs: int = 10, 
    max_attempts: int = 0,
    random_seed: int = -1,
    clear_confs: bool = True,
    use_random_coords: bool = False,
    box_size_mult: float = 2.0,
    random_neg_eig: bool = True,
    num_zero_fail: int = 1,
    prune_rms_thresh: float = -1.0,
    coord_map: ty.Optional[ty.Dict[int, ty.Tuple[float, float, float]]] = {},
    force_tol: float = 1e-3,
    ignore_smoothing_failures: bool = False,
    enforce_chirality: bool = True,
    num_threads: int = 1,
    use_exp_torsion_angle_prefs: bool = True,
    use_basic_knowledge: bool = True,
    print_exp_torsion_angles: bool = False,
    use_small_ring_torsions: bool = False,
    use_macrocycle_torsions: bool = False,
    et_version: int = 1,
    optimize_confs: bool = True,
    max_iters_optimization: int = 200,
    mmff_variant: str = "MMFF94",
    non_bonded_thresh: float = 100.0,
    ignore_interfrag_interactions: bool = True,
) -> ty.List[Chem.Conformer]:
    """
    Embeds and optionally optimizes conformers for a molecule.

    Parameters
    ----------
    mol : Chem.Mol
        The molecule to embed conformers for.
    num_confs : int
        The number of conformers to embed.
    max_attempts : int, optional
        The maximum number of embedding attempts, by default 0.
        If 0, the number of conformers is equal to num_confs.
    random_seed : int, optional
        The random seed to use for embedding, by default -1.
        If -1, a random seed is generated.
    clear_confs : bool, optional
        Whether to clear the conformers before embedding, by default True.
    use_random_coords : bool, optional
        Whether to use random coordinates for embedding, by default True.
        Uses eigenvalues of the distance matrix if False.
    box_size_mult : float, optional
        The box size multiplier for embedding, by default 2.0.
    random_neg_eig : bool, optional
        Whether to use random eigenvalues for embedding, by default True.
    num_zero_fail : int, optional
        The number of zero eigenvalue failures allowed, by default 1.
    prune_rms_thresh : float, optional
        The RMS threshold for pruning, by default -1.0.
    coord_map : ty.Optional[ty.Dict[int, ty.Tuple[float, float, float]]], optional
        The coordinate map for embedding, by default {}.
    force_tol : float, optional
        The force tolerance for embedding, by default 1e-3.
    ignore_smoothing_failures : bool, optional
        Whether to ignore smoothing failures, by default False.
    enforce_chirality : bool, optional
        Whether to enforce chirality, by default True.
    num_threads : int, optional
        The number of threads to use for embedding and optimization, by default 1.
    use_exp_torsion_angle_prefs : bool, optional
        Whether to use experimental torsion angle preferences, by default True.
    use_basic_knowledge : bool, optional
        Whether to use basic knowledge, by default True.
    print_exp_torsion_angles : bool, optional
        Whether to print experimental torsion angles, by default False.
    use_small_ring_torsions : bool, optional
        Whether to use small ring torsions, by default False.
    use_macrocycle_torsions : bool, optional
        Whether to use macrocycle torsions, by default False.
    et_version : int, optional
        The experimental torsion angle version, by default 1.
    optimize_confs : bool, optional
        Whether to optimize the conformers, by default True.
    max_iters_optimization : int, optional
        The maximum number of iterations for optimization, by default 200.
    mmff_variant : int, optional
        The MMFF variant to use for optimization, by default "MMFF94".
    non_bonded_thresh : float, optional
        The non-bonded threshold for optimization, by default 100.0.
    ignore_interfrag_interactions : bool, optional
        Whether to ignore inter-fragment interactions, by default True.

    Returns
    -------
    ty.List[Chem.Conformer]
        The conformers of the molecule.
    """
    # Embed conformers.
    conf_ids = AllChem.EmbedMultipleConfs(
        mol, 
        numConfs=num_confs, 
        maxAttempts=max_attempts,
        randomSeed=random_seed,
        clearConfs=clear_confs,
        useRandomCoords=use_random_coords,
        boxSizeMult=box_size_mult,
        randNegEig=random_neg_eig,
        numZeroFail=num_zero_fail,
        pruneRmsThresh=prune_rms_thresh,
        coordMap=coord_map,
        forceTol=force_tol,
        ignoreSmoothingFailures=ignore_smoothing_failures,
        enforceChirality=enforce_chirality,
        numThreads=num_threads,
        useExpTorsionAnglePrefs=use_exp_torsion_angle_prefs,
        useBasicKnowledge=use_basic_knowledge,
        printExpTorsionAngles=print_exp_torsion_angles,
        useSmallRingTorsions=use_small_ring_torsions,
        useMacrocycleTorsions=use_macrocycle_torsions,
        ETversion=et_version,
    )

    # Optimize conformers.
    if optimize_confs:
        AllChem.MMFFOptimizeMoleculeConfs(
            mol,
            numThreads=num_threads,
            maxIters=max_iters_optimization,
            mmffVariant=mmff_variant,
            nonBondedThresh=non_bonded_thresh,
            ignoreInterfragInteractions=ignore_interfrag_interactions,
        )

    # Get conformers.
    conformers = [mol.GetConformer(conf_id) for conf_id in conf_ids]

    return conformers

def get_conformer_energy(conf: Chem.Conformer) -> float:
    """
    Gets the energy of a conformer.

    Parameters
    ----------
    conf : Chem.Conformer
        The conformer to get the energy of.

    Returns
    -------
    float
        The energy of the conformer.
    """
    energy = AllChem.MMFFGetMoleculeForceField(
        conf.GetOwningMol(), 
        AllChem.MMFFGetMoleculeProperties(
            conf.GetOwningMol(), 
            mmffVariant="MMFF94"
        ),
        confId=conf.GetId(),
    ).CalcEnergy()

    return energy

def calculate_center_of_mass(coords: np.array, weights: ty.List[float]) -> np.array:
    """
    Calculates the center of mass of a set of coordinates.

    Parameters
    ----------
    coords : np.array
        The coordinates to calculate the center of mass of.
    weights : ty.List[float]
        The weights of the coordinates.

    Returns
    -------
    np.array
        The center of mass of the coordinates.
    """
    center_of_mass = np.average(coords, axis=0, weights=weights)

    return center_of_mass

def calculate_moments_of_inertia_and_axes(
    coords: np.array, 
    weights: ty.List[float]
) -> np.array:
    """
    Calculate moment of inertia tensor and principal axes.

    Parameters
    ----------
    coords : np.array
        The coordinates to calculate the moment of inertia tensor and principal axes of.
    weights : ty.List[float]
        The weights of the coordinates.
    
    Returns
    -------
    np.array
        The moment of inertia tensor.
    np.array
        The principal axes.
    """
    # Check if coords is a numpy array of shape (n, 3).
    if not isinstance(coords, np.ndarray):
        raise TypeError("coords must be a numpy array.")
    if coords.shape[1] != 3:
        raise ValueError("coords must be of shape (n, 3).")

    # Check if weights is a list of length n.
    if not isinstance(weights, list):
        raise TypeError("weights must be a list.")
    if len(weights) != coords.shape[0]:
        raise ValueError("weights must be of length n.")
    
    # Calculate center of mass.
    center_of_mass = calculate_center_of_mass(coords, weights)

    # Translate coordinates to center of mass.
    coords -= center_of_mass

    # Calculate moment of inertia tensor.
    tensor = np.zeros((3, 3))
    for i, (x, y, z) in enumerate(coords):
        tensor[0, 0] += weights[i] * (y**2 + z**2)
        tensor[1, 1] += weights[i] * (x**2 + z**2)
        tensor[2, 2] += weights[i] * (x**2 + y**2)
        tensor[0, 1] -= weights[i] * x * y
        tensor[0, 2] -= weights[i] * x * z
        tensor[1, 2] -= weights[i] * y * z
    tensor[1, 0] = tensor[0, 1]
    tensor[2, 0] = tensor[0, 2]
    tensor[2, 1] = tensor[1, 2]

    # Calculate principal axes.
    eigvals, eigvecs = eig(tensor)
    idx = eigvals.argsort()
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    return eigvals, eigvecs