#!/usr/bin/env python3
"""
pmi_analysis.py
===============
Author:         David Meijer
Licence:        MIT License
Description:    Script to create PMI plot of set of molecules.
Note:           See https://pubs.acs.org/doi/full/10.1021/ci025599w# for more
                information on the PMI plot.
"""
import argparse 
import typing as ty

from rdkit import Chem
from tqdm import tqdm 
import plotly.express as px
import plotly.graph_objects as go

from retromol_core.chem_utils import smiles_to_mol, fragment_mol, neutralize_mol
from retromol_core.geom import get_conformers, calculate_moments_of_inertia_and_exes

def cli() -> argparse.Namespace:
    """
    Command line interface.

    Returns
    -------
    argparse.Namespace
        The parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=str, required=True, 
        help="Input tsv file as 'name\tsmiles\n'."
    )
    parser.add_argument(
        "-ih", "--input-has-header", action="store_true", 
        help="Whether the input file has a header."
    )
    parser.add_argument(
        "-o", "--out", type=str, required=True, 
        help="Output html file."
    )
    return parser.parse_args()

def parse_input_file(
    input_file: str, 
    input_file_has_header: bool
) -> ty.List[ty.Tuple[str, str]]:
    """
    Parse input file.
    
    Parameters
    ----------
    input_file : str
        The input file.
    input_file_has_header : bool
        Whether the input file has a header.
    
    Returns
    -------
    ty.List[ty.Tuple[str, str]]
        The list of (name, smiles) tuples.
    """
    with open(input_file, "r") as file_open:
        # Skip header.
        if input_file_has_header:
            file_open.readline()

        lines = file_open.readlines()

    return [tuple(line.strip().split("\t")) for line in lines]

def parse_smiles(smiles: str) -> Chem.Mol:
    """
    Parse SMILES string.

    Parameters
    ----------
    smiles : str
        The SMILES string.

    Returns
    -------
    Chem.Mol
        The RDKit molecule.
    """
    # Parse SMILES string.
    mol = smiles_to_mol(smiles)

    # Neutralize molecule.
    mol = neutralize_mol(mol)

    # Fragment molecule and pick largest fragment.
    mols = fragment_mol(mol, sort_ascending=True)
    mol = mols[-1]

    # Reset flags.
    Chem.SanitizeMol(mol)

    # Add hydrogens.
    mol = Chem.AddHs(mol)

    return mol

def main() -> None:
    """
    Driver function.
    """
    # Parse command line arguments.
    args = cli()

    # Parse input file, get list of (name, smiles) tuples.
    records = parse_input_file(args.input, args.input_has_header)

    # Calculate principal moments of inertia.
    ratios = []
    for name, smiles in tqdm(records, leave=False):
        try:
            # Parse input SMILES string into RDKit molecule.
            mol = parse_smiles(smiles)

            # Calculate conformers.
            confs = get_conformers(
                mol, 
                num_confs=1, 
                use_random_coords=True, 
                optimize_confs=False
            )
            # Pick first and only conformer.
            conf = confs[0]
            
            # Get positions and weights. 
            coords = conf.GetPositions()
            weights = [atom.GetMass() for atom in mol.GetAtoms()]

            # Calculate principal moments of inertia and principal axes.
            moments, _ = calculate_moments_of_inertia_and_exes(coords, weights)

            # Get normalized PMI ratios.
            i13 = moments[0] / moments[2]
            i23 = moments[1] / moments[2]

            # Append to list.
            ratios.append((name, float(i13), float(i23)))
        
        except Exception as e:
            print(f"Error: {e}")
            print(f"Skipping {name}.")
            continue

    # Plot PMI ratios.
    labels = [ratio[0] for ratio in ratios]
    x = [ratio[1] for ratio in ratios]
    y = [ratio[2] for ratio in ratios]
    fig = px.scatter(x=x, y=y, hover_name=labels)
    fig.update_traces(marker=dict(symbol="triangle-up", size=15, color="blue"))

    triangle_x = [0.0, 0.5, 1.0, 0.0]
    triangle_y = [1.0, 0.0, 1.0, 1.0]
    fig.add_trace(go.Scatter(
        x=triangle_x, 
        y=triangle_y, 
        line=dict(color="black", width=2, dash="dash"),
        mode="lines",
        name="Shape envelope"    
    ))

    fig.add_annotation(text="Rod", x=-0.025, y=1.025, showarrow=False, font=dict(size=14, family="serif"))
    fig.add_annotation(text="Disc", x=0.5, y=-0.025, showarrow=False, font=dict(size=14, family="serif"))
    fig.add_annotation(text="Sphere", x=1.025, y=1.025, showarrow=False, font=dict(size=14, family="serif"))

    fig.update_xaxes(
        title_text="$I_1/I_3$", 
        scaleanchor="y", 
        scaleratio=1, 
        tickfont=dict(size=16, family="serif"),
        linecolor="black",
    )
    fig.update_yaxes(
        title_text="$I_2/I_3$", 
        scaleanchor="x", 
        scaleratio=1, 
        tickfont=dict(size=16, family="serif"),
        linecolor="black",
    )
    fig.update_layout(
        font=dict(size=18, family="serif"), 
        title_text="Normalized principal moments of inertia",
        paper_bgcolor="white", 
        plot_bgcolor="white",
    )
    fig.write_html(
        args.out, 
        include_mathjax="cdn" # To save LaTeX annotations appropriately.
    )

    exit(0)

if __name__ == "__main__":
    main()