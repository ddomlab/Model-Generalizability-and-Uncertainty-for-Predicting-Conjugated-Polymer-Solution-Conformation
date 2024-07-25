# ATTN: When done, check the following:
#  - All rows that have interlayer have interlayer descriptors
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw, Mol

sys.path.append("../cleaning")
from cleaning_utils import find_identical_molecules

HERE: Path = Path(__file__).resolve().parent
DATASETS: Path = HERE.parent.parent / "datasets"


def test_tanimoto_similarity(dataset: pd.DataFrame) -> None:
    radius: int = 5
    nbits: int = 512
    for material in ["Acceptor", "Donor"]:
        print(f"Checking {material}s by Tanimoto similarity...")
        overlaps: int = find_identical_molecules(dataset[f"{material} SMILES"], radius=radius, bits=nbits)
        assert overlaps == 0, f"Found {overlaps} identical {material}s by Tanimoto similarity"


def test_has_smiles(dataset: pd.DataFrame) -> None:
    for material in ["Acceptor", "Donor"]:
        print(f"Checking {material}s for missing SMILES...")
        no_smiles = dataset[material][dataset[f"{material} SMILES"].isna()].unique()
        assert len(no_smiles) == 0, f"Found {material}s without SMILES: \n{no_smiles}"


def check_non_null_values(df: pd.DataFrame, column_name: str) -> tuple[bool, pd.DataFrame]:
    """
    Check if all rows in the specified column of a DataFrame have non-null values.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column to check.

    Returns:
        bool: True if all rows have non-null values, False otherwise.
    """
    column: pd.Series = df[column_name]
    null_rows = df[column.isnull()]
    return column.notnull().all(), null_rows


def test_has_solvent_descriptors(dataset: pd.DataFrame) -> None:
    for solvent in ["solvent", "solvent additive"]:
        print(f"Checking {solvent} descriptors...")
        filtered_dataset: pd.DataFrame = dataset.dropna(subset=[solvent])
        has_descriptors, null_rows = check_non_null_values(filtered_dataset, f"{solvent} descriptors")
        assert has_descriptors, f"Found {solvent}s without descriptors:\n{null_rows}"


def draw_molecule_with_label(smiles: str, label: str):
    mol: Mol = Chem.MolFromSmiles(smiles)
    # Generate the molecule's image
    img = Draw.MolToImage(mol)

    # Show the image with the molecule's index as the title
    plt.imshow(img)
    plt.axis('off')
    plt.title(label)
    plt.show()
    plt.close()


if __name__ == "__main__":
    min_dir: Path = DATASETS / "Min_2020_n558"
    dataset_file: Path = min_dir / "cleaned_dataset.pkl"
    dataset: pd.DataFrame = pd.read_pickle(dataset_file)

    test_tanimoto_similarity(dataset)
    # test_has_smiles(dataset)
    # test_has_solvent_descriptors(dataset)
    # last_label: str = "IFT-ECA"
    # test_correct_structures(dataset, last_label)
