import json
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Mol

HERE: Path = Path(__file__).resolve().parent
DATASETS: Path = HERE.parent.parent / "datasets"



def man_main():
    raw_data = DATASETS / "Min_2020_n558" / "raw"
    r_groups_file = raw_data.parent / "cleaned R groups.csv"
    r_groups = pd.read_csv(r_groups_file)

    # for material in ["Donor", "Acceptor"]:
    for material in ["Acceptor"]:
        master_file = raw_data / f"min_{material.lower()}s_smiles_master_EDITED.csv"
        master_df = pd.read_csv(master_file)
        reference_file = raw_data / f"reference {material.lower()}s.csv"
        reference_df = pd.read_csv(reference_file)

        clean_df: pd.DataFrame = clean_structures(material, master_df, reference_df)
        clean_smiles_df: pd.DataFrame = replace_arbitrary_with_sidechain(
            replace_r_with_arbitrary(clean_df, r_groups),
            r_groups
        )

        clean_file = raw_data.parent / f"cleaned {material.lower()}s.csv"
        clean_smiles_df.to_csv(clean_file, index=False)


if __name__ == "__main__":
    man_main()
