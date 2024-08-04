import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import selfies
from rdkit import Chem
from scipy.stats import norm

from preprocess_utils import canonicalize_column, generate_brics, generate_fingerprint, tokenizer_factory

HERE: Path = Path(__file__).resolve().parent
DATASETS: Path = HERE.parent.parent / "datasets"



correct_structure_name: dict[str,list[str]] = {'PPFOH': ['PPFOH','PPFOH-L', 'PPFOH-H'],
                                 'P3HT': ['rr-P3HT', 'P3DT-d21','P3HT (high Mw)','P3HT_a','P3HT_b','P3HT'],
                                 'PQT-12': ['PQT12','PQT-12'],
                                 'PTHS': ['PTHS1','PTHS2','PTHS3','PTHS'],
                                 'PBTTT-C14' : ['pBTTT-C14','PBTTT-C14','PBTTT_C14_1','PBTTT_C14_2','PBTTT_C14_3', 'PBTTT_C14_4'],
                                 'PBTTT-C16' : ['PBTTT-C16','pBTTTC16',],
                                 'PFO' : ['PFO', 'PF8'],
                                 'DTVI-TVB' : ['DTVI1-TVB99', 'DTVI5-TVB95', 'DTVI10-TVB90', 'DTVI25-TVB75', 'DTVI50-TVB50', 'DTVI-TVB'],
                                 'DPPDTT' : ['DPPDTT1', 'DPPDTT2', 'DPPDTT3', 'DPPDTT'],
                                 'PII-2T' : ['PII-2T', 'High MW PII-2T'],
                                 'MEH-PPV' : ['MEH-PPV', 'MEH-PPV-100', 'MEH-PPV-30', 'MEH-PPV-70',],
                                 'PFO' : ['PFO', 'PF8', 'PFO-d34'],
                                 'PFT3' : ['S_PFT', 'PFT3'],
                                 'P(NDI2OD-T2)': ['P(NDI2OD-T2)', 'PNDI-C0', 'NDI-C0', 'NDI-2T-2OD'],
                                 }


def generate_ECFP_fingerprint(mol, radius: int = 3, nbits: int = 1024,count_vector:bool=True) -> np.array:
    """
    Generate ECFP fingerprint.

    Args:
        mol: RDKit Mol object
        radius: Fingerprint radius
        nbits: Number of bits in fingerprint

    Returns:
        ECFP fingerprint as numpy array
    """
    if count_vector:
      fingerprint: np.array = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits, includeChirality=False,countSimulation=False).GetCountFingerprintAsNumPy(mol)
    else:
      fingerprint: np.array = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits, includeChirality=False,countSimulation=False).GetFingerprintAsNumPy(mol)
    
    return fingerprint


def canonicalize_column(data = pd.DataFrame,smiles_column: str='SMILES') -> pd.DataFrame:
        """Canonicalize SMILES."""
        data[smiles_column]=data[smiles_column].apply(lambda smiles: CanonSmiles(smiles))


class ECFP_Processor:

    def __init__(self, smile_source: pd.DataFrame,
                 oligomer_represenation:str='SMILES') -> None:
        self.smile_source = smile_source.copy()
        self.oligomer_represenation =oligomer_represenation
        canonicalize_column(self.smile_source)
        self.all_mols: pd.Series = self.smile_source[self.oligomer_represenation].map(lambda smiles: MolFromSmiles(smiles))



    def assign_ECFP(self, radius: int = 3, nbits: int = 1024,count_vector:bool=True) -> None:
        """
        Assigns ECFP fingerprints to the dataset.
        """
        self.smile_source[f"CV_{count_vector}_ECFP_{2 * radius}_{nbits}_{self.oligomer_represenation}"] = self.all_mols.map(
                          lambda mol: generate_ECFP_fingerprint(mol, radius, nbits, count_vector=count_vector))
        print(f"Done assigning CV_{count_vector}_ECFP_{2 * radius} fingerprints with {nbits} bits on {self.oligomer_represenation}.")


    def main_ecfp(self, fp_radii: list[int], fp_bits: list[int],count_vector:bool=True) -> pd.DataFrame:

        # # self.dataset[f"{material} BigSMILES"] = self.assign_bigsmiles(material, self.dataset[f"{material} SMILES"])
        # self.dataset[f"{material} BRICS"] = self.assign_brics(self.dataset[f"{material} Mol"])
        for r, b in zip(fp_radii,fp_bits):
            self.assign_ECFP(radius=r, nbits=b, count_vector=count_vector)
        return self.smile_source
      

# example:
# fp_rad: list[int] = [3,4]
# fp_bi: list[int] = [512,1024]
# df_clean = ECFP_Processor(structural_features_test,oligomer_represenation='SMILES').main_ecfp(fp_rad, fp_bi,count_vector=True)
# df_clean


# maccs is here!
class MACCS_Processor:
    def __init__(self, smile_source: pd.DataFrame,
                 oligomer_represenation:str='SMILES') -> None:
        self.smile_source = smile_source.copy()
        self.oligomer_represenation =oligomer_represenation
        canonicalize_column(self.smile_source)
        self.all_mols: pd.Series = self.smile_source[self.oligomer_represenation].map(lambda smiles: MolFromSmiles(smiles))


    def assign_MACCS(self):
        self.smile_source[f"MACCS_{self.oligomer_represenation}"] = self.all_mols.map(
            lambda mol: MACCSkeys.GenMACCSKeys(mol))
        print(f"Done assigning MACCS_{self.oligomer_represenation}_fingerprints")
        
        return self.smile_source

# example:
# maccs = MACCS_Processor(structural_features_test,oligomer_represenation='SMILES').assign_MACCS()

# unrolling maccs:
# def compute_MACCS(self):
#     MACCS_list = []
#     header = ['bit' + str(i) for i in range(167)]
#     for mol in self.all_mols:
#         ds = list(MACCSkeys.GenMACCSKeys(mol)
#         MACCS_list.append(ds)
#     df = pd.DataFrame(MACCS_list,columns=header)

#     return df




class MordredCalculator:
    def __init__(self, smile_source: pd.DataFrame, oligomer_represenation:str ='SMILES') -> None:
        self.smile_source = smile_source
        self.oligomer_represenation = oligomer_represenation
        canonicalize_column(self.smile_source)
        self.all_mols: pd.Series = self.smile_source[self.oligomer_represenation].map(lambda smiles: MolFromSmiles(smiles))

        
    def assign_Mordred(self):
        self.smile_source[f"Mordred_{self.oligomer_represenation}"] = self.all_mols.map(
                    lambda mol: get_mordred_dict(mol))
        
        print(f"Done assigning Mordred_{self.oligomer_represenation}_fingerprints")
                
        return self.smile_source

# example:
# mordred_calc: MordredCalculator = MordredCalculator(structural_features.iloc[:3])
# mordred_calc.assign_Mordred()

# To-Do: change the below:

def pre_main(fp_radii: list[int], fp_bits: list[int], solv_props_as_nan: bool):
    min_dir: Path = DATASETS / "Min_2020_n558"
    min_raw_dir: Path = min_dir / "raw"

    # Whether to treated missing solvent properties as NaN
    if solv_props_as_nan:
        solv_prop_fname = "solvent properties_nan.csv"
        dataset_fname = "cleaned_dataset_nans.pkl"
    else:
        solv_prop_fname = "solvent properties.csv"
        dataset_fname = "cleaned_dataset.pkl"

    # Import raw dataset downloaded from Google Drive
    raw_dataset_file = min_raw_dir / "raw dataset.csv"
    raw_dataset: pd.DataFrame = pd.read_csv(raw_dataset_file, index_col="ref")

    # Get list of duplicated Donor and Acceptor labels
    duplicated_labels_file = min_raw_dir / "duplicate labels.csv"
    duplicated_labels: pd.DataFrame = pd.read_csv(duplicated_labels_file, index_col="Name0")

    # Get selected properties
    selected_properties_file = min_dir / "selected_properties.json"
    with selected_properties_file.open("r") as f:
        selected_properties = json.load(f)

    # Get solvent and solvent additive properties
    solvent_properties_file = min_raw_dir / solv_prop_fname
    solvent_properties: pd.DataFrame = pd.read_csv(solvent_properties_file, index_col="Name")
    selected_solvent_properties: list[str] = selected_properties["solvent"]

    # Get interlayer properties
    interlayer_properties_file = min_raw_dir / "interlayer properties.csv"
    interlayer_properties: pd.DataFrame = pd.read_csv(interlayer_properties_file, index_col="Name")
    selected_interlayer_properties: list[str] = selected_properties["interlayer"]

    # Clean features in the dataset
    dataset: pd.DataFrame = FeatureProcessor(raw_dataset,
                                             duplicated_labels,
                                             solvent_properties,
                                             interlayer_properties,
                                             solvent_descriptors=selected_solvent_properties
                                             ).main()

    # Load cleaned donor and acceptor structures
    donor_structures_file = min_dir / "cleaned donors.csv"
    donor_structures: pd.DataFrame = pd.read_csv(donor_structures_file)
    acceptor_structures_file = min_dir / "cleaned acceptors.csv"
    acceptor_structures: pd.DataFrame = pd.read_csv(acceptor_structures_file)

    # Add structural representations to the dataset
    dataset: pd.DataFrame = StructureProcessor(dataset, donor_structures=donor_structures,
                                               acceptor_structures=acceptor_structures).main(fp_radii, fp_bits)

    # Get datatypes of categorical features
    feature_types_file = min_dir / "feature_types.json"
    with feature_types_file.open("r") as f:
        feature_types: dict = json.load(f)
    dataset: pd.DataFrame = assign_datatypes(dataset, feature_types)

    # Specify paths for saving
    dataset_csv = min_dir / "cleaned_dataset.csv"
    dataset_pkl = min_dir / dataset_fname

    # Save the dataset
    dataset.to_pickle(dataset_pkl)
    readable: pd.DataFrame = get_readable_only(dataset)
    readable.to_csv(dataset_csv)


if __name__ == "__main__":
    fp_radii: list[int] = [3, 4, 5, 6]
    fp_bits: list[int] = [512, 1024, 2048, 4096]
    for solv_props_as_nan in [True, False]:
        print(f"Running with solv_props_as_nan={solv_props_as_nan}")
        pre_main(fp_radii=fp_radii, fp_bits=fp_bits, solv_props_as_nan=solv_props_as_nan)
