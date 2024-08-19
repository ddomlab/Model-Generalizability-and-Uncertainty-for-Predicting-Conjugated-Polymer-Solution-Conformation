import json
from pathlib import Path
# from typing import Optional

import numpy as np
import pandas as pd
# from rdkit import Chem
# from scipy.stats import norm


from rdkit.Chem import Draw, MolFromSmiles, CanonSmiles, MolToSmiles
from rdkit.Chem import Mol
from rdkit.Chem import rdFingerprintGenerator
import mordred
import mordred.descriptors
from mordred import Calculator
from rdkit.Chem import MACCSkeys


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
                 oligomer_represenation:str) -> None:
        self.smile_source = smile_source.copy()
        self.oligomer_represenation =oligomer_represenation
        canonicalize_column(self.smile_source,smiles_column=self.oligomer_represenation)
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
        canonicalize_column(self.smile_source,smiles_column=self.oligomer_represenation)
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



def get_mordred_dict(mol: Mol) -> dict[str, float]:
    """
    Get Mordred descriptors for a given molecule.

    Args:
        mol: RDKit molecule

    Returns:
        Mordred descriptors as dictionary
    """
    calc: Calculator = Calculator(mordred.descriptors, ignore_3D=True)
    descriptors: dict[str, float] = calc(mol).asdict()
    return descriptors


class MordredCalculator:
    def __init__(self, smile_source: pd.DataFrame, oligomer_represenation:str ='SMILES') -> None:
        self.smile_source = smile_source
        self.oligomer_represenation = oligomer_represenation
        canonicalize_column(self.smile_source,smiles_column=self.oligomer_represenation)
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

def pre_main(fp_radii: list[int], fp_bits: list[int], count_v:list[bool]):

    min_dir: Path = DATASETS / 'raw'

    pu_file = min_dir / "pu_processed.csv"
    transferred_dir: Path = DATASETS / 'fingerprint'
    pu_dataset: pd.DataFrame = pd.read_csv(pu_file)
    pu_used_dir = min_dir / 'pu_columns_used.json'
    with open(pu_used_dir, 'r') as file:
        pu_used: list[str] = json.load(file)

    # for test =>  fp_dataset: pd.DataFrame = pu_dataset.iloc[:3]

    fp_dataset: pd.DataFrame = pu_dataset
    for polymer_unit in pu_used:
        fp_dataset: pd.DataFrame = MordredCalculator(fp_dataset, oligomer_represenation=polymer_unit).assign_Mordred()
        fp_dataset: pd.DataFrame = MACCS_Processor(fp_dataset, oligomer_represenation=polymer_unit).assign_MACCS()
        for c_b in count_v:
            fp_dataset = pd.DataFrame = ECFP_Processor(fp_dataset, oligomer_represenation=polymer_unit).main_ecfp(fp_radii, fp_bits, count_vector= c_b)

    # save fie
    fp_dataset.to_csv(transferred_dir/'structural_features.csv', index=False)
    fp_dataset.to_pickle(transferred_dir/'structural_features.pkl')




if __name__ == "__main__":
    fp_radii: list[int] = [3, 4, 5, 6]
    fp_bits: list[int] = [512, 1024, 2048, 4096]
    count_v = [True,False]
    pre_main(fp_radii=fp_radii, fp_bits=fp_bits, count_v=count_v)
