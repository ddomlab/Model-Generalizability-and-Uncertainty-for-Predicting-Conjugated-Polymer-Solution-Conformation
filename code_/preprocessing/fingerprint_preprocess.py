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

# Parallelization
import time
from pandarallel import pandarallel
from argparse import ArgumentParser


HERE: Path = Path(__file__).resolve().parent
DATASETS: Path = HERE.parent.parent / "datasets"



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


def canonicalize_dataset_parallel(data=pd.DataFrame) -> pd.DataFrame:
    """Canonicalize SMILES."""

    data.loc[:, data.columns != 'Name'] = data.loc[:, data.columns != 'Name'].parallel_applymap(
        lambda smiles: CanonSmiles(smiles))

class ECFP_Processor:

    def __init__(self, smile_source: pd.DataFrame,
                 oligomer_represenation:str) -> None:
        self.smile_source = smile_source.copy()
        self.oligomer_represenation =oligomer_represenation
        self.all_mols: pd.Series = self.smile_source[self.oligomer_represenation].parallel_map(lambda smiles: MolFromSmiles(smiles))



    def assign_ECFP(self, radius: int = 3, nbits: int = 1024,count_vector:bool=True) -> None:
        """
        Assigns ECFP fingerprints to the dataset.
        """
        if count_vector:
            vector_type = 'count'
        else:
            vector_type = 'binary'

        self.smile_source[f"{self.oligomer_represenation.split()[0]}_ECFP{2 * radius}_{vector_type}_{nbits}bits"] = self.all_mols.parallel_map(
                          lambda mol: generate_ECFP_fingerprint(mol, radius, nbits, count_vector=count_vector))
        print(f"Done with generating {self.oligomer_represenation.split()[0]}_ECFP{2 * radius}_{vector_type}_{nbits}bits")


    def main_ecfp(self, fp_radii: list[int], fp_bits: list[int],count_vector:bool=True) -> pd.DataFrame:

        # # self.dataset[f"{material} BigSMILES"] = self.assign_bigsmiles(material, self.dataset[f"{material} SMILES"])
        # self.dataset[f"{material} BRICS"] = self.assign_brics(self.dataset[f"{material} Mol"])
        for r, b in zip(fp_radii,fp_bits):
            self.assign_ECFP(radius=r, nbits=b, count_vector=count_vector)
        return self.smile_source
      

class MACCS_Processor:
    def __init__(self, smile_source: pd.DataFrame,
                 oligomer_represenation:str='SMILES') -> None:
        self.smile_source = smile_source.copy()
        self.oligomer_represenation =oligomer_represenation
        self.all_mols: pd.Series = self.smile_source[self.oligomer_represenation].parallel_map(lambda smiles: MolFromSmiles(smiles))


    def assign_MACCS(self):
        self.smile_source[f"{self.oligomer_represenation.split()[0]}_MACCS"] = self.all_mols.parallel_map(
            lambda mol: MACCSkeys.GenMACCSKeys(mol))
        print(f"Done assigning {self.oligomer_represenation.split()[0]}_MACCS representation")
        
        return self.smile_source


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
        self.all_mols: pd.Series = self.smile_source[self.oligomer_represenation].parallel_map(lambda smiles: MolFromSmiles(smiles))


    def assign_Mordred(self):
        
        descriptors: pd.Series = self.all_mols.parallel_map(lambda mol: get_mordred_dict(mol))
        #unpacking the descriptors
        mordred_descriptors: pd.DataFrame = pd.DataFrame.from_records(descriptors, index=self.all_mols.index)
        # Remove any columns with calculation errors
        mordred_descriptors = mordred_descriptors.infer_objects()
        mordred_descriptors = mordred_descriptors.select_dtypes(exclude=["object"])
        # Remove any columns with nan values
        mordred_descriptors.dropna(axis=1, how='any', inplace=True)
        # Remove any columns with zero variance
        descriptor_variances: pd.Series = mordred_descriptors.var(numeric_only=True)
        variance_mask: pd.Series = descriptor_variances.eq(0)
        zero_variance: pd.Series = variance_mask[variance_mask == True]
        invariant_descriptors: list[str] = zero_variance.index.to_list()
        mordred_descriptors: pd.DataFrame = mordred_descriptors.drop(invariant_descriptors, axis=1)
        print("Done generating Mordred descriptors.")
        self.smile_source[f"{self.oligomer_represenation.split()[0]}_Mordred"] = mordred_descriptors.to_dict(orient='records')    
        print(f"Done assigning {self.oligomer_represenation.split()[0]}_Mordred representation")
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
    
    canonicalize_dataset_parallel(pu_dataset)
    fp_dataset: pd.DataFrame = pu_dataset.copy()
    for polymer_unit in pu_used:
        fp_dataset: pd.DataFrame = MordredCalculator(fp_dataset, oligomer_represenation=polymer_unit).assign_Mordred()
        fp_dataset: pd.DataFrame = MACCS_Processor(fp_dataset, oligomer_represenation=polymer_unit).assign_MACCS()
        for c_b in count_v:
            fp_dataset = pd.DataFrame = ECFP_Processor(fp_dataset, oligomer_represenation=polymer_unit).main_ecfp(fp_radii, fp_bits, count_vector= c_b)

    # save fie
    fp_dataset.to_csv(transferred_dir/'structural_features.csv', index=False)
    fp_dataset.to_pickle(transferred_dir/'structural_features.pkl')




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '-- nb_workers',
        action='store',
        default=1,
        type=int,
        help= 'insert number of cores'
    )
    
    FLAGS = parser.parse_args()
    pandarallel.initialize(nb_workers=FLAGS.nb_workers)
    
    fp_radii: list[int] = [3, 4, 5, 6]
    fp_bits: list[int] = [512, 1024, 2048, 4096]
    count_v = [True,False]
    start_time = time.time()
    pre_main(fp_radii=fp_radii, fp_bits=fp_bits, count_v=count_v)
    end_time = time.time()
    runing_rime = end_time-start_time
    print(f"runing time = {runing_rime}")
