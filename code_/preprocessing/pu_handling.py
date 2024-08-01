import numpy as np
from typing import Callable, Optional, Union
from types import NoneType

import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw, MolFromSmiles, CanonSmiles, MolToSmiles
from rdkit.Chem import Mol

HERE: Path = Path(__file__).resolve().parent
DATASETS: Path = HERE.parent.parent / "datasets"



def monomer_propagation(monomer_SMILES,n_unit, termination_oo:bool = True,Regioregularity:bool=True):
    new_monomer_smiles = monomer_SMILES
    for i in range(n_unit-1):
        new_monomer_smiles = attach_molecule(new_monomer_smiles,monomer_SMILES,Regioregularity=Regioregularity)
    if termination_oo:
        new_monomer_smiles = termination_with_asterisk(new_monomer_smiles,['[Zr]', '[Pd]'])
    # img = Draw.MolToImage(MolFromSmiles(new_monomer_smiles), size=(1000, 1000))
    # display(img)
    return new_monomer_smiles


def replace_asterisk(smiles, replacement:list[str]):
  mol = MolFromSmiles(smiles)
  for rep in replacement:
      mol = Chem.ReplaceSubstructs(mol, Chem.MolFromSmiles('*'), Chem.MolFromSmiles(rep))[0]
  Chem.SanitizeMol(mol)
  smiles = MolToSmiles(mol)
  return smiles


def termination_with_asterisk(smiles, replacement:list[str]):
    mol = MolFromSmiles(smiles)
    for rep in replacement:
      mol = Chem.ReplaceSubstructs(mol,Chem.MolFromSmiles(rep), Chem.MolFromSmiles('*'), replaceAll=True)[0]
    Chem.SanitizeMol(mol)
    smiles = MolToSmiles(mol)
    return smiles


def atomic_number_calculator(atom_str):
  mol = Chem.MolFromSmiles(atom_str)
  atom = mol.GetAtomWithIdx(0)
  atomic_number = atom.GetAtomicNum()
  return int(atomic_number)


def replace_attachment_point(mol, dumi_smiles,replacement_smiles):
    dumi_atn = atomic_number_calculator(dumi_smiles)
    replacement_atn = atomic_number_calculator(replacement_smiles)
    mol_r = Chem.RWMol(mol)
    for atom in mol_r.GetAtoms():
        if atom.GetAtomicNum() == dumi_atn:
                mol_r.ReplaceAtom(atom.GetIdx(), Chem.Atom(replacement_atn))
    return mol_r


def close_ring(smiles):
  # print(smiles)
  bond_type=get_bond_type(smiles)
  # print(bond_type)
  editeable_mol = Chem.RWMol(MolFromSmiles(smiles))
  Pd_idx = [atom.GetIdx() for atom in editeable_mol.GetAtoms() if atom.GetSymbol() == '*']
  # print(Pd_idx)
  # Zr_idx = [atom.GetIdx() for atom in editeable_mol.GetAtoms() if atom.GetSymbol() == '*'][0]
  if bond_type == {2}:
    editeable_mol.AddBond(Pd_idx[0], Pd_idx[1], Chem.BondType.DOUBLE)
  else:
    editeable_mol.AddBond(Pd_idx[0], Pd_idx[1], Chem.BondType.SINGLE)
  
  final_mol = editeable_mol.GetMol()
  Chem.SanitizeMol(final_mol)
  final_smiles = MolToSmiles(final_mol)
  if bond_type == {2}:
    print('yes')
    final_smiles=final_smiles.replace('=*=*','')
  else:
    final_smiles = final_smiles.replace('**','')                    
  img = Draw.MolToImage(MolFromSmiles(final_smiles), size=(1000, 1000))
  display(img)
  return final_smiles


def get_bond_type(smiles):
    mol = MolFromSmiles(smiles)
    star_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == '*']
    bond_set = set()
    for star_atom_idx in star_atoms:
      star_atom = mol.GetAtomWithIdx(star_atom_idx)

      # Get bonds connected to this atom
      bonds = star_atom.GetBonds()

      for star_atom_idx in star_atoms:
          star_atom = mol.GetAtomWithIdx(star_atom_idx)

          # Get bonds connected to this atom
          bonds = star_atom.GetBonds()
          bond_type = [bond.GetBondTypeAsDouble() for bond in bonds]
          bond_set.add(bond_type[0])

    return bond_set


def attach_molecule(smiles1,smiles2, Regioregularity:bool=True):
    bond_type = get_bond_type(smiles2)
    # print(bond_type)
    # print(smiles1)
    new_smiles1 = replace_asterisk(smiles1, ['[Zr]', '[Pd]'])
    new_smiles2 = replace_asterisk(smiles2, ['[Zr]', '[Pd]'])
    # Convert SMILES to RDKit molecule objects
    mol1 = Chem.MolFromSmiles(new_smiles1)
    mol2 = Chem.MolFromSmiles(new_smiles2)
    # img = Draw.MolToImage(mol2, size=(500, 500))
    # display(img)
    # Replace Br and Cl with a wildcard (*) in the first molecule
    # mol1_no_br = Chem.ReplaceSubstructs(mol1, Chem.MolFromSmiles('[Zr]'), Chem.MolFromSmiles('*'), replaceAll=True)[0]
    # mol2_no_cl = Chem.ReplaceSubstructs(mol2, Chem.MolFromSmiles('[Pd]'), Chem.MolFromSmiles('*'), replaceAll=True)[0]
    # Combine the two molecules
    combined_mol = Chem.CombineMols(mol1, mol2)
    # Create an editable molecule object from the combined molecule
    editable_combined_mol = Chem.RWMol(combined_mol)
    # Find the indices of the wildcard atoms (*) to form a bond between them
    pd_idx = [atom.GetIdx() for atom in combined_mol.GetAtoms() if atom.GetSymbol() == 'Pd']
    zr_idx = [atom.GetIdx() for atom in combined_mol.GetAtoms() if atom.GetSymbol() == 'Zr']
    # Add a single bond between the two wildcard atoms

    if Regioregularity:
        if bond_type== {2}:
            editable_combined_mol.AddBond(pd_idx[0], zr_idx[1], Chem.BondType.DOUBLE)
        else: 
            editable_combined_mol.AddBond(pd_idx[0], zr_idx[1], Chem.BondType.SINGLE)

          # print(smiles1)
    else:
        if len(pd_idx)==2:
          editable_combined_mol.AddBond(zr_idx[0], zr_idx[1], Chem.BondType.SINGLE)
        elif len(pd_idx)==3:
          editable_combined_mol.AddBond(pd_idx[1], pd_idx[2], Chem.BondType.SINGLE)

    final_mol = editable_combined_mol.GetMol()
    Chem.SanitizeMol(final_mol)
    #final_mol = replace_attachment_point(final_mol, '*', 'C')        if you wanna include a bond between two monomers
    final_smiles = Chem.MolToSmiles(final_mol)
    if Regioregularity:

        if bond_type== {2}:
            for i in ['=[Pd]=[Zr]','=[Zr]=[Pd]']:
              if i in final_smiles:
                final_smiles=final_smiles.replace(i,'')

        else:
            for i in ['[Pd][Zr]','[Zr][Pd]']:
              if i in final_smiles:
                final_smiles=final_smiles.replace(i,'')
    else:
      for i in ['[Pd][Pd]','[Zr][Zr]']:
          if i in final_smiles:
            final_smiles=final_smiles.replace(i,'')
    return final_smiles


#generate RU, dimer, trimer

def main():
    # Load cleaned donor and acceptor structures
    dataset_dir: Path = DATASETS / "raw_SMILES"
    raw_smiles: pd.DataFrame = pd.read_csv(dataset_dir)


    # Load dataset
    dataset_pkl = dataset_dir / "cleaned_dataset.pkl"
    dataset: pd.DataFrame = pd.read_pickle(dataset_pkl)

    # Save mordred descriptor IDs
    mordred_json = dataset_dir / "mordred_descriptors.json"

    # Save dataset
    mordred_pkl = dataset_dir / "cleaned_dataset_mordred.pkl"

    run(donor_structures, acceptor_structures, dataset, mordred_json, mordred_pkl)
