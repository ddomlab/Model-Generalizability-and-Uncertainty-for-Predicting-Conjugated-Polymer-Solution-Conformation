import numpy as np
from typing import Callable, Optional, Union
from typing import List, Union, Dict
from types import NoneType

import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw, MolFromSmiles, CanonSmiles, MolToSmiles
from rdkit.Chem import Mol
import pandas as pd

from pathlib import Path

HERE: Path = Path(__file__).resolve().parent
DATASETS: Path = HERE.parent.parent / "datasets"/'raw'


def monomer_propagation(monomer_SMILES,n_unit, termination_oo:bool = True,Regioregularity:bool=True):
    new_monomer_smiles = monomer_SMILES
    for i in range(n_unit-1):
        new_monomer_smiles,elements = attach_molecule(new_monomer_smiles,monomer_SMILES,Regioregularity=Regioregularity)
    if termination_oo:
        new_monomer_smiles = termination_with_asterisk(new_monomer_smiles,elements)
    # img = Draw.MolToImage(MolFromSmiles(new_monomer_smiles), size=(1000, 1000))
    # display(img)
    return new_monomer_smiles


def replace_asterisk(smiles, replacement):
  mol = MolFromSmiles(smiles)
  for rep in replacement:
      mol = Chem.ReplaceSubstructs(mol, Chem.MolFromSmiles('*'), Chem.MolFromSmiles(rep))[0]
  Chem.SanitizeMol(mol)
  smiles = MolToSmiles(mol)
  return smiles

def termination_with_asterisk(smiles, replacement):
    for rep in replacement:
      smiles = smiles.replace(rep,'*')
    return smiles

# print(replace_asterisk('*c1cc(CCCCCCCCCCCC)c(*)s1', ['[Zr]', '[Pd]']))

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
    bond_type, idx =get_bond_type(smiles)
    editeable_mol = Chem.RWMol(MolFromSmiles(smiles))
    if len(bond_type)==1:
        # normal_ bond
        # print(idx)
        # Zr_idx = [atom.GetIdx() for atom in editeable_mol.GetAtoms() if atom.GetSymbol() == '*'][0]
        if bond_type == {2}:
          final_smiles= closing(editeable_mol, Chem.BondType.DOUBLE)
        else:
          final_smiles= closing(editeable_mol, Chem.BondType.SINGLE)
        final_smiles2 = final_smiles
        # final_smiles=final_smiles.replace('=*=*','')
        # final_smiles2 = final_smiles.replace('**','')

    else:
        editeable_mol.AddBond(idx['DOUBLE'][0], idx['DOUBLE'][1], Chem.BondType.DOUBLE)
        editeable_mol.AddBond(idx['SINGLE'][0], idx['SINGLE'][1], Chem.BondType.SINGLE)

        final_mol = editeable_mol.GetMol()
        final_smiles = MolToSmiles(final_mol)
        Chem.SanitizeMol(final_mol)
        final_smiles = final_smiles.replace('=*=*','')
        final_smiles2 = final_smiles.replace('**','')

    img = Draw.MolToImage(MolFromSmiles(final_smiles2), size=(1000, 1000))
    display(img)
    return final_smiles2


def attach_molecule(smiles1,smiles2, Regioregularity:bool=True):
    bond_type,_ = get_bond_type(smiles2)

    if len(bond_type)==1:
          # this is normal polymer
          elements = ['[Zr]', '[Pd]']
          new_smiles1 = replace_asterisk(smiles1, elements)
          new_smiles2 = replace_asterisk(smiles2, elements)
          mol1 = Chem.MolFromSmiles(new_smiles1)
          mol2 = Chem.MolFromSmiles(new_smiles2)
          # img = Draw.MolToImage(mol2, size=(500, 500))
          # display(img)
          combined_mol = Chem.CombineMols(mol1, mol2)
          sub_elements = ['Zr','Pd']
          bond_information = bond_info(combined_mol,sub_elements)
          # group the elements with the same bond
          groped_elements = group_by_bond_type(bond_information)
          final_smiles=add_bonding(combined_mol,groped_elements,Regioregularity)

    elif len(bond_type)==2:
          #ladder polymer
          elements = ['[Au]', '[Lr]', '[Ag]', '[La]']
          new_smiles1 = replace_asterisk(smiles1, elements)
          new_smiles2 = replace_asterisk(smiles2, elements)
          mol1 = Chem.MolFromSmiles(new_smiles1)
          mol2 = Chem.MolFromSmiles(new_smiles2)
          combined_mol = Chem.CombineMols(mol1, mol2)
          # getting dict of bond type of each element
          sub_elements = ['Au', 'Lr', 'Ag', 'La']
          bond_information = bond_info(combined_mol,sub_elements)
          # group the elements with the same bond
          groped_elements = group_by_bond_type(bond_information)
          # add bonding between the elements and replace the middle elements with specific patterns
          final_smiles=add_bonding(combined_mol,groped_elements)


    return final_smiles,elements

def get_bond_type(smiles):
    mol = MolFromSmiles(smiles)
    star_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == '*']
    bond_set = set()
    bond_idx = {'DOUBLE': [],
                'SINGLE': []
                }
    for star_atom_idx in star_atoms:
      star_atom = mol.GetAtomWithIdx(star_atom_idx)
      # Get bonds connected to this atom
      bonds = star_atom.GetBonds()
      bond_type = [bond.GetBondTypeAsDouble() for bond in bonds]
      bond_set.add(bond_type[0])
      if bond_type[0] ==1:
        bond_idx['SINGLE'].append(star_atom_idx)
      elif bond_type[0] ==2:
        bond_idx['DOUBLE'].append(star_atom_idx)

    return bond_set,bond_idx



def bond_info(mol, elements):
    # Specify the elements to look for
    element_atomic_nums = [Chem.GetPeriodicTable().GetAtomicNumber(el) for el in elements]

    # Analyze bonds for the specified elements
    bonding_info = {}

    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        if atomic_num in element_atomic_nums:
            atom_idx = atom.GetIdx()
            element_symbol = Chem.GetPeriodicTable().GetElementSymbol(atomic_num)
            bonds = atom.GetBonds()
            bond_types = [bond.GetBondType() for bond in bonds]

            # Initialize the dictionary for the element if it doesn't exist
            if element_symbol not in bonding_info:
                bonding_info[element_symbol] = {
                    'idx': [],
                    'bond_type': set()
                }

            # Append the atom index and bond types to the respective lists/sets
            bonding_info[element_symbol]['idx'].append(atom_idx)
            bonding_info[element_symbol]['bond_type'].update(bond_types)

    return bonding_info

def group_by_bond_type(bond_inf):
    grouped_dict = {}

    for element, details in bond_inf.items():
        bond_type = next(iter(details['bond_type']))  # Extract bond type from the set
        if bond_type not in grouped_dict:
            grouped_dict[bond_type] = {'elements': [], 'indices': []}
        grouped_dict[bond_type]['elements'].append(element)
        grouped_dict[bond_type]['indices'].append(details['idx'])

    return grouped_dict

def add_bonding(combined_mol, group_bond,Regioregularity:bool=True):
    editable_combined_mol = Chem.RWMol(combined_mol)
    if Regioregularity:
      for bond, info in group_bond.items():
          # print(info['indices'])
          # print(bond)
          # print(info['elements'])
          editable_combined_mol.AddBond(info['indices'][0][0], info['indices'][1][1], bond)
    else:
      for bond, info in group_bond.items():
          # print(info['indices'])
          # print(bond)
          # print(info['elements'])
          if len(info['indices'][0])==2:
              editable_combined_mol.AddBond(info['indices'][0][0], info['indices'][0][1], bond)
          elif len(info['indices'][0])==3:
              editable_combined_mol.AddBond(info['indices'][0][1], info['indices'][0][2], bond)

    final_mol = editable_combined_mol.GetMol()
    Chem.SanitizeMol(final_mol)
    final_smiles = Chem.MolToSmiles(final_mol)

    for _, info in group_bond.items():
        final_smiles = replace_middle_elements(final_smiles,info['elements'])

    return final_smiles


def replace_middle_elements(smiles, elements):
    # Generate the patterns for both orders
    patterns = [
        ''.join([f'[{element}]' for element in elements]),  # e.g., [Au][Ag]
        ''.join([f'[{element}]' for element in reversed(elements)]),  # e.g., [Ag][Au]
        '='+'='.join([f'[{element}]' for element in elements]), # =[]=[]
        '='+'='.join([f'[{element}]' for element in reversed(elements)])
    ]
    patterns.extend([f'[{element}][{element}]' for element in elements])
    # Replace each pattern with an empty string
    for pattern in patterns:
        smiles = smiles.replace(pattern, '')

    return smiles


def get_atom_idx(mol,elements):
  return [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == elements]


def get_asterisk_neighbors(mol):
    ast_idx=get_atom_idx(mol,'*')
    atom1 = mol.GetAtomWithIdx(ast_idx[0])
    atom2 = mol.GetAtomWithIdx(ast_idx[1])

    neighbors1 = [neighbor.GetIdx() for neighbor in atom1.GetNeighbors()]
    neighbors2 = [neighbor.GetIdx() for neighbor in atom2.GetNeighbors()]

    neighbors1.extend(neighbors2)
    
    neighbors1 = list(filter(lambda x: x not in ast_idx, neighbors1))

    return neighbors1


def replace_atom_by_idx(mol,idx,atom_sign):
  replacement_atn = atomic_number_calculator(atom_sign)
  mol.ReplaceAtom(idx, Chem.Atom(replacement_atn))


def closing(editeable_mol, bond_type):
    astrisk_idx = get_atom_idx(editeable_mol,'*')
    astrisk_idx.sort(reverse=True)
    n_idx_list = get_asterisk_neighbors(editeable_mol)
      # replace the neighbor with dumi atoms
    for id in n_idx_list:
      replace_atom_by_idx(editeable_mol,id,'[Si]')
    editeable_mol.AddBond(n_idx_list[0], n_idx_list[1], bond_type)
    
    for atom_idx in astrisk_idx:
        editeable_mol.RemoveAtom(atom_idx)

    si_atom_idx = get_atom_idx(editeable_mol,'Si')
    for si_idx in si_atom_idx:
        editeable_mol.ReplaceAtom(si_idx, Chem.Atom('C'))

    final_mol = editeable_mol.GetMol()
    final_smiles = MolToSmiles(final_mol) 
    Chem.SanitizeMol(final_mol)
    return final_smiles



def run(oligomer_length:list[int],oligomer_name:list[str],rru_name:list[str]) -> None:
    # Load cleaned donor and acceptor structures
    dataset_dir: Path = DATASETS / "SMILES_to_BigSMILES_Conversion_wo_block_copolymer.xlsx"
    raw_smiles: pd.DataFrame = pd.read_excel(dataset_dir)
    raw_structure = raw_smiles[['Name', 'SMILES']].rename(columns={'SMILES': 'Monomer'})

    for length, name in zip(oligomer_length,oligomer_name):
            raw_structure[name]  = raw_structure.apply(
            lambda row: monomer_propagation(
                row['Monomer'],
                n_unit=2,
                termination_oo=True,
                Regioregularity=(row['Name'] != 'rra-P3HT')
            ), axis=1
            )

    for name in rru_name:
            raw_structure[f'RRU_{name}'] = raw_structure.apply(
                lambda row: close_ring(
                    row[name]
                    ), axis=1
                )

    # Load dataset
    pu_pkl = DATASETS / "pu_processed.pkl"
    pu_csv = DATASETS / "pu_processed.csv"

    raw_structure.to_pickle(pu_pkl)
    raw_structure.to_csv(pu_csv)


oligomer_length = [2,3]
oligomer_name = ['Dimer','Trimer']
rru_name = ['Monomer', 'Dimer', 'Trimer']

run(oligomer_length, oligomer_name,rru_name)
