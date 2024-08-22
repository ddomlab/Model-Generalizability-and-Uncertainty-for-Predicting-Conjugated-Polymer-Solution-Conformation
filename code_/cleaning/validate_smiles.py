from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

# clean the data by removing block and random copolymers
# def find_identical_molecules(series, radius, bits):
#     # Get unique SMILES
#     series = series.apply(lambda x: Chem.CanonSmiles(x))
#     molecules = list(set(series))

#     # create ECFP fingerprints for all molecules in the series
#     fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(mol), radius, nBits=bits) for mol in molecules]

#     # compare all pairs of unique SMILES
#     clashes = 0
#     for i, mol1 in enumerate(molecules):
#         for j, mol2 in enumerate(molecules):
#             if j > i:
#                 fp1 = fps[i]
#                 fp2 = fps[j]
#                 sim = DataStructs.TanimotoSimilarity(fp1, fp2)
#                 if sim == 1:
#                     clashes += 1
#                     print(f"Molecule {i+1}: {mol1}")
#                     print(f"Molecule {j+1}: {mol2}\n")
#     print("radius:", radius, "\tbits:", bits, "\t\tmolecule clashes:", clashes, "\n\n")
#     return clashes

# To-Do: clean this code:

def cleaning_smiles(smiles: pd.Series, polymer: pd.Series, mode: str = "read"):
    """
    Mode ("read" or "edit") determines whether SMILES and/or polymer names should be edited.
    """
    mode = mode.lower()
    duplicates: list[int] = []
    clashes: int = 0
    shared_names: int = 0

    canonicalized_smiles: list[str] = smiles.apply(lambda x: Chem.CanonSmiles(x))
    fps: list = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(mol), 6, nBits=4096) for mol in canonicalized_smiles]

    for x in polymer_structures_df.index:
      row = polymer_structures_df.iloc[x]
      if row["Validation (TRUE/FALSE/TBD/D(Delete))"] == "D":
          duplicates.append(x)
    print(len(duplicates))

    for i, mol1 in enumerate(canonicalized_smiles):
        if i in duplicates:
                continue
        for j, mol2 in enumerate(canonicalized_smiles):
            if j > i:
              if j in duplicates:
                continue

              similarity: float = DataStructs.TanimotoSimilarity(fps[i], fps[j])
              if similarity == 1:
                  clashes += 1
                  if polymer[j] == polymer[i]:
                    duplicates.append(j)
                    #Should rows with the same structures and name automatically be deleted?  Yes
                  else:
                    print(f"Molecules {polymer[i]} and {polymer[j]} clashes at rows:{i+2,j+2}")
                    if mode == "edit":
                      detele_row: int = int(input("Version to be deleted (Enter row number)"))
                      duplicates.append(detele_row-2)
                      continue

              elif polymer[j] == polymer[i]:
                shared_names += 1
                print(f"The diffrent structures at rows:{i+2,j+2} are called {polymer[i]}")
                if mode == "edit":
                  action: str = input("Delete or Rename? (Click Enter to pass)").lower()
                  if action == "rename":
                    rename_row: int = int(input("Version to be renamed (Enter row number)"))
                    polymer[rename_row-2] = input("New name:")
                  elif action == "delete":
                      delete_row: int = int(input("Version to be deleted (Enter row number)"))
                      duplicates.append(delete_row-2)
                      continue

    print("\tmolecule clashes:", clashes, "\n\tshared names:", shared_names)
    print(len(duplicates))


def smile_main():
    r_groups_csv: Path = DATASETS / "Min_2020_n558" / "raw" / "r_groups.csv"
    clean_csv: Path = DATASETS / "Min_2020_n558" / "cleaned R groups.csv"

    r_groups: pd.DataFrame = pd.read_csv(r_groups_csv)
    r_groups: pd.DataFrame = ingest_r_groups(r_groups)
    r_groups.to_csv(clean_csv, index=False)


if __name__ == "__main__":
    r_main()
