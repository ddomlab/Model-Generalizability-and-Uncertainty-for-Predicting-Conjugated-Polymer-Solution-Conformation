import os
from pathlib import Path
import re


HERE: Path = Path(__file__).resolve().parent
RESULTS: Path = HERE.parent/ 'results'

targer_dir: Path = Path(RESULTS/'target_Lp')






def transform_filename_location(old_filename,target_dir,old_file_path):
        structures = ["Monomer", "Dimer", "Trimer", "RRU Monomer", "RRU Dimer", "RRU Trimer"]

        # Join the list into a regex pattern
        structure_pattern = "|".join(structures)
        splitted_file_name:list[str] = old_filename.split('_')
        if "model" in splitted_file_name[0]:
            #different pattern should be proposed
            numerical_pattern = re.compile(
                r"^(RF|MLR)\s+model_mean\s+imputer_(\w+)_feats_(predictions|scores)\.(json|csv)$"
                          )

            representation_numerical_pattern = re.compile(
                rf"^(RF|MLR)\s+model_mean\s+imputer_(\w+)_feats_(MACCS|Mordred)_({structure_pattern})_(predictions|scores)\.(json|csv)$"
                )


            if numerical_pattern.match(old_filename):
                  match = numerical_pattern.match(old_filename)
                  if match:
                      model, numerical, result_type, extension = match.groups()
                      new_filename = f"({numerical})_{model}_mean_{result_type}.{extension}"
                      representation = "scaler"

            elif representation_numerical_pattern.match(old_filename):
                  match = representation_numerical_pattern.match(old_filename)
                  if match:
                      model, numerical, fingerprint, representation, result_type, extension = match.groups()
                      new_filename = f"({fingerprint}_{numerical})_{model}_mean_{result_type}.{extension}"
                      representation = f'{representation}_scaler'
            else:
                print("Filename format does not match the expected pattern.")

        else:
            # regular type
            pattern = rf"^(RF|MLR)_(ECFP(\d+)|MACCS|Mordred)_(binary|count)?_?(?:(\d+)?bits)?_?({structure_pattern})_(scores|predictions)\.(json|csv)$"
            
            
            match = re.match(pattern, old_filename)
            if match:
                # Extract the components from the old filename
                model, fingerprint, radius, count_type, bits, representation, result_type, extension = match.groups()

                if radius:
                # If ECFP number exists, format it accordingly
                    new_fingerprint = f"ECFP{int(radius)*2}"
                    new_filename = f"({new_fingerprint})_{count_type}_{bits.strip('_')}_{model}_{result_type}.{extension}"

                else:
                    # If MACCS, just use the fingerprint directly
                    new_filename = f"({fingerprint})_{model}_{result_type}.{extension}"

            else:
              print('does not match')
        print(f'new_filename: {new_filename}')
        if representation:
            new_folder = os.path.join(target_dir, representation)
            os.makedirs(new_folder, exist_ok=True)            
            new_file_path = os.path.join(new_folder, new_filename)
            os.rename(old_file_path, new_file_path)
            print(f"Renamed and moved to: {new_file_path}")







# for item in os.listdir(targer_dir):
#     if item != 'test':
#         old_folder_path = os.path.join(targer_dir, item)
    
#     for filename in os.listdir(old_folder_path):
#         old_file_path = os.path.join(old_folder_path, filename)
#         if os.path.isfile(old_file_path):
#             print(f"Old filename: {filename}")
#             transform_filename_location(filename,target_dir=targer_dir,old_file_path=old_file_path)





                

