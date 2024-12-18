import os
from pathlib import Path
import re


HERE: Path = Path(__file__).resolve().parent
RESULTS: Path = HERE.parent/ 'results'

targer_dir: Path = Path(RESULTS/'target_Rg')




for item in os.listdir(targer_dir):
    if item != 'test':
        old_folder_path = os.path.join(targer_dir, item)
    
    for filename in os.listdir(old_folder_path):
        old_file_path = os.path.join(old_folder_path, filename)
        if os.path.isfile(old_file_path):
            print(f"Old filename: {filename}")
            if "Standard" not in filename and "Robust Scaler" not in filename:
                    old_file_name_split =  filename.split("_")
                    old_file_name_split.insert(2, "Standard")
                    new_filename = "_".join(old_file_name_split)
                    new_file_path = os.path.join(old_folder_path, new_filename)
                    os.rename(old_file_path, new_file_path)
                    # print(new_file_path)
