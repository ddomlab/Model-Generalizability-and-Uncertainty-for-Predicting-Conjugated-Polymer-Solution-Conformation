import os
from pathlib import Path
import re


HERE: Path = Path(__file__).resolve().parent
RESULTS: Path = HERE.parent/ 'results'

targer_dir: Path = Path(RESULTS/'target_log Rg (nm)')




for item in os.listdir(targer_dir):
    old_folder_path = os.path.join(targer_dir, item)

    for filename in os.listdir(old_folder_path):
        old_file_path = os.path.join(old_folder_path, filename)

        if filename.endswith("_shape.json"):
            long_path = r"\\?\{}".format(old_file_path)  # allow >260 char paths
            os.remove(long_path)
            print(f"Deleted: {old_file_path}")

