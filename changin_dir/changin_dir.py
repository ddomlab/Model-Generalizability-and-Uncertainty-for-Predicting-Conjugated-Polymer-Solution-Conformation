import os
from pathlib import Path
import shutil


HERE: Path = Path(__file__).resolve().parent
RESULTS: Path = HERE.parent/ 'results'

old_dir = Path(RESULTS)
for root, dirs, files in os.walk(old_dir):  # Adjust pattern as needed
    print(dirs)
