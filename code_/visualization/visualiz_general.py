import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import json
from visualization_setting import save_img_path

HERE = Path(__file__).resolve().parent
DATASETS = HERE.parent.parent/'datasets'
JSONS =  DATASETS/'json_resources'




with open(JSONS/'data_summary_monitor.json', 'r') as f:
    data = json.load(f)
targets = data["Target"]
initial_cleaned = np.array(data["Initial Cleaned"])
block_copolymer = initial_cleaned - np.array(data["After dropping block copolymer"])
solid_additives = np.array(data["After dropping block copolymer"]) - np.array(data["After dropping solid additives"])
polymer_hsps = np.array(data["After dropping solid additives"]) - np.array(data["After dropping unknown polymer HSPs"])

def clean_target_labels(targets):
    # Replace specific labels with cleaned versions
    replacements = {
        "Rh (IW avg log)": "Rh (nm)",
        "Rg1 (nm)": "Rg (nm)"
    }
    return [replacements.get(target, target) for target in targets]

targets = clean_target_labels(targets)

fig, ax = plt.subplots(figsize=(7, 6))
bar_width = 0.5
x = np.arange(len(targets)) 

bars = ax.bar(x, initial_cleaned, bar_width, label='Final dataset', color='#66C2A5')
ax.bar(x, block_copolymer, bar_width, bottom=initial_cleaned - block_copolymer, label='Dropped Block Copolymer', color='#EA6F00')
ax.bar(x, solid_additives, bar_width, bottom=initial_cleaned - block_copolymer - solid_additives, label='Dropped Solid Additives', color='#FFAA5D')
ax.bar(x, polymer_hsps, bar_width, bottom=initial_cleaned - block_copolymer - solid_additives - polymer_hsps, label='Dropped unknown Polymer HSPs', color='#FFC997')

remaining_data_points = initial_cleaned - block_copolymer - solid_additives - polymer_hsps
for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height()/2, str(remaining_data_points[i]), 
                ha='center', va='bottom', fontsize=15, color='black')



ax.set_ylabel('Number of Data Points', fontsize=16)
ax.set_title('Data Reduction at Each Step of Cleaning', fontsize=18, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(targets, rotation=45, ha='right')
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)

ax.legend(fontsize=12)
plt.tight_layout()
save_img_path(HERE/'analysis and test','Data Reduction at Each Step of cleaning.png')
plt.show()
plt.close()
