import json
from itertools import product
from pathlib import Path
from typing import List

# import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rc


HERE: Path = Path(__file__).parent
filters = HERE.parent / "training" / "filters.json"
with open(filters, "r") as f:
    FILTERS: dict = json.load(f)

score_bounds: dict[str, int] = {"r": 1, "r2": 1, "mae": 7.5, "rmse": 7.5}
var_titles: dict[str, str] = {"stdev": "Standard Deviation", "stderr": "Standard Error"}
