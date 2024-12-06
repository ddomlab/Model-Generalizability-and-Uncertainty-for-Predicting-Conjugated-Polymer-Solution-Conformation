from typing import Union, List, Optional, Callable, Dict

from matplotlib.pylab import RandomState
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import BaseShuffleSplit, GroupShuffleSplit
from sklearn.model_selection._split import _validate_shuffle_split, _num_samples
from sklearn.cluster import MiniBatchKMeans

from sklearn.model_selection import ShuffleSplit, KFold




def get_mood_splitters(smiles, n_splits: int = 5, random_state: int = 0, n_jobs: Optional[int] = None):
    scaffolds = [dm.to_smiles(dm.to_scaffold_murcko(dm.to_mol(smi))) for smi in smiles]
    splitters = {
        "Random": ShuffleSplit(n_splits=n_splits, random_state=random_state),
        "Scaffold": PredefinedGroupShuffleSplit(
            groups=scaffolds, n_splits=n_splits, random_state=random_state
        ),
        "Perimeter": PerimeterSplit(
            n_clusters=25, n_splits=n_splits, random_state=random_state, n_jobs=n_jobs
        ),
        "Maximum Dissimilarity": MaxDissimilaritySplit(
            n_clusters=25, n_splits=n_splits, random_state=random_state, n_jobs=n_jobs
        ),
    }
    return splitters



class IID_splitter(KFold):
    
    
    def __init__(self, ):
        super().__init__()  
