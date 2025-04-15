
import numpy as np
from scipy.integrate import simps
from typing import Callable, Optional, Tuple, OrderedDict
from scipy.stats import pearsonr, spearmanr, norm, kendalltau


class CVPPDiagram():
    ''' Metric introduced in arXiv:2010.01118 [cs.LG] based on cross-validatory
    predictive p-values
    '''

    def __init__(self, name='cvpp'):
        self.name = name

    @staticmethod
    def c(y_true, y_pred, y_err, q):
        lhs = np.abs((y_pred - y_true) / y_err)
        rhs = norm.ppf(((1.0 + q) / 2.0), loc=0., scale=1.)
        return np.sum((lhs < rhs).astype(int)) / y_true.shape[0]

    def compute(self, y_true, y_pred, y_err, num_bins=10):
        qs = np.linspace(0, 1, num_bins)
        Cqs = np.empty(qs.shape)
        for ix, q in enumerate(qs):
            Cqs[ix] = self.c(y_true, y_pred, y_err, q)

        return qs, Cqs


class AbsoluteMiscalibrationArea():
    ''' absolute miscalibration area metric with CVPP
    '''

    def __init__(self, name='ama'):
        self.name = name

    def compute(self, y_true, y_pred, y_err, num_bins=10):
        scorer = CVPPDiagram()
        qs, Cqs = scorer.compute(y_true, y_pred, y_err, num_bins=num_bins)

        # compute area
        res = simps(np.abs(Cqs - qs), qs)
        return res
    

def get_calibration_confidence_interval(y_true: np.ndarray, y_pred: np.ndarray, y_err: np.array,
                                        metric_fn: Callable, n_samples: int = 1000) -> Tuple[float, float]:
    '''Bootstrap for 95% confidence interval.
    '''
    if n_samples == 1:
        return metric_fn(y_true, y_pred, y_err), 0

    results = np.zeros((n_samples, 1))
    n = len(y_true)
    for i in range(n_samples):
        idx = np.random.choice(n, int(n * 1.0), replace=True)
        results[i] = (metric_fn(y_true[idx], y_pred[idx], y_err[idx]))

    m = np.mean(results)
    ci_top = np.percentile(results, 97.5)
    ci_bot = np.percentile(results, 2.5)

    return m, ci_bot, ci_top



# ama_mean, ama_ci_low, ama_ci_top = get_calibration_confidence_interval(y_true, y_pred, y_err,
#                                                                                          AbsoluteMiscalibrationArea.compute,n_samples=1)



