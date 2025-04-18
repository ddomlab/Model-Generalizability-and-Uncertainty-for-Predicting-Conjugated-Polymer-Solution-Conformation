
import numpy as np
from scipy.integrate import simpson
from typing import Callable, Optional, Tuple, OrderedDict
from scipy.stats import pearsonr, spearmanr, norm, kendalltau


class CVPPDiagram():
    ''' Metric introduced in arXiv: https://arxiv.org/pdf/2010.01118
    '''

    def __init__(self, name='cvpp'):
        self.name = name

    @staticmethod
    def c(y_true, y_pred, y_err, q):
        lhs = np.abs((y_pred - y_true) / y_err)
        rhs = norm.ppf(((1.0 + q) / 2.0), loc=0., scale=1.)
        return np.sum((lhs < rhs).astype(int)) / y_true.shape[0]

    def compute(self, y_true, y_pred, y_err, num_bins=10)-> Tuple[np.ndarray, np.ndarray]:
        qs = np.linspace(0, 1, num_bins)
        Cqs = np.empty(qs.shape)
        for ix, q in enumerate(qs):
            Cqs[ix] = self.c(y_true, y_pred, y_err, q)

        return qs, Cqs


class AbsoluteMiscalibrationArea(CVPPDiagram):
    ''' absolute miscalibration area metric with CVPP
    '''

    def __init__(self, name='ama'):
        self.name = name

    def compute(self, y_true, y_pred, y_err, num_bins=10):
        scorer = CVPPDiagram()
        qs, Cqs = scorer.compute(y_true, y_pred, y_err, num_bins=num_bins)

        # compute area
        res = simpson(np.abs(Cqs - qs), qs)
        return res
    

def get_calibration_confidence_interval(y_true: np.ndarray, y_pred: np.ndarray, y_err: np.array,
                                        metric_fn: Callable, n_samples: int = 1000) -> Tuple[float, float]:
    '''Bootstrap for 95% confidence interval.
    '''
    if n_samples == 1:
        return metric_fn.compute(y_true, y_pred, y_err), 0

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
#                                                                                          AbsoluteMiscalibrationArea.compute,n_samples=1000)



    # def predict(self, smi_split: types.ArraySplit, x: types.ArraySplit, y: types.ArraySplit):
    #     uniq_smi = smi_split.values
    #     y_mu, y_std = [], []

    #     for m in self.model:
    #         if self.task == enums.TaskType.regression:
    #             y_dists = m.pred_dist(x.values)
    #             y_mu.append(y_dists.loc)
    #             y_std.append(np.sqrt(y_dists.var))
    #         elif self.task == enums.TaskType.binary:
    #             y_mu.append(m.predict(x.values))
    #             y_std.append(m.predict_proba(x.values)[:, 1])
    #         else:
    #             raise NotImplementedError(f'{self.task} not implemented')

    #     # transpose and turn into numpy array
    #     y_mu = np.transpose(y_mu).reshape(y.values.shape)
    #     y_std = np.transpose(y_std).reshape(y.values.shape)

    #     return uniq_smi, y_mu, y_std


    # MatFOLD take the std of predictions in different seeds and CVs
    # def predict_baseline_uncertainty(model, X):
    # """Predicts the uncertainty of the model.

    # For GP regressors, we use the included uncertainty estimation.
    # For classifiers, the entropy of the prediction is used as uncertainty.
    # For regressors, the variance of the prediction is used as uncertainty.
    # """
    # if isinstance(model, ClassifierMixin):
    #     uncertainty = model.predict_proba(X)

    # elif isinstance(model, GaussianProcessRegressor):
    #     std = model.predict(X, return_std=True)[1]
    #     uncertainty = std**2

    # else:
    #     # VotingRegressor or RandomForestRegressor
    #     preds = dm.utils.parallelized(lambda x: x.predict(X), model.estimators_, n_jobs=model.n_jobs)
    #     uncertainty = np.var(preds, axis=0)

    # return uncertainty


    # In MOOD :
    # def weighted_pearson_calibration(preds, target, uncertainty, sample_weights=None):
    # error = torch.abs(preds - target)
    # return weighted_pearson(error, uncertainty, sample_weights)