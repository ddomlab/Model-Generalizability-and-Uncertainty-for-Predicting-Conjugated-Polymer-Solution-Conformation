import numpy as np
from scipy.integrate import simpson
from typing import Callable, Optional, Tuple, OrderedDict
from scipy.stats import pearsonr, spearmanr, norm
from sklearn.metrics import root_mean_squared_error


def compute_cvpp(y_true: np.ndarray, 
                y_pred: np.ndarray,
                y_err: np.ndarray,
                step: float = 0.05
                ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the CVPP diagram values: observed coverage C(q) at various nominal confidence levels q.

    Parameters:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted means.
        y_err (np.ndarray): Predicted standard deviations.
        step (float): Step size for confidence levels (default: 0.05).

    Returns:
        Tuple[np.ndarray, np.ndarray]: (qs, Cqs), where
            - qs: array of nominal confidence levels
            - Cqs: corresponding empirical coverage values
    """
    qs = np.arange(0, 1.0 + step, step)
    Cqs = np.empty_like(qs)

    for i, q in enumerate(qs):
        z = norm.ppf((1.0 + q) / 2.0)
        standardized_error  = np.abs((y_pred - y_true) / y_err)
        Cqs[i] = np.mean(standardized_error  < z)

    return qs, Cqs


def compute_ama(y_true: np.ndarray, 
                y_pred: np.ndarray,
                y_err: np.ndarray,
                step: float = 0.05) -> float:
    
    """
    Computes the Absolute Miscalibration Area (AMA) using CVPP.

    Returns:
        float: The AMA score.
    """
    qs, Cqs = compute_cvpp(y_true, y_pred, y_err, step=step)
    ama = simpson(np.abs(Cqs - qs), qs)
    return ama


def compute_residual_error_cal(y_true: np.ndarray, 
                                y_pred: np.ndarray,
                                y_err: np.ndarray,
                                ):

    res = np.abs(y_true-y_pred)
    correlation = spearmanr(res, y_err)[0]
    return correlation


def gaussian_nll(
                y_true,
                y_pred,
                y_err,
                eps:float=1e-6, 
                reduce='mean'
                ):
    """
    Computes Negative Log-Likelihood (NLL) using scipy.stats.norm.logpdf.

    Parameters:
    -----------
    y_true : np.ndarray
        True values, shape (n_samples,)
    y_pred_mean : np.ndarray
        Predicted means, shape (n_samples,)
    y_pred_std : np.ndarray
        Predicted standard deviations, shape (n_samples,)
    reduce : str
        One of 'mean', 'sum', or 'none' to control the reduction of NLL

    Returns:
    --------
    nll : float or np.ndarray
        The computed NLL value(s)
    """
    eps = 1e-6 
    y_err = np.clip(y_err, eps, None)

    log_probs = norm.logpdf(y_true, loc=y_pred, scale=y_err)
    nll = -log_probs  

    if reduce == 'mean':
        return np.mean(nll)
    elif reduce == 'sum':
        return np.sum(nll)
    elif reduce == 'none':
        return nll
    else:
        raise ValueError("reduce must be one of: 'mean', 'sum', or 'none'")





def uncertainty_metrics(y_true: np.ndarray, 
                        y_pred: np.ndarray,
                        y_err: np.ndarray,
                        step: float = 0.05
                        ):
    

    spearman_cor= compute_residual_error_cal(y_true, y_pred, y_err)
    calibration_area = compute_ama(y_true, y_pred, y_err, step=step)
    sharpness = np.sqrt(np.mean(y_err**2))
    cv = np.sqrt(((y_err - y_err.mean())**2).sum() / (len(y_err)-1)) / y_err.mean()
    nll_score = gaussian_nll(y_true, y_pred, y_err, reduce='mean')
    
    return {
            'NLL': nll_score,
            'Sharpness': sharpness,
            'CV': cv,
            'Spearman': spearman_cor,
            'Calibration Area': calibration_area
            }
