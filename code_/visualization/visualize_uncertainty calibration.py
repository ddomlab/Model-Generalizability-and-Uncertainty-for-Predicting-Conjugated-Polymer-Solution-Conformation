


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