"""R²_oos и тест Кларка-Уэста."""

import numpy as np
from scipy import stats


def r2_oos(actual, forecast, benchmark):
    actual, forecast, benchmark = map(np.asarray, (actual, forecast, benchmark))
    return 1.0 - np.sum((actual - forecast) ** 2) / np.sum((actual - benchmark) ** 2)


def newey_west_var(x, lags):
    x = np.asarray(x, dtype=float)
    n = len(x)
    e = x - x.mean()
    total = np.dot(e, e) / n
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        total += 2.0 * weight * np.dot(e[lag:], e[:-lag]) / n
    return total / n


def clark_west(actual, forecast, benchmark, lags=11):
    """H0: R²_oos <= 0. Лаги Ньюи-Уэста нужны из-за перекрытия годовых доходностей."""
    actual, forecast, benchmark = map(np.asarray, (actual, forecast, benchmark))
    f = (actual - benchmark) ** 2 - (actual - forecast) ** 2 + (benchmark - forecast) ** 2
    var = newey_west_var(f, lags)
    if var <= 0:
        return float("nan"), float("nan")
    stat = f.mean() / np.sqrt(var)
    return float(stat), float(1.0 - stats.norm.cdf(stat))
