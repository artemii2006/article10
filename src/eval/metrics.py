"""R²_oos и тест Кларка-Уэста."""

import numpy as np
from scipy import stats


def r2_oos(actual: np.ndarray, forecast: np.ndarray, benchmark: np.ndarray) -> float:
    """R²_oos по Campbell & Thompson (2007): сравнение с прогнозом-бенчмарком.

    В статье бенчмарк — гипотеза ожиданий, то есть историческое среднее
    избыточной доходности, посчитанное на той же информации, что и прогноз.
    """
    actual, forecast, benchmark = map(np.asarray, (actual, forecast, benchmark))
    sse_model = np.sum((actual - forecast) ** 2)
    sse_benchmark = np.sum((actual - benchmark) ** 2)
    return 1.0 - sse_model / sse_benchmark


def _newey_west_var(x: np.ndarray, lags: int) -> float:
    """Дисперсия среднего с поправкой Ньюи-Уэста на автокорреляцию."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    e = x - x.mean()
    gamma0 = np.dot(e, e) / n
    total = gamma0
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        gamma = np.dot(e[lag:], e[:-lag]) / n
        total += 2.0 * weight * gamma
    return total / n


def clark_west(
    actual: np.ndarray,
    forecast: np.ndarray,
    benchmark: np.ndarray,
    lags: int = 11,
) -> tuple[float, float]:
    """Статистика MSPE-adjusted Кларка-Уэста (2007) и односторонний p-value.

    H0: R²_oos <= 0 против H1: R²_oos > 0. Поправка снимает смещение, возникающее
    из-за того, что вложенная большая модель оценивает лишние параметры и
    поэтому проигрывает по MSPE даже когда предсказуемость реально есть.

    lags — порядок поправки Ньюи-Уэста. По умолчанию 11: годовые доходности на
    месячных данных перекрываются на 11 месяцев, без этого t-статистика
    завышается в разы.
    """
    actual, forecast, benchmark = map(np.asarray, (actual, forecast, benchmark))
    f = (
        (actual - benchmark) ** 2
        - (actual - forecast) ** 2
        + (benchmark - forecast) ** 2
    )
    mean_f = f.mean()
    var_f = _newey_west_var(f, lags)
    if var_f <= 0:
        return float("nan"), float("nan")
    stat = mean_f / np.sqrt(var_f)
    p_value = 1.0 - stats.norm.cdf(stat)
    return float(stat), float(p_value)
