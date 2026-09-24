"""Рекурсивный прогноз в расширяющемся окне.

Модель переоценивается каждый месяц. Внутри доступной истории первые 85% идут
в train, последние 15% — в validation, разбиение хронологическое. Гиперпараметр
выбирается на validation, затем модель переобучается на всей истории.

gap — отступ от даты прогноза до последней обучающей цели. Годовая доходность с
датой t известна только в t+12, поэтому gap=12 исключает заглядывание вперёд,
а gap=1 воспроизводит статью.
"""

import numpy as np
import pandas as pd

VALIDATION_FRACTION = 0.15


def forecast(X, y, predict, grid=None, gap=1, first_origin=None, validation_fraction=VALIDATION_FRACTION):
    """Возвращает таблицу с колонками actual, forecast, benchmark и выбранные гиперпараметры."""
    common = X.index.intersection(y.dropna().index)
    X, y = X.loc[common], y.loc[common]
    origins = X.index[X.index >= first_origin]

    actual, predicted, benchmark, chosen = [], [], [], []

    for origin in origins:
        history = X.index <= origin - gap
        X_hist, y_hist = X[history].values, y[history].values
        X_now = X.loc[[origin]].values

        if grid is None:
            predicted.append(float(predict(X_hist, y_hist, X_now)[0]))
            chosen.append(None)
        else:
            n_val = max(1, int(round(validation_fraction * len(y_hist))))
            X_train, y_train = X_hist[:-n_val], y_hist[:-n_val]
            X_val, y_val = X_hist[-n_val:], y_hist[-n_val:]

            errors = [
                np.mean((y_val - predict(X_train, y_train, X_val, param)) ** 2)
                for param in grid
            ]
            best = grid[int(np.argmin(errors))]
            chosen.append(best)
            predicted.append(float(predict(X_hist, y_hist, X_now, best)[0]))

        benchmark.append(float(y_hist.mean()))
        actual.append(float(y.loc[origin]))

    result = pd.DataFrame(
        {"actual": actual, "forecast": predicted, "benchmark": benchmark},
        index=origins,
    )
    return result, chosen
