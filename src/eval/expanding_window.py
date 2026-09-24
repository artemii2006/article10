"""Рекурсивный прогноз в расширяющемся окне.

Схема из статьи (раздел 3.3):
  * модель переоценивается каждый месяц, обучающая выборка растёт;
  * внутри доступной истории первые 85% идут в train, последние 15% — в
    validation, разбиение строго хронологическое, без перемешивания;
  * гиперпараметры выбираются на validation, затем модель переобучается на
    train + validation;
  * методам без гиперпараметров (OLS, PCR, PLS с фиксированным числом
    компонент) validation не нужна, они учатся на всей доступной истории.

Ключевой момент, которого в статье явно нет, но без которого возникает
заглядывание вперёд: цель с датой t реализуется только в t + 12. Поэтому в
момент прогноза t обучаться можно лишь на целях с датой не позже t - 12.
"""

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd

HORIZON = 12
VALIDATION_FRACTION = 0.15


@dataclass
class ForecastResult:
    """Прогнозы, реализации и бенчмарк на тестовом участке."""

    origins: pd.PeriodIndex
    actual: np.ndarray
    forecast: np.ndarray
    benchmark: np.ndarray
    chosen_params: list = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "actual": self.actual,
                "forecast": self.forecast,
                "benchmark": self.benchmark,
            },
            index=self.origins,
        )


def expanding_window_forecast(
    X: pd.DataFrame,
    y: pd.Series,
    build_model: Callable,
    first_origin: pd.Period,
    param_grid: Sequence | None = None,
    horizon: int = HORIZON,
    validation_fraction: float = VALIDATION_FRACTION,
) -> ForecastResult:
    """Прогнозирует y расширяющимся окном, начиная с даты first_origin.

    build_model(param) возвращает необученную модель с методами fit/predict.
    Если param_grid задан, на каждом шаге перебираются все значения и берётся
    то, что минимизирует MSE на validation.
    """
    common = X.index.intersection(y.dropna().index)
    X, y = X.loc[common], y.loc[common]

    origins = X.index[X.index >= first_origin]
    forecasts, actuals, benchmarks, chosen = [], [], [], []

    for origin in origins:
        # Обучаться можно только на целях, которые уже реализовались к моменту origin.
        trainable = X.index <= origin - horizon
        X_hist, y_hist = X[trainable], y[trainable]
        if len(y_hist) < 60:
            raise ValueError(f"слишком короткая история на {origin}: {len(y_hist)}")

        if param_grid is None:
            model = build_model()
            model.fit(X_hist.values, y_hist.values)
            chosen.append(None)
        else:
            n_val = max(1, int(round(validation_fraction * len(y_hist))))
            X_tr, y_tr = X_hist.iloc[:-n_val], y_hist.iloc[:-n_val]
            X_val, y_val = X_hist.iloc[-n_val:], y_hist.iloc[-n_val:]

            errors = []
            for param in param_grid:
                candidate = build_model(param)
                candidate.fit(X_tr.values, y_tr.values)
                pred = candidate.predict(X_val.values)
                errors.append(np.mean((y_val.values - pred) ** 2))

            best = param_grid[int(np.argmin(errors))]
            chosen.append(best)
            model = build_model(best)
            model.fit(X_hist.values, y_hist.values)

        forecasts.append(float(np.ravel(model.predict(X.loc[[origin]].values))[0]))
        # Бенчмарк гипотезы ожиданий: среднее по той же доступной истории.
        benchmarks.append(float(y_hist.mean()))
        actuals.append(float(y.loc[origin]))

    return ForecastResult(
        origins=pd.PeriodIndex(origins),
        actual=np.array(actuals),
        forecast=np.array(forecasts),
        benchmark=np.array(benchmarks),
        chosen_params=chosen,
    )
