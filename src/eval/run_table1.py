"""Воспроизведение таблицы 1, панели A и B: прогноз по форвардным ставкам.

Панель A — регрессии на главные компоненты и PLS, панель B — штрафные
регрессии. Бенчмарк — гипотеза ожиданий (историческое среднее). Первый прогноз
делается в 1989:01 и реализуется в 1990:01, последний реализуется в 2018:12.

Запуск:  python -m src.eval.run_table1
"""

from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd

from src.eval.expanding_window import expanding_window_forecast
from src.eval.metrics import clark_west, r2_oos
from src.models.linear import (
    ENET_GRID,
    LASSO_GRID,
    RIDGE_GRID,
    make_elastic_net,
    make_lasso,
    make_pcr,
    make_pls,
    make_ridge,
)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data_processed"
RESULTS_DIR = ROOT / "results"

MATURITIES = [2, 3, 4, 5, 7, 10]
FIRST_ORIGIN = pd.Period("1989-01", freq="M")

# (название как в таблице, фабрика модели, сетка гиперпараметров)
SPECIFICATIONS = [
    ("PCA (10 components)", partial(make_pcr, 10), None),
    ("PCA (5 components)", partial(make_pcr, 5), None),
    ("PCA (3 components)", partial(make_pcr, 3), None),
    ("PCA-Squared (5 components)", partial(make_pcr, 5, squared=True), None),
    ("PCA-Squared (3 components)", partial(make_pcr, 3, squared=True), None),
    ("Partial Least Squares (5 components)", partial(make_pls, 5), None),
    ("Partial Least Squares (3 components)", partial(make_pls, 3), None),
    ("Ridge", make_ridge, RIDGE_GRID),
    ("Lasso", make_lasso, LASSO_GRID),
    ("Elastic Net", make_elastic_net, ENET_GRID),
]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    forwards = pd.read_csv(PROCESSED_DIR / "forward_rates.csv", index_col="date")
    excess = pd.read_csv(PROCESSED_DIR / "excess_returns.csv", index_col="date")
    forwards.index = pd.PeriodIndex(forwards.index, freq="M")
    excess.index = pd.PeriodIndex(excess.index, freq="M")
    return forwards, excess[[f"rx_{n}" for n in MATURITIES]]


def run_specification(name, factory, grid, X, excess, gap: int) -> dict:
    """Прогоняет одну модель по всем срокам и считает R²_oos с p-value."""
    row = {"model": name}
    stacked = {}

    for n in MATURITIES:
        result = expanding_window_forecast(
            X=X,
            y=excess[f"rx_{n}"],
            build_model=factory,
            first_origin=FIRST_ORIGIN,
            param_grid=grid,
            horizon=gap,
        )
        stacked[n] = result
        row[f"r2_{n}"] = r2_oos(result.actual, result.forecast, result.benchmark)
        _, p_value = clark_west(result.actual, result.forecast, result.benchmark)
        row[f"p_{n}"] = p_value

    # Равновзвешенный портфель: усредняем по шести срокам.
    actual_ew = np.mean([stacked[n].actual for n in MATURITIES], axis=0)
    forecast_ew = np.mean([stacked[n].forecast for n in MATURITIES], axis=0)
    benchmark_ew = np.mean([stacked[n].benchmark for n in MATURITIES], axis=0)
    row["r2_ew"] = r2_oos(actual_ew, forecast_ew, benchmark_ew)
    row["p_ew"] = clark_west(actual_ew, forecast_ew, benchmark_ew)[1]

    n_obs = len(stacked[MATURITIES[0]].origins)
    return row, n_obs


def format_table(results: pd.DataFrame) -> pd.DataFrame:
    """Приводит к виду таблицы в статье: R² в процентах, p-value только при R² > 0."""
    out = pd.DataFrame(index=results["model"])
    for key in [str(n) for n in MATURITIES] + ["ew"]:
        r2 = results[f"r2_{key}"].values
        p = results[f"p_{key}"].values
        out[f"xr_{key}"] = [f"{100 * v:.1f}%" for v in r2]
        out[f"p_{key}"] = [
            f"{pv:.3f}" if r2v > 0 and np.isfinite(pv) else "" for r2v, pv in zip(r2, p)
        ]
    return out


def main(gap: int = 1) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    forwards, excess = load_data()

    rows, n_obs = [], None
    for name, factory, grid in SPECIFICATIONS:
        row, n_obs = run_specification(name, factory, grid, forwards, excess, gap)
        rows.append(row)
        print(
            f"{name:38s} "
            + " ".join(f"{100 * row[f'r2_{n}']:6.1f}%" for n in MATURITIES)
            + f"  EW {100 * row['r2_ew']:6.1f}%"
        )

    suffix = f"gap{gap}"
    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_DIR / f"table1_panels_ab_raw_{suffix}.csv", index=False)
    formatted = format_table(results)
    formatted.to_csv(RESULTS_DIR / f"table1_panels_ab_{suffix}.csv")
    print(f"\nпрогнозов вне выборки: {n_obs} (реализации 1990:01-2018:12)")
    print(f"сохранено: {RESULTS_DIR / f'table1_panels_ab_{suffix}.csv'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gap",
        type=int,
        default=1,
        help=(
            "сколько месяцев отступать от даты прогноза до последней обучающей цели. "
            "1 — как в статье; 12 — без заглядывания вперёд"
        ),
    )
    main(**vars(parser.parse_args()))
