"""Таблица 1, панели A и B: прогноз избыточных доходностей по форвардным ставкам.

Первый прогноз делается в 1989:01 и реализуется в 1990:01, последний — в 2018:12.

Запуск:  python -m src.eval.run_table1 [--gap 12]
"""

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.expanding_window import forecast
from src.eval.metrics import clark_west, r2_oos
from src.models.linear import (
    ENET_GRID,
    LASSO_GRID,
    RIDGE_GRID,
    elastic_net,
    lasso,
    pcr,
    pls,
    ridge,
)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data_processed"
RESULTS_DIR = ROOT / "results"

MATURITIES = [2, 3, 4, 5, 7, 10]
FIRST_ORIGIN = pd.Period("1989-01", freq="M")

SPECIFICATIONS = [
    ("PCA (10 components)", partial(pcr, n_components=10), None),
    ("PCA (5 components)", partial(pcr, n_components=5), None),
    ("PCA (3 components)", partial(pcr, n_components=3), None),
    ("PCA-Squared (5 components)", partial(pcr, n_components=5, squared=True), None),
    ("PCA-Squared (3 components)", partial(pcr, n_components=3, squared=True), None),
    ("Partial Least Squares (5 components)", partial(pls, n_components=5), None),
    ("Partial Least Squares (3 components)", partial(pls, n_components=3), None),
    ("Ridge", ridge, RIDGE_GRID),
    ("Lasso", lasso, LASSO_GRID),
    ("Elastic Net", elastic_net, ENET_GRID),
]


def load_data():
    forwards = pd.read_csv(PROCESSED_DIR / "forward_rates.csv", index_col="date")
    excess = pd.read_csv(PROCESSED_DIR / "excess_returns.csv", index_col="date")
    forwards.index = pd.PeriodIndex(forwards.index, freq="M")
    excess.index = pd.PeriodIndex(excess.index, freq="M")
    return forwards, excess[[f"rx_{n}" for n in MATURITIES]]


def run_specification(name, predict, grid, X, excess, gap):
    row = {"model": name}
    predictions = {}

    for n in MATURITIES:
        result, _ = forecast(
            X, excess[f"rx_{n}"], predict, grid=grid, gap=gap, first_origin=FIRST_ORIGIN
        )
        predictions[n] = result
        row[f"r2_{n}"] = r2_oos(result.actual, result.forecast, result.benchmark)
        row[f"p_{n}"] = clark_west(result.actual, result.forecast, result.benchmark)[1]

    portfolio = sum(predictions[n] for n in MATURITIES) / len(MATURITIES)
    row["r2_ew"] = r2_oos(portfolio.actual, portfolio.forecast, portfolio.benchmark)
    row["p_ew"] = clark_west(portfolio.actual, portfolio.forecast, portfolio.benchmark)[1]
    return row, len(portfolio)


def format_table(results):
    out = pd.DataFrame(index=results["model"])
    for key in [str(n) for n in MATURITIES] + ["ew"]:
        r2 = results[f"r2_{key}"].values
        p = results[f"p_{key}"].values
        out[f"xr_{key}"] = [f"{100 * v:.1f}%" for v in r2]
        out[f"p_{key}"] = [
            f"{pv:.3f}" if r2v > 0 and np.isfinite(pv) else "" for r2v, pv in zip(r2, p)
        ]
    return out


def main(gap=1):
    RESULTS_DIR.mkdir(exist_ok=True)
    forwards, excess = load_data()

    rows, n_obs = [], None
    for name, predict, grid in SPECIFICATIONS:
        row, n_obs = run_specification(name, predict, grid, forwards, excess, gap)
        rows.append(row)
        print(
            f"{name:38s} "
            + " ".join(f"{100 * row[f'r2_{n}']:6.1f}%" for n in MATURITIES)
            + f"  EW {100 * row['r2_ew']:6.1f}%"
        )

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_DIR / f"table1_panels_ab_raw_gap{gap}.csv", index=False)
    format_table(results).to_csv(RESULTS_DIR / f"table1_panels_ab_gap{gap}.csv")
    print(f"\nпрогнозов вне выборки: {n_obs} (реализации 1990:01-2018:12)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gap", type=int, default=1)
    main(**vars(parser.parse_args()))
