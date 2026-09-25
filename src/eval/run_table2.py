"""Таблица 2, панели A и B: прогноз по форвардным ставкам и макропанели.

Спецификации из §4.2 статьи. Фактор Кохрейна-Пьяццези (CP) — подогнанные значения
регрессии средней избыточной доходности на все форвардные ставки; главные компоненты
макропанели тоже оцениваются внутри каждого окна, поэтому заглядывания вперёд нет.

Панель A:
  PCA - first 8 PCs        [F1..F8, CP]
  PCA as in Ludvigson-Ng   [F1, F1^3, F3, F4, F8, CP]
  PLS - 8 components       PLS по [макро, CP]
Панель B, два варианта:
  "using CP factor"        [макро, CP]
  "using fwd rates"        [макро, f1..f10]

Запуск:  python -m src.eval.run_table2
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge, enet_path

from src.eval.backtest import clark_west, r2_oos
from src.eval.run_table1 import FIRST_ORIGIN, GAP, MATURITIES, PROCESSED_DIR, RESULTS_DIR

warnings.filterwarnings("ignore", category=ConvergenceWarning)

VALIDATION_FRACTION = 0.15
RIDGE_GRID = np.logspace(-2, 5, 15)

# Для lasso и elastic net сетку строит сам sklearn: путь начинается со штрафа,
# при котором все коэффициенты равны нулю, и идёт вниз на три порядка. Абсолютные
# значения здесь не годятся, потому что нужный масштаб зависит от разброса цели,
# а он у десятилетних доходностей в шесть раз больше, чем у двухлетних.
N_ALPHAS = 12
ALPHA_EPS = 1e-3


def load_data():
    forwards = pd.read_csv(PROCESSED_DIR / "forward_rates.csv", index_col="date")
    excess = pd.read_csv(PROCESSED_DIR / "excess_returns.csv", index_col="date")
    macro = pd.read_csv(PROCESSED_DIR / "macro_panel.csv", index_col="date")
    for frame in (forwards, excess, macro):
        frame.index = pd.PeriodIndex(frame.index, freq="M")
    return forwards, excess[[f"rx_{n}" for n in MATURITIES]], macro


def cp_factor(F_hist, rx_mean_hist, F_now):
    """Фактор Кохрейна-Пьяццези: подгонка средней доходности на форварды."""
    design = np.column_stack([np.ones(len(F_hist)), F_hist])
    beta, *_ = np.linalg.lstsq(design, rx_mean_hist, rcond=None)
    fitted = design @ beta
    now = np.concatenate([[1.0], F_now.ravel()]) @ beta
    return fitted.reshape(-1, 1), np.array([[now]])


def macro_factors(M_hist, M_now, n_components):
    mean, std = M_hist.mean(0), M_hist.std(0)
    std[std == 0] = 1.0
    pca = PCA(n_components=n_components).fit((M_hist - mean) / std)
    F_hist = pca.transform((M_hist - mean) / std)
    F_now = pca.transform((M_now - mean) / std)
    scale = F_hist.std(0)
    scale[scale == 0] = 1.0
    return F_hist / scale, F_now / scale


def build_design(spec, F_hist, F_now, M_hist, M_now, cp_hist, cp_now):
    if spec == "pca8":
        P_hist, P_now = macro_factors(M_hist, M_now, 8)
        return np.hstack([P_hist, cp_hist]), np.hstack([P_now, cp_now])
    if spec == "ln":
        P_hist, P_now = macro_factors(M_hist, M_now, 8)
        pick = lambda P: np.column_stack([P[:, 0], P[:, 0] ** 3, P[:, 2], P[:, 3], P[:, 7]])
        return np.hstack([pick(P_hist), cp_hist]), np.hstack([pick(P_now), cp_now])
    if spec in ("pls8", "cp"):
        return np.hstack([M_hist, cp_hist]), np.hstack([M_now, cp_now])
    if spec == "fwd":
        return np.hstack([M_hist, F_hist]), np.hstack([M_now, F_now])
    raise ValueError(spec)


def standardize(X_train, X_test):
    mean, std = X_train.mean(0), X_train.std(0)
    std[std == 0] = 1.0
    return (X_train - mean) / std, (X_test - mean) / std


def fit_predict(model_name, X_train, y_train, X_test, alpha=None):
    """X уже стандартизован вызывающей стороной — иначе мы пересчитывали бы
    средние и дисперсии сотню раз на каждое окно."""
    if model_name == "ols":
        model = LinearRegression()
    elif model_name == "pls":
        model = PLSRegression(n_components=8, scale=False)
    elif model_name == "ridge":
        model = Ridge(alpha=alpha)
    elif model_name == "lasso":
        model = Lasso(alpha=alpha, max_iter=10_000)
    elif model_name == "enet":
        model = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=10_000)
    else:
        raise ValueError(model_name)
    model.fit(X_train, y_train)
    return np.ravel(model.predict(X_test))


def choose_alpha(model_name, X_train, y_train, X_val, y_val):
    """Ошибка на валидации вдоль всего пути регуляризации за один вызов."""
    l1_ratio = 1.0 if model_name == "lasso" else 0.5
    centered = y_train - y_train.mean()
    alphas, coefs, _ = enet_path(
        X_train, centered, l1_ratio=l1_ratio, n_alphas=N_ALPHAS, eps=ALPHA_EPS
    )
    predictions = X_val @ coefs + y_train.mean()
    errors = ((y_val[:, None] - predictions) ** 2).mean(axis=0)
    return float(alphas[int(np.argmin(errors))])


SPECIFICATIONS = [
    ("PCA - first 8 PCs", "pca8", "ols", None),
    ("PCA as in Ludvigson and Ng (2009)", "ln", "ols", None),
    ("PLS - 8 components", "pls8", "pls", None),
    ("Ridge (using CP factor)", "cp", "ridge", RIDGE_GRID),
    ("Lasso (using CP factor)", "cp", "lasso", "path"),
    ("Elastic Net (using CP factor)", "cp", "enet", "path"),
    ("Ridge (using fwd rates directly)", "fwd", "ridge", RIDGE_GRID),
    ("Lasso (using fwd rates directly)", "fwd", "lasso", "path"),
    ("Elastic Net (using fwd rates directly)", "fwd", "enet", "path"),
]


def run(forwards, excess, macro):
    common = forwards.index.intersection(macro.index).intersection(excess.dropna().index)
    F, M, Y = forwards.loc[common], macro.loc[common], excess.loc[common]
    origins = common[common >= FIRST_ORIGIN]

    results = {name: {n: ([], [], []) for n in MATURITIES} for name, *_ in SPECIFICATIONS}
    started = time.time()

    for i, origin in enumerate(origins):
        if i and i % 25 == 0:
            done = time.time() - started
            print(
                f"  [{i}/{len(origins)}] {origin}  прошло {done:.0f}с  "
                f"осталось ~{done / i * (len(origins) - i):.0f}с",
                flush=True,
            )
        hist = common <= origin - GAP
        F_hist, F_now = F[hist].values, F.loc[[origin]].values
        M_hist, M_now = M[hist].values, M.loc[[origin]].values
        Y_hist = Y[hist].values
        cp_hist, cp_now = cp_factor(F_hist, Y_hist.mean(1), F_now)

        designs = {
            spec: build_design(spec, F_hist, F_now, M_hist, M_now, cp_hist, cp_now)
            for spec in {s for _, s, _, _ in SPECIFICATIONS}
        }

        n_val = max(1, int(round(VALIDATION_FRACTION * len(Y_hist))))
        scaled = {}
        for spec, (X_hist, X_now) in designs.items():
            full = standardize(X_hist, X_now)
            split = standardize(X_hist[:-n_val], X_hist[-n_val:])
            scaled[spec] = (full, split)

        for name, spec, model, grid in SPECIFICATIONS:
            (Xs_hist, Xs_now), (Xs_train, Xs_val) = scaled[spec]
            for j, n in enumerate(MATURITIES):
                y_hist = Y_hist[:, j]
                if grid is None:
                    pred = fit_predict(model, Xs_hist, y_hist, Xs_now)[0]
                elif isinstance(grid, str):
                    best = choose_alpha(
                        model, Xs_train, y_hist[:-n_val], Xs_val, y_hist[-n_val:]
                    )
                    pred = fit_predict(model, Xs_hist, y_hist, Xs_now, best)[0]
                else:
                    errors = [
                        np.mean(
                            (
                                y_hist[-n_val:]
                                - fit_predict(model, Xs_train, y_hist[:-n_val], Xs_val, alpha)
                            )
                            ** 2
                        )
                        for alpha in grid
                    ]
                    best = grid[int(np.argmin(errors))]
                    pred = fit_predict(model, Xs_hist, y_hist, Xs_now, best)[0]

                a, f, b = results[name][n]
                a.append(Y.loc[origin].values[j])
                f.append(float(pred))
                b.append(float(y_hist.mean()))

    return results, len(origins)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    results, n_obs = run(*load_data())

    rows = []
    for name, *_ in SPECIFICATIONS:
        row = {"model": name}
        stacked = {n: [np.array(v) for v in results[name][n]] for n in MATURITIES}
        for n in MATURITIES:
            a, f, b = stacked[n]
            row[f"r2_{n}"] = r2_oos(a, f, b)
            row[f"p_{n}"] = clark_west(a, f, b)[1]
        ew = [np.mean([stacked[n][k] for n in MATURITIES], axis=0) for k in range(3)]
        row["r2_ew"] = r2_oos(*ew)
        row["p_ew"] = clark_west(*ew)[1]
        rows.append(row)
        print(
            f"{name:40s} "
            + " ".join(f"{100 * row[f'r2_{n}']:6.1f}%" for n in MATURITIES)
            + f"  EW {100 * row['r2_ew']:6.1f}%"
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS_DIR / f"table2_panels_ab_raw_gap{GAP}.csv", index=False)
    print(f"\nпрогнозов вне выборки: {n_obs}")


if __name__ == "__main__":
    main()
