"""Сравнение воспроизведённой таблицы 1 (панели A и B) с числами из статьи.

Опубликованные значения R²_oos взяты из Table 1 статьи Bianchi, Buchner &
Tamoni (RFS 2021, стр. 47), панели A и B.

Запуск:  python -m src.eval.compare_to_paper
"""

from pathlib import Path

import pandas as pd

from src.eval.run_table1 import GAP, MATURITIES

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"

# R²_oos в процентах: rx(2), rx(3), rx(4), rx(5), rx(7), rx(10), равновзвешенный
PUBLISHED = {
    "PCA (10 components)": [-53.0, -36.2, -27.5, -20.3, -12.4, -0.7, -17.7],
    "PCA (5 components)": [-54.9, -38.8, -30.4, -20.3, -15.3, -3.1, -20.0],
    "PCA (3 components)": [-17.6, -10.2, -5.3, 1.0, 2.9, 10.2, 1.0],
    "PCA-Squared (5 components)": [-55.0, -42.1, -32.5, -22.6, -16.4, -5.5, -22.4],
    "PCA-Squared (3 components)": [-46.8, -38.8, -30.6, -21.0, -17.0, -8.9, -22.5],
    "Partial Least Squares (5 components)": [-56.3, -40.5, -33.4, -25.7, -18.6, -9.5, -24.8],
    "Partial Least Squares (3 components)": [-57.9, -40.7, -31.9, -22.9, -14.6, -1.8, -20.2],
    "Ridge": [-40.1, -25.8, -18.9, -11.4, -6.0, 4.6, -9.9],
    "Lasso": [-11.5, -8.1, -2.5, 0.2, 2.3, 9.3, 2.3],
    "Elastic Net": [-10.7, -8.4, -5.1, 0.4, 0.3, 7.0, 0.7],
}


PUBLISHED_TABLE2 = {
    "PCA - first 8 PCs": [-9.8, -2.9, 0.3, 3.0, 3.3, 4.5, 1.8],
    "PCA as in Ludvigson and Ng (2009)": [-3.4, 0.2, 1.6, 1.6, -1.4, -4.7, -1.3],
    "PLS - 8 components": [-40.7, -19.7, -12.0, -8.2, -2.7, 3.4, -6.4],
    "Ridge (using CP factor)": [-45.3, -23.6, -16.7, -13.2, -3.1, 5.3, -5.6],
    "Lasso (using CP factor)": [6.4, 11.2, 12.9, 14.4, 19.6, 23.7, 21.0],
    "Elastic Net (using CP factor)": [6.4, 11.0, 14.3, 15.7, 21.7, 29.1, 22.0],
    "Ridge (using fwd rates directly)": [-52.2, -28.7, -22.7, -18.3, -13.1, -3.5, -15.4],
    "Lasso (using fwd rates directly)": [11.0, 12.0, 12.3, 16.4, 19.9, 23.6, 20.7],
    "Elastic Net (using fwd rates directly)": [10.2, 14.2, 16.0, 13.2, 19.9, 23.6, 21.0],
}


def compare(mine, published, out_name):
    columns = [f"r2_{n}" for n in MATURITIES] + ["r2_ew"]
    labels = [f"rx_{n}" for n in MATURITIES] + ["EW"]
    rows = []
    for model, values in published.items():
        reproduced = (100 * mine.loc[model, columns]).values
        for label, pub, rep in zip(labels, values, reproduced):
            rows.append({"model": model, "target": label, "paper": pub,
                         "reproduced": round(float(rep), 1),
                         "diff": round(float(rep) - pub, 1)})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(RESULTS_DIR / out_name, index=False)
    wide = comparison.pivot(index="model", columns="target", values="diff")
    print(wide[labels].reindex(published.keys()).to_string(float_format=lambda v: f"{v:6.1f}"))
    d = comparison["diff"].abs()
    print(f"медиана |расхождения|: {d.median():.1f} п.п., максимум: {d.max():.1f} "
          f"({comparison.loc[d.idxmax(), 'model']})")
    return comparison


def main():
    mine = pd.read_csv(RESULTS_DIR / f"table1_panels_ab_raw_gap{GAP}.csv").set_index("model")
    columns = [f"r2_{n}" for n in MATURITIES] + ["r2_ew"]
    labels = [f"rx_{n}" for n in MATURITIES] + ["EW"]

    rows = []
    for model, published in PUBLISHED.items():
        reproduced = (100 * mine.loc[model, columns]).values
        for label, pub, rep in zip(labels, published, reproduced):
            rows.append(
                {
                    "model": model,
                    "target": label,
                    "paper": pub,
                    "reproduced": round(float(rep), 1),
                    "diff": round(float(rep) - pub, 1),
                }
            )

    comparison = pd.DataFrame(rows)
    comparison.to_csv(RESULTS_DIR / "table1_comparison.csv", index=False)

    wide = comparison.pivot(index="model", columns="target", values="diff")
    wide = wide[labels].reindex(PUBLISHED.keys())
    print("Расхождение с таблицей статьи, процентные пункты R²_oos:\n")
    print(wide.to_string(float_format=lambda v: f"{v:6.1f}"))

    abs_diff = comparison["diff"].abs()
    print(
        f"\nмедиана |расхождения|: {abs_diff.median():.1f} п.п., "
        f"максимум: {abs_diff.max():.1f} п.п. "
        f"({comparison.loc[abs_diff.idxmax(), 'model']}, "
        f"{comparison.loc[abs_diff.idxmax(), 'target']})"
    )
    print(f"сохранено: {RESULTS_DIR / 'table1_comparison.csv'}")


if __name__ == "__main__":
    main()
