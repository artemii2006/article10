"""Сравнение воспроизведённой таблицы 1 (панели A и B) с числами из статьи.

Опубликованные значения R²_oos взяты из Table 1 статьи Bianchi, Buchner &
Tamoni (RFS 2021, стр. 47), панели A и B.

Запуск:  python -m src.eval.compare_to_paper
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"

MATURITIES = [2, 3, 4, 5, 7, 10]

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


def main(gap: int = 1) -> None:
    mine = pd.read_csv(RESULTS_DIR / f"table1_panels_ab_raw_gap{gap}.csv").set_index("model")
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
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gap", type=int, default=1)
    main(**vars(parser.parse_args()))
