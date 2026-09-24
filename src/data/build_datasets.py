"""Сборка очищенных датасетов в data_processed/.

Строит три файла:
  * forward_rates.csv   — форвардные ставки f^(1)..f^(10) (предикторы Cochrane-Piazzesi)
  * excess_returns.csv  — избыточные доходности rx^(2)..rx^(10) за год владения
  * macro_panel.csv     — панель FRED-MD после трансформаций tcode

Обозначения (все ставки — непрерывно начисляемые, в долях, срок n в годах):
    p_t^(n)  = -n * y_t^(n)                       лог-цена бескупонной облигации
    f_t^(n)  = p_t^(n-1) - p_t^(n)                форвардная ставка, n = 2..10
    f_t^(1)  = y_t^(1)                            однолетняя ставка
    rx_{t+1}^(n) = p_{t+1}^(n-1) - p_t^(n) - y_t^(1)   избыточная доходность

Сдвиг на год = 12 месяцев. В excess_returns.csv строка с датой t содержит
доходность, реализованную за период с t по t+12, то есть цель для прогноза,
сделанного в момент t. Для последних 12 месяцев окна такая доходность
реализовалась бы уже за его пределами, поэтому там стоит NaN: выборка статьи
заканчивается в 2018:12, значит последний прогноз делается в 2017:12.

Запуск:  python -m src.data.build_datasets
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data_raw"
OUT_DIR = ROOT / "data_processed"

# Выборка статьи Bianchi, Buchner & Tamoni (RFS 2021): 10-летние ноты
# выпускаются с сентября 1971, что и определяет начало периода.
DEFAULT_START = "1971-08"
DEFAULT_END = "2018-12"

MAX_MATURITY = 10  # лет
HORIZON = 12  # месяцев в периоде владения


def load_liu_wu_yields() -> pd.DataFrame:
    """Годовые сроки 1..10 лет из файла Liu-Wu, доходности в долях."""
    raw = pd.read_excel(RAW_DIR / "liu_wu_yields_monthly.xlsx", header=None)

    dates = pd.to_numeric(raw[0], errors="coerce")
    data = raw.loc[dates.between(190001, 210012)].copy()
    index = pd.PeriodIndex(data[0].astype(int).astype(str), freq="M")

    # Колонка j соответствует сроку j месяцев; берём 12, 24, ..., 120.
    months = [12 * n for n in range(1, MAX_MATURITY + 1)]
    yields = data[months].astype(float) / 100.0
    yields.index = index
    yields.columns = [f"y_{n}" for n in range(1, MAX_MATURITY + 1)]
    return yields.sort_index()


def build_forward_rates(yields: pd.DataFrame) -> pd.DataFrame:
    """f^(1) = y^(1); f^(n) = p^(n-1) - p^(n) для n = 2..10."""
    log_prices = pd.DataFrame(
        {n: -n * yields[f"y_{n}"] for n in range(1, MAX_MATURITY + 1)},
        index=yields.index,
    )

    forwards = {"f_1": yields["y_1"]}
    for n in range(2, MAX_MATURITY + 1):
        forwards[f"f_{n}"] = log_prices[n - 1] - log_prices[n]
    return pd.DataFrame(forwards, index=yields.index)


def build_excess_returns(yields: pd.DataFrame) -> pd.DataFrame:
    """rx_{t+1}^(n) = p_{t+1}^(n-1) - p_t^(n) - y_t^(1), строка датирована моментом t."""
    log_prices = pd.DataFrame(
        {n: -n * yields[f"y_{n}"] for n in range(1, MAX_MATURITY + 1)},
        index=yields.index,
    )

    excess = {}
    for n in range(2, MAX_MATURITY + 1):
        # Цена через год, когда до погашения остаётся n-1 лет.
        price_next = log_prices[n - 1].shift(-HORIZON)
        excess[f"rx_{n}"] = price_next - log_prices[n] - yields["y_1"]
    return pd.DataFrame(excess, index=yields.index)


def _apply_tcode(series: pd.Series, tcode: int) -> pd.Series:
    """Трансформации FRED-MD (McCracken & Ng, 2016)."""
    if tcode == 1:
        return series
    if tcode == 2:
        return series.diff()
    if tcode == 3:
        return series.diff().diff()
    if tcode == 4:
        return np.log(series)
    if tcode == 5:
        return np.log(series).diff()
    if tcode == 6:
        return np.log(series).diff().diff()
    if tcode == 7:
        return (series / series.shift(1) - 1.0).diff()
    raise ValueError(f"неизвестный tcode: {tcode}")


def build_macro_panel() -> pd.DataFrame:
    """FRED-MD после tcode. Стандартизация НЕ делается: её нужно считать
    только по обучающей части на каждом шаге expanding window."""
    raw = pd.read_csv(RAW_DIR / "fred_md_current.csv")

    tcodes = raw.iloc[0, 1:].astype(float).astype(int)
    data = raw.iloc[1:].copy()
    data = data[data.iloc[:, 0].notna()]

    index = pd.PeriodIndex(pd.to_datetime(data.iloc[:, 0], format="%m/%d/%Y"), freq="M")
    data = data.iloc[:, 1:].astype(float)
    data.index = index

    transformed = pd.DataFrame(
        {name: _apply_tcode(data[name], tcodes[name]) for name in data.columns},
        index=index,
    )
    return transformed.sort_index()


def main(start: str = DEFAULT_START, end: str = DEFAULT_END, balanced: bool = True) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    window = slice(pd.Period(start, freq="M"), pd.Period(end, freq="M"))

    yields = load_liu_wu_yields()
    forwards = build_forward_rates(yields).loc[window]
    excess = build_excess_returns(yields).loc[window]
    macro = build_macro_panel().loc[window]

    # Доходность за год владения, начатый позже чем за 12 месяцев до конца окна,
    # реализуется уже вне выборки — такие цели не используем.
    last_origin = pd.Period(end, freq="M") - HORIZON
    excess.loc[excess.index > last_origin] = np.nan

    if balanced:
        incomplete = macro.columns[macro.isna().any()].tolist()
        if incomplete:
            print(
                "исключены ряды с неполной историей на выборке: "
                + ", ".join(incomplete)
            )
        macro = macro.drop(columns=incomplete)

    outputs = {
        "forward_rates.csv": forwards,
        "excess_returns.csv": excess,
        "macro_panel.csv": macro,
    }
    for name, frame in outputs.items():
        frame.to_csv(OUT_DIR / name, index_label="date")
        n_missing = int(frame.isna().any(axis=1).sum())
        print(
            f"{name}: {frame.shape[0]} строк x {frame.shape[1]} колонок, "
            f"{frame.index[0]}..{frame.index[-1]}, строк с пропусками: {n_missing}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START, help="начало выборки, YYYY-MM")
    parser.add_argument("--end", default=DEFAULT_END, help="конец выборки, YYYY-MM")
    parser.add_argument(
        "--unbalanced",
        dest="balanced",
        action="store_false",
        help="оставить ряды FRED-MD с неполной историей (по умолчанию они исключаются)",
    )
    main(**vars(parser.parse_args()))
