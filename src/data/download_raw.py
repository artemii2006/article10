"""Скачивание сырых данных в data_raw/.

Источники:
  * Liu & Wu (2021), "Reconstructing the Yield Curve", JFE 142(3) — месячные
    непрерывно начисляемые доходности бескупонных облигаций США, сроки 1-360 мес.
    https://sites.google.com/view/jingcynthiawu/yield-data
  * FRED-MD (McCracken & Ng) — месячная макропанель.
    https://www.stlouisfed.org/research/economists/mccracken/fred-databases

Запуск:  python -m src.data.download_raw
"""

from pathlib import Path
import urllib.request

RAW_DIR = Path(__file__).resolve().parents[2] / "data_raw"

LIU_WU_SHEET_ID = "1-wmStGZHLx55dSYi3gQK2vb3F8dMw_Nb"
SOURCES = {
    "liu_wu_yields_monthly.xlsx": (
        f"https://docs.google.com/spreadsheets/d/{LIU_WU_SHEET_ID}/export?format=xlsx"
    ),
    "fred_md_current.csv": (
        "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed"
        "/research/fred-md/monthly/current.csv"
    ),
}


def download(force: bool = False) -> None:
    RAW_DIR.mkdir(exist_ok=True)
    for name, url in SOURCES.items():
        target = RAW_DIR / name
        if target.exists() and not force:
            print(f"пропуск (уже есть): {target.name}")
            continue
        print(f"скачиваю {name} ...")
        urllib.request.urlretrieve(url, target)
        print(f"  сохранено: {target} ({target.stat().st_size / 1e6:.1f} МБ)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="перекачать поверх существующих файлов")
    download(**vars(parser.parse_args()))
