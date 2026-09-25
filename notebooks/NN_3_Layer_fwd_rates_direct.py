"""
BBT (2020) — Table 2, NN 3 Layer (32, 16, 8), fwd rates direct.

Hybrid-архитектура (Fig. 3, Panel a):
  - macro (N колонок) → MLP(32, 16, 8) → hidden факторы
  - forward rates (10 колонок) → напрямую в выходной слой
  - Output: 6 excess returns

Методология как в статье:
  - 100 seeds, top-10 (App. E.4)
  - epochs=1000, patience=20 (App. E.4)
  - SGD + Nesterov, lr=0.01 (App. E.4, Alg. 5)
  - batch=32, mini-batch
  - BatchNorm после ПОСЛЕДНЕГО ReLU (App. E.4: "we apply batch
    normalization to the activations after the last ReLU layer")
  - He init
  - Train/val split 85/15
  - Y НЕ стандартизуется (как в статье)
  - R²_oos Campbell-Thompson, Clark-West NW lags=11
  - FIRST_ORIGIN=1989-01, GAP=1

FIRST_ORIGIN=1989-01, а не 1990-01: "recursive forecast which starts in
January 1990" в статье относится к дате реализации, а первая ошибка
прогноза (§4.1) сравнивает доходность за февраль 1989 - январь 1990 с
прогнозом, сделанным в январе 1989. Даёт 348 прогнозов вместо 336.

Сетка HYPER_GRID — импровизация: Table F.1 описывает другую
спецификацию (полная CV против group-ensembling из Table C.3) и задаёт
отдельные dropout и штрафы для макро-сети и для сети форвардных ставок,
тогда как здесь они общие.
"""

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

try:
    torch.set_num_threads(2)
except RuntimeError:
    pass

# =============================================================================
# CONFIG
# =============================================================================
ROOT = Path(__file__).resolve().parent
LW_FILE = ROOT / "LW_monthly.xlsx"
MD_FILE = ROOT / "FRED-MD_2018m12.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SAMPLE_START = "1971-08"
SAMPLE_END   = "2018-12"
HOLDING      = 12

FIRST_ORIGIN = pd.Period("1989-01", freq="M")   # прогноз в 1989-01 реализуется в 1990-01
GAP          = 1
MATURITIES   = [2, 3, 4, 5, 7, 10]
CW_LAGS      = 11

FORWARD_HORIZONS = list(range(2, 11))

NN_CFG = dict(
    hidden=(32, 16, 8),
    dropout=0.3,
    weight_decay=1e-3,
    lr=1e-2,
    n_seeds=100,
    top_k=10,
    epochs=1000,
    patience=20,
    batch_size=32,
    train_frac=0.85,
)

HYPER_GRID = [
    dict(dropout=0.1, weight_decay=1e-2),
    dict(dropout=0.1, weight_decay=1e-3),
    dict(dropout=0.1, weight_decay=1e-4),
    dict(dropout=0.3, weight_decay=1e-2),
    dict(dropout=0.3, weight_decay=1e-3),
    dict(dropout=0.3, weight_decay=1e-4),
    dict(dropout=0.5, weight_decay=1e-2),
    dict(dropout=0.5, weight_decay=1e-3),
    dict(dropout=0.5, weight_decay=1e-4),
]
TUNE_EVERY = 60
TUNE_SEEDS = 5

STANDARDIZE_Y = False


# =============================================================================
# DATA — Liu-Wu
# =============================================================================
_MONTH_RE = re.compile(r"^\s*(\d+)\s*[_ ]?\s*m\s*$", re.I)


def _parse_months(col):
    m = _MONTH_RE.match(str(col))
    return int(m.group(1)) if m else None


def _to_yyyymm(v):
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return int(v)
    return int(str(v).strip().split(".")[0])


def load_liu_wu(path: Path) -> pd.DataFrame:
    print(f"[data] Читаю {path.name}...")
    ext = path.suffix.lower()
    if ext in (".xlsx", ".xls"):
        raw = pd.read_excel(path, header=None, dtype=str)
    else:
        raw = pd.read_csv(path, header=None, dtype=str)

    header_row = None
    for i in range(min(30, len(raw))):
        row = raw.iloc[i].astype(str).tolist()
        if sum(_MONTH_RE.match(str(x)) is not None for x in row) >= 5:
            header_row = i
            break
    if header_row is None:
        raise ValueError("Liu-Wu: заголовки не найдены")

    header = raw.iloc[header_row].tolist()
    header[0] = "date"
    data = raw.iloc[header_row + 1:].copy()
    data.columns = header
    data = data.reset_index(drop=True)

    data["date"] = data["date"].map(_to_yyyymm)
    data["date"] = pd.PeriodIndex(data["date"].astype(str), freq="M")
    data = data.set_index("date").sort_index()

    mat_cols = [c for c in data.columns if _parse_months(c) is not None]
    data = data[mat_cols]

    def _to_num(x):
        if isinstance(x, str):
            x = x.replace(",", ".")
        return pd.to_numeric(x, errors="coerce")
    data = data.applymap(_to_num)

    data.columns = [_parse_months(c) / 12.0 for c in data.columns]
    data = data[sorted(data.columns)]

    if data.abs().max().max() > 1.0:
        data = data / 100.0
    print(f"  shape={data.shape}, {data.index[0]}..{data.index[-1]}")
    return data


def build_forward_rates(y: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=y.index)
    out["short_rate"] = y[1.0]
    for n in FORWARD_HORIZONS:
        cur, prev = y[float(n)], y[float(n - 1)]
        out[f"fwd_{n}y"] = n * cur - (n - 1) * prev
    return out


def build_excess_returns(y: pd.DataFrame, mats=MATURITIES, holding=HOLDING):
    out = pd.DataFrame(index=y.index)
    for n in mats:
        y_n, y_n1, y_1 = y[float(n)], y[float(n - 1)], y[1.0]
        future = y_n1.shift(-holding)
        out[f"xr_{n}y"] = -(n - 1) * (future - y_n) + (y_n - y_1)
    return out


# =============================================================================
# DATA — FRED-MD
# =============================================================================
def load_fred_md(path: Path) -> pd.DataFrame:
    print(f"[data] Читаю {path.name} (macro)...")
    raw = pd.read_csv(path, header=None, dtype=str)

    header = raw.iloc[0].tolist()
    header[0] = "date"
    tcodes_raw = raw.iloc[1].tolist()[1:]
    tcodes = {}
    for h, v in zip(header[1:], tcodes_raw):
        try:
            tcodes[h] = int(float(v))
        except Exception:
            tcodes[h] = 1

    data = raw.iloc[2:].copy()
    data.columns = header
    data = data.reset_index(drop=True)
    data["date"] = pd.PeriodIndex(pd.to_datetime(data["date"]), freq="M")
    data = data.set_index("date").sort_index()

    def _t(s, c):
        s = pd.to_numeric(s, errors="coerce")
        if c == 1:  return s
        if c == 2:  return s.diff()
        if c == 3:  return s.diff().diff()
        if c == 4:  return np.log(s.where(s > 1e-6))
        if c == 5:  return np.log(s.where(s > 1e-6)).diff()
        if c == 6:  return np.log(s.where(s > 1e-6)).diff().diff()
        if c == 7:  return s.pct_change(fill_method=None).diff()
        return s

    # pd.concat вместо поштучного присваивания колонок (убирает PerformanceWarning)
    transformed = [_t(data[col], tcodes.get(col, 1)) for col in data.columns]
    out = pd.concat(transformed, axis=1)
    out.columns = data.columns

    # убираем строки с NaT в дате (бывает в некоторых версиях FRED-MD)
    out = out[out.index.notna()]

    print(f"  macro shape={out.shape}, {out.index[0]}..{out.index[-1]}")
    return out


# =============================================================================
# DATA — сборка
# =============================================================================
def load_data():
    print("=" * 60)
    print("Данные")
    print("=" * 60)
    lw = load_liu_wu(LW_FILE)
    lo = pd.Period(SAMPLE_START, freq="M")
    hi = pd.Period(SAMPLE_END, freq="M")
    lw = lw[(lw.index >= lo) & (lw.index <= hi)]

    fwd = build_forward_rates(lw)
    exc = build_excess_returns(lw, MATURITIES, HOLDING)
    md  = load_fred_md(MD_FILE)

    # пересечение дат
    common = fwd.index.intersection(exc.dropna().index).intersection(md.index)
    common = common[(common >= lo) & (common <= hi)]

    # Обрезаем macro по нашим датам
    md_slice = md.loc[common]

    # Ключевой фикс: отбрасываем КОЛОНКИ с любым NaN в этом периоде.
    # Это позволяет сохранить все строки, а не "срезать" всё до 1992-03.
    md_slice = md_slice.dropna(axis=1, how="any")
    dropped = md.shape[1] - md_slice.shape[1]
    print(f"  macro после dropna: {md_slice.shape[1]} колонок "
          f"(отброшено {dropped} с NaN)")

    X_macro = md_slice.loc[common]
    X_fwd   = fwd.loc[common]
    Y       = exc.loc[common]

    # убираем строки, если хоть одна колонка всё ещё NaN
    # (теоретически после dropna по колонкам их не должно быть)
    valid = ~(X_macro.isna().any(axis=1) |
              X_fwd.isna().any(axis=1) |
              Y.isna().any(axis=1))
    X_macro = X_macro[valid]
    X_fwd   = X_fwd[valid]
    Y       = Y[valid]

    print(f"X_macro={X_macro.shape}, X_fwd={X_fwd.shape}, Y={Y.shape}")
    print(f"Период: {X_macro.index[0]}..{X_macro.index[-1]}")
    print(f"Y stats: mean={Y.values.mean():.5f}, std={Y.values.std():.5f}")
    for c in Y.columns:
        print(f"  {c}: mean={Y[c].mean():+.5f}, std={Y[c].std():.5f}")

    return X_macro, X_fwd, Y


# =============================================================================
# NN — Hybrid 3 Layer (32,16,8), fwd rates direct
# =============================================================================
class HybridMLP(nn.Module):
    """
    Fig. 3, Panel (a) BBT:
      - macro → MLP(32, 16, 8) → h (8)
      - [h, fwd_rates] → Linear(8+10, 6) → output
    """
    def __init__(self, d_macro, d_fwd, d_out,
                 hidden=(32, 16, 8), dropout=0.3):
        super().__init__()

        macro_layers, prev = [], d_macro
        for i, h in enumerate(hidden):
            macro_layers += [
                nn.Linear(prev, h),
                nn.ReLU(inplace=True),
            ]
            if i == len(hidden) - 1:
                macro_layers.append(nn.BatchNorm1d(h))
            if dropout > 0:
                macro_layers.append(nn.Dropout(dropout))
            prev = h
        self.macro_net = nn.Sequential(*macro_layers)
        self.hidden_dim = prev

        self.out = nn.Linear(self.hidden_dim + d_fwd, d_out)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_macro, x_fwd):
        h = self.macro_net(x_macro)
        combined = torch.cat([h, x_fwd], dim=1)
        return self.out(combined)


def _train_one(model, Xm_t, Xf_t, Yt, Xm_v, Xf_v, Yv, cfg):
    opt = optim.SGD(model.parameters(),
                    lr=cfg["lr"], momentum=0.9, nesterov=True,
                    weight_decay=cfg["weight_decay"])
    crit = nn.MSELoss()

    bs = cfg.get("batch_size", 32)
    bs = max(2, min(bs, len(Xm_t) // 2))
    ds = TensorDataset(torch.tensor(Xm_t), torch.tensor(Xf_t),
                       torch.tensor(Yt))
    loader = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True)

    Xm_v_t = torch.tensor(Xm_v)
    Xf_v_t = torch.tensor(Xf_v)
    Yv_t   = torch.tensor(Yv)

    best_val, best_state, bad = float("inf"), None, 0
    for _ in range(cfg["epochs"]):
        model.train()
        for xm, xf, yb in loader:
            opt.zero_grad()
            loss = crit(model(xm, xf), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vl = crit(model(Xm_v_t, Xf_v_t), Yv_t).item()
        if vl < best_val - 1e-8:
            best_val, bad = vl, 0
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def train_ensemble(Xm_tr, Xf_tr, Y_tr, cfg, hyper, n_seeds, top_k):
    Xm_tr = np.asarray(Xm_tr, dtype=np.float32)
    Xf_tr = np.asarray(Xf_tr, dtype=np.float32)
    Y_tr  = np.asarray(Y_tr,  dtype=np.float32)

    n = len(Xm_tr)
    n_tr = int(n * cfg["train_frac"])
    tr, va = np.arange(n_tr), np.arange(n_tr, n)

    sc_m = StandardScaler().fit(Xm_tr[tr])
    sc_f = StandardScaler().fit(Xf_tr[tr])

    Xm_t = sc_m.transform(Xm_tr[tr]).astype(np.float32)
    Xm_v = sc_m.transform(Xm_tr[va]).astype(np.float32)
    Xf_t = sc_f.transform(Xf_tr[tr]).astype(np.float32)
    Xf_v = sc_f.transform(Xf_tr[va]).astype(np.float32)

    if STANDARDIZE_Y:
        y_mean = Y_tr[tr].mean(axis=0)
        y_std  = Y_tr[tr].std(axis=0) + 1e-8
        Yt = ((Y_tr[tr] - y_mean) / y_std).astype(np.float32)
        Yv = ((Y_tr[va] - y_mean) / y_std).astype(np.float32)
    else:
        y_mean = np.zeros(Y_tr.shape[1], dtype=np.float64)
        y_std  = np.ones(Y_tr.shape[1], dtype=np.float64)
        Yt = Y_tr[tr].astype(np.float32)
        Yv = Y_tr[va].astype(np.float32)

    d_macro = Xm_t.shape[1]
    d_fwd   = Xf_t.shape[1]
    d_out   = Yt.shape[1]

    models, losses = [], []
    for s in range(n_seeds):
        torch.manual_seed(s); np.random.seed(s)
        m = HybridMLP(d_macro, d_fwd, d_out,
                      hidden=cfg["hidden"], dropout=hyper["dropout"])
        m, vl = _train_one(m, Xm_t, Xf_t, Yt, Xm_v, Xf_v, Yv,
                           {**cfg, "weight_decay": hyper["weight_decay"]})
        models.append(m); losses.append(vl)

    keep = np.argsort(losses)[:top_k]
    return ([models[i] for i in keep],
            sc_m, sc_f, y_mean, y_std, float(min(losses)))


def predict_ensemble(models, Xm_new, Xf_new, sc_m, sc_f, y_mean, y_std):
    Xm = sc_m.transform(np.asarray(Xm_new, dtype=np.float32)).astype(np.float32)
    Xf = sc_f.transform(np.asarray(Xf_new, dtype=np.float32)).astype(np.float32)
    xm_t = torch.tensor(Xm)
    xf_t = torch.tensor(Xf)
    outs = []
    for m in models:
        m.eval()
        with torch.no_grad():
            outs.append(m(xm_t, xf_t).numpy())
    pred = np.mean(outs, axis=0)
    if STANDARDIZE_Y:
        pred = pred * y_std + y_mean
    return pred


def tune_hyperparameters(Xm, Xf, Y, cfg, grid):
    best_loss, best_hyper = np.inf, None
    for hyper in grid:
        try:
            _, _, _, _, _, vl = train_ensemble(
                Xm, Xf, Y, cfg, hyper,
                n_seeds=TUNE_SEEDS, top_k=2)
            if vl < best_loss:
                best_loss, best_hyper = vl, hyper
        except Exception:
            pass
    return best_hyper


# =============================================================================
# METRICS
# =============================================================================
def r2_oos(actual, forecast, benchmark):
    a, f, b = map(np.asarray, (actual, forecast, benchmark))
    ss_res = np.sum((a - f) ** 2)
    ss_tot = np.sum((a - b) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def newey_west_var(x, lags):
    x = np.asarray(x, dtype=float)
    n = len(x)
    e = x - x.mean()
    total = np.dot(e, e) / n
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)
        total += 2.0 * w * np.dot(e[lag:], e[:-lag]) / n
    return total / n


def clark_west(actual, forecast, benchmark, lags=CW_LAGS):
    a, f, b = map(np.asarray, (actual, forecast, benchmark))
    h = (a - b) ** 2 - (a - f) ** 2 + (b - f) ** 2
    var = newey_west_var(h, lags)
    if var <= 0:
        return np.nan, np.nan
    stat = h.mean() / np.sqrt(var)
    return float(stat), float(1.0 - stats.norm.cdf(stat))


# =============================================================================
# FORECAST
# =============================================================================
def forecast_all(X_macro, X_fwd, Y, cfg, first_origin, gap,
                 hyper_grid, tune_every):
    common = X_macro.index.intersection(X_fwd.index).intersection(Y.index)
    Xm_all = X_macro.loc[common]
    Xf_all = X_fwd.loc[common]
    Y_all  = Y.loc[common]

    origins = Xm_all.index[Xm_all.index >= first_origin]
    mat_cols = list(Y_all.columns)
    n_mat = len(mat_cols)

    actual    = np.full((len(origins), n_mat), np.nan)
    forecast  = np.full_like(actual, np.nan)
    benchmark = np.full_like(actual, np.nan)

    current_hyper = dict(dropout=cfg["dropout"],
                         weight_decay=cfg["weight_decay"])
    last_tune_t = -tune_every
    n_tunings = 0
    t_start = time.time()

    for i, origin in enumerate(origins):
        hist_mask = Xm_all.index <= origin - gap
        Xm_hist = Xm_all[hist_mask].values
        Xf_hist = Xf_all[hist_mask].values
        Y_hist  = Y_all[hist_mask].values

        Xm_now = Xm_all.loc[[origin]].values
        Xf_now = Xf_all.loc[[origin]].values

        if len(Xm_hist) < 50:
            continue

        if i - last_tune_t >= tune_every:
            t0 = time.time()
            best = tune_hyperparameters(Xm_hist, Xf_hist, Y_hist,
                                        cfg, hyper_grid)
            if best is not None:
                current_hyper = best
            last_tune_t = i
            n_tunings += 1
            print(f"    [tune #{n_tunings}] t={origin} → "
                  f"dropout={current_hyper['dropout']}, "
                  f"wd={current_hyper['weight_decay']:.0e} "
                  f"({time.time()-t0:.0f}s)")

        ens, sc_m, sc_f, y_mean, y_std, _ = train_ensemble(
            Xm_hist, Xf_hist, Y_hist, cfg, current_hyper,
            n_seeds=cfg["n_seeds"], top_k=cfg["top_k"])

        pred = predict_ensemble(ens, Xm_now, Xf_now, sc_m, sc_f,
                                y_mean, y_std)
        actual[i, :] = Y_all.loc[origin].values
        forecast[i, :] = pred[0]
        benchmark[i, :] = Y_hist.mean(axis=0)

        if (i + 1) % 6 == 0:
            el = time.time() - t_start
            eta = el / (i + 1) * (len(origins) - i - 1)
            print(f"    [{i+1}/{len(origins)}] t={origin} "
                  f"elapsed={el:.0f}s ETA={eta:.0f}s")

    df = pd.DataFrame(index=origins)
    for j, c in enumerate(mat_cols):
        df[f"actual_{c}"]    = actual[:, j]
        df[f"forecast_{c}"]  = forecast[:, j]
        df[f"benchmark_{c}"] = benchmark[:, j]
    return df, mat_cols


def report(df, mat_cols):
    print()
    print("=" * 70)
    print("ИТОГ: R²_oos (%)")
    print("=" * 70)
    print(f"{'maturity':>10s}  {'R²':>8s}  {'p-value':>10s}  {'n_obs':>6s}")
    print("-" * 70)
    for c in mat_cols:
        a = df[f"actual_{c}"].values
        f = df[f"forecast_{c}"].values
        b = df[f"benchmark_{c}"].values
        mask = ~(np.isnan(a) | np.isnan(f) | np.isnan(b))
        r2 = r2_oos(a[mask], f[mask], b[mask])
        _, p = clark_west(a[mask], f[mask], b[mask])
        print(f"{c:>10s}  {100*r2:>+8.2f}  {p:>10.4f}  {mask.sum():>6d}")

    actuals    = np.column_stack([df[f"actual_{c}"].values    for c in mat_cols])
    forecasts  = np.column_stack([df[f"forecast_{c}"].values  for c in mat_cols])
    benchmarks = np.column_stack([df[f"benchmark_{c}"].values for c in mat_cols])
    mask = ~(np.isnan(actuals) | np.isnan(forecasts) | np.isnan(benchmarks))
    rows = mask.all(axis=1)
    r2_ew = r2_oos(actuals[rows].mean(1), forecasts[rows].mean(1),
                   benchmarks[rows].mean(1))
    _, p_ew = clark_west(actuals[rows].mean(1), forecasts[rows].mean(1),
                         benchmarks[rows].mean(1))
    print("-" * 70)
    print(f"{'EW':>10s}  {100*r2_ew:>+8.2f}  {p_ew:>10.4f}  {rows.sum():>6d}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("#" * 70)
    print("# BBT (2020) — Table 2, NN 3 Layer (32, 16, 8), fwd rates direct")
    print(f"# FIRST_ORIGIN={FIRST_ORIGIN}, GAP={GAP}")
    print(f"# STANDARDIZE_Y={STANDARDIZE_Y}")
    print(f"# seeds={NN_CFG['n_seeds']}, top_k={NN_CFG['top_k']}, "
          f"epochs={NN_CFG['epochs']}, patience={NN_CFG['patience']}")
    print(f"# hidden={NN_CFG['hidden']}, dropout={NN_CFG['dropout']}, "
          f"wd={NN_CFG['weight_decay']}")
    print(f"# CV каждые {TUNE_EVERY} мес, grid={len(HYPER_GRID)} combos")
    print("#" * 70)

    t0 = time.time()
    X_macro, X_fwd, Y = load_data()

    df, mat_cols = forecast_all(X_macro, X_fwd, Y, NN_CFG,
                                FIRST_ORIGIN, GAP, HYPER_GRID, TUNE_EVERY)
    df.to_csv(RESULTS_DIR / "bbt_nn_table2_3l32_16_8.csv")
    report(df, mat_cols)

    print()
    print(f"Общее время: {time.time()-t0:.0f}s "
          f"({(time.time()-t0)/60:.1f} мин)")
    print(f"Сохранено: {RESULTS_DIR}/bbt_nn_table2_3l32_16_8.csv")


if __name__ == "__main__":
    main()