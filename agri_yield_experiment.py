# %% [markdown]
# # Multisource Explainable Crop-Yield Study — complete Kaggle experiment
# **Working title:** *A Multisource Explainable Learning Framework for Robust Crop Yield Prediction under Agro-Climatic Variability*
#
# Implements the protocol in `Precision_Agriculture_Complete_Experiment_Plan.docx` (Phases A–N, Figures 1–10, Tables 1–8, decision gates, reproducibility export).
#
# **Before you run**
# 1. *Add Input* → attach `agriyield-2025` (competition, accept the rules first), **`shaikasif89/wheat-yeild`** (this is D2), `abhinand05/crop-production-in-india` and `siddharthss/crop-recommendation-dataset`. D2 is found by its columns and copied to `/kaggle/working/D2_wheat/train.csv` automatically. To use the `kaggle-cup-2025-01` competition instead, set `CFG["d2_source"]="competition"` and `CFG["target_override"]["D2"]=None`, `CFG["manual_blocks"]["D2"]={}`.
# 2. Settings → Accelerator **GPU T4 x1** (XGBoost, CatBoost and the MLP use it).
# 3. Run **Cell 1 → Cell 3 first** (setup, data discovery, pre-flight time estimate). Check the printed *feature-block assignment* and the *time projection* before launching the long run.
# 4. Set `PROFILE = "smoke"` for a ~5-minute dry run, then `"full"`.
#
# The notebook is **schema-adaptive**: it never hard-codes column names of D1/D2. It infers target, IDs, date, coordinates, and the weather / soil / remote-sensing blocks from names, prints the result, and lets you override via `CFG["manual_blocks"]`.
# Every fold is checkpointed to `/kaggle/working/agri_outputs/ckpt`, so if the session dies you can re-run and it resumes.

# %% 01 — Setup, configuration, reproducibility
import os, sys, json, time, random, warnings, platform, re, math, pickle, gc, shutil, itertools, subprocess, datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sps
from scipy.optimize import nnls
from sklearn.model_selection import KFold, GroupKFold
from sklearn.linear_model import LinearRegression, RidgeCV, ElasticNetCV
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")


def _ensure(pkg, import_name=None):
    try:
        __import__(import_name or pkg)
    except ImportError:
        print(f"Installing {pkg} (needs Internet ON in notebook settings) ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)


for _p in ["xgboost", "lightgbm", "catboost", "optuna", "shap"]:
    _ensure(_p)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
try:
    import shap
except Exception:
    shap = None
try:
    import torch
except Exception:
    torch = None

T0 = time.time()
SEED = 42
random.seed(SEED); np.random.seed(SEED); os.environ["PYTHONHASHSEED"] = str(SEED)
if torch is not None:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def detect_gpu():
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
        return r.returncode == 0 and "GPU" in r.stdout
    except Exception:
        return False


HAS_GPU = detect_gpu()
XGB_V = tuple(int(x) for x in xgb.__version__.split(".")[:2])
N_CPU = os.cpu_count() or 2

# ----------------------------- PROFILES --------------------------------------------
PROFILE = os.environ.get("AGRI_PROFILE", "full")          # "smoke" (dry run) | "full"
PROFILES = {
    "full":  dict(n_est=2000, rf_est=150, rf_max_rows=120_000, trials=15, n_folds=5, t_folds=3,
                  rob_reps=2, mlp_epochs=80, shap_rows=2000, inter_rows=400, perm_rows=1500,
                  spatial_k=10, boot=1000, max_rows=None, cqr_est=400),
    "smoke": dict(n_est=80, rf_est=25, rf_max_rows=5000, trials=3, n_folds=3, t_folds=2,
                  rob_reps=1, mlp_epochs=6, shap_rows=300, inter_rows=80, perm_rows=300,
                  spatial_k=4, boot=200, max_rows=3000, cqr_est=60),
}
P = PROFILES[PROFILE]

CFG = dict(
    seed=SEED,
    profile=PROFILE,
    input_root=os.environ.get("AGRI_INPUT", "/kaggle/input"),
    out_dir=os.environ.get("AGRI_OUT", "/kaggle/working/agri_outputs"),
    datasets=["D1", "D2"],                  # order = priority; D1 primary, D2 independent validation
    d2_source="wheat",                      # "wheat" = Kaggle shaikasif89/wheat-yeild (default) | "competition" = kaggle-cup-2025-01
    paths={"D1": None, "D2": None, "D3": None, "D4": None},   # optional manual overrides (directory for D1/D2, file for D3/D4)
    manual_blocks={"D1": {}, "D2": {"visibility": "weather"}},   # wheat file: keyword matcher would otherwise put 'visibility' under soil     # e.g. {"D1": {"some_col": "soil"}}  blocks: weather|soil|remote|context|drop
    target_override={"D1": None, "D2": "Yield"},                 # wheat file has no sample_submission to infer the target from
    use_coords_as_features=False,           # coordinates are used for spatial splits only (default)
    unassigned_to="soil",                   # block for numeric columns whose names match nothing (printed loudly)
    drop_exact_duplicates=True,
    leak_corr_threshold=0.995,              # |Spearman| with target above this => excluded & flagged
    split_fracs=dict(val=0.10, cal=0.15),   # inside each training partition: early-stopping / conformal calibration
    ood_frac=0.20,                          # S5: most climate-extreme 20% held out
    alpha=0.10,                             # 90% prediction intervals
    time_budget_h=7.0,                      # soft limit for the whole notebook (you have 8 h of quota)
    finalize_reserve_h=0.6,                 # time reserved for analysis/figures/export
    run_mlp=True, run_elasticnet=True,
    robust_splits=["S2_group"],             # splits on which stress tests are run (S3_temporal can be added)
    robust_models=["Ridge", "RF", "XGBoost", "LightGBM", "CatBoost", "MLP", "Ensemble"],
    shap_splits=["S1_random", "S2_group", "S3_temporal", "S5_ood"],
    show_figs=True,
    xgb_device="cuda" if HAS_GPU else "cpu",
    catboost_task="GPU" if HAS_GPU else "CPU",
    **{k: v for k, v in P.items()},
)
OUT = Path(CFG["out_dir"]); (OUT / "ckpt").mkdir(parents=True, exist_ok=True)
for sub in ["figures", "tables", "audit", "repro"]:
    (OUT / sub).mkdir(exist_ok=True)


def restore_checkpoints():
    """Resume after a crashed/committed run: add that run's output as an Input (or keep the same live session) and finished folds are reused."""
    n = 0
    try:
        for p in Path(CFG["input_root"]).rglob("*.pkl"):
            if p.parent.parent.name == "ckpt":
                dst = OUT / "ckpt" / p.parent.name / p.name
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy(p, dst); n += 1
    except Exception as e:
        print("checkpoint restore skipped:", repr(e)[:100])
    have = len(list((OUT / "ckpt").rglob("*.pkl")))
    print(f"Checkpoints: restored {n} from /kaggle/input; {have} fold checkpoint(s) available for resume.")


restore_checkpoints()


def elapsed_h():
    return (time.time() - T0) / 3600


def log(msg):
    print(f"[{elapsed_h():5.2f} h | {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def time_left_h():
    return CFG["time_budget_h"] - elapsed_h()


def save_fig(name, fig=None):
    fig = fig or plt.gcf()
    fig.tight_layout()
    fig.savefig(OUT / "figures" / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "figures" / f"{name}.pdf", bbox_inches="tight")
    if CFG["show_figs"]:
        plt.show()
    plt.close(fig)


def save_table(df, name, index=True):
    df.to_csv(OUT / "tables" / f"{name}.csv", index=index)
    return df


log(f"Profile={PROFILE} | GPU={HAS_GPU} | CPUs={N_CPU} | xgboost {xgb.__version__} | torch={'yes' if torch else 'no'} | shap={'yes' if shap else 'no'}")

# %% 02 — Data discovery (dynamic, no hard-coded paths) and loading
def sanitize(c):
    return re.sub(r"[^0-9a-zA-Z_]+", "_", str(c).strip()).strip("_")


def header_cols(p):
    return [sanitize(c) for c in pd.read_csv(p, nrows=0).columns]


def discover_inputs():
    root = Path(CFG["input_root"])
    assert root.exists(), f"{root} does not exist — attach the datasets via 'Add Input'."
    files = sorted(p for p in root.rglob("*.csv"))
    print("CSV files visible under", root)
    for p in files[:60]:
        print("  ", p)
    found = {"D1": None, "D2": None, "D3": None, "D4": None}
    for p in files:
        if p.name.lower() == "crop_production.csv": found["D3"] = found["D3"] or p
        if p.name.lower() == "crop_recommendation.csv": found["D4"] = found["D4"] or p
    train_dirs = sorted({p.parent for p in files if p.name.lower() == "train.csv"})
    for d in train_dirs:
        cols = {c.lower() for c in header_cols(d / "train.csv")}
        name = str(d).lower()
        is_d2 = ("kaggle-cup" in name or "ml-competition" in name or
                 ("field_id" in cols and any("ndvi" in c for c in cols)))
        is_d1 = "agriyield" in name
        if is_d1 and not found["D1"]: found["D1"] = d
        elif is_d2 and not found["D2"]: found["D2"] = d
    wheat_dirs = set()
    if CFG["d2_source"] == "wheat":           # D2 = Kaggle 'shaikasif89/wheat-yeild' (match by columns, not by file name)
        found["D2"] = None
        for p in files:
            try: cols = {c.lower() for c in header_cols(p)}
            except Exception: continue
            if {"yield", "ndvi", "latitude", "longitude", "dayinseason"} <= cols:
                dst = Path(CFG["out_dir"]).parent / "D2_wheat"; dst.mkdir(parents=True, exist_ok=True)
                shutil.copy(p, dst / "train.csv"); found["D2"] = dst; wheat_dirs.add(p.parent)
                print(f"\nD2 = wheat dataset: {p}  ->  copied to {dst / 'train.csv'}"); break
        if found["D2"] is None:
            print("\n[WARN] D2 (wheat) not found: attach Kaggle dataset 'shaikasif89/wheat-yeild' via Add Input (needs columns Yield, NDVI, Latitude, Longitude, DayInSeason).")
    for d in train_dirs:                      # fallback: first unclaimed train dir -> D1
        if d not in (found["D1"], found["D2"]) and d not in wheat_dirs and not found["D1"]:
            found["D1"] = d
    for k, v in CFG["paths"].items():
        if v: found[k] = Path(v)
    print("\nResolved sources:")
    for k, v in found.items():
        print(f"  {k}: {v}")
    return found


SRC = discover_inputs()


def infer_target(tr, te, ss):
    if ss is not None and ss.shape[1] >= 2:
        for c in ss.columns[1:]:
            if c in tr.columns: return c
    if te is not None:
        diff = [c for c in tr.columns if c not in te.columns]
        if len(diff) == 1: return diff[0]
    for c in tr.columns:
        if c.lower() in ("yield", "target", "yield_kg_ha", "yield_kg_per_ha", "crop_yield"): return c
    raise ValueError("Cannot infer target column — set CFG['target_override'].")


def load_competition(name):
    d = SRC[name]
    assert d is not None, f"{name} not found; attach it or set CFG['paths']['{name}']"
    d = Path(d)
    tr = pd.read_csv(d / "train.csv"); tr.columns = [sanitize(c) for c in tr.columns]
    te = ss = None
    if (d / "test.csv").exists():
        te = pd.read_csv(d / "test.csv"); te.columns = [sanitize(c) for c in te.columns]
    if (d / "sample_submission.csv").exists():
        ss = pd.read_csv(d / "sample_submission.csv"); ss.columns = [sanitize(c) for c in ss.columns]
    target = CFG["target_override"].get(name) or infer_target(tr, te, ss)
    id_col = ss.columns[0] if ss is not None and ss.columns[0] in tr.columns else None
    return tr, te, target, id_col


# %% 03 — Schema inference: target, IDs, date, coordinates, and feature blocks
def toks(c):
    c = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(c))
    return [t for t in re.split(r"[^a-z0-9]+", c.lower()) if t]


RS_SUB = ("ndvi", "ndwi", "savi", "ndre", "ndmi", "vegetation", "reflect", "sentinel", "landsat", "modis",
          "satellite", "chlorophyll", "spectral", "greenness")
RS_TOK = {"evi", "lai", "nir", "swir", "fpar", "nbr", "vi", "msavi", "osavi", "gci", "ci"}
WX_SUB = ("temp", "tmax", "tmin", "rain", "precip", "prcp", "humid", "wind", "radiat", "sunshine", "evap", "vapor",
          "vapour", "vpd", "gdd", "dewpoint", "pressure", "cloud", "weather", "climate", "moist", "drought", "frost", "solar")
WX_TOK = {"rh", "tavg", "t2m", "et", "par", "srad", "pet"}
SOIL_SUB = ("soil", "clay", "sand", "silt", "organic", "carbon", "nitrogen", "phosph", "potass", "cec", "bulk", "texture",
            "nutrient", "elevation", "aspect", "drain", "salin", "zinc", "sulph", "sulf", "calcium", "magnes", "boron",
            "fertil", "manure", "slope")
SOIL_TOK = {"ph", "n", "p", "k", "ec", "om", "soc"}
CTX_TOK = {"month", "day", "doy", "week"}


def classify_block(col):
    t = toks(col); low = col.lower()
    def hit(sub, tk): return any(s in low for s in sub) or any(x in tk for x in t)
    if hit(RS_SUB, RS_TOK): return "remote"
    if hit(WX_SUB, WX_TOK): return "weather"
    if hit(SOIL_SUB, SOIL_TOK): return "soil"
    if any(x in CTX_TOK for x in t): return "context"
    return None


def infer_schema(df, target, id_col, name):
    cols = [c for c in df.columns if c != target]
    S = dict(target=target, id=id_col, field=None, date=None, year=None, lat=None, lon=None, region=None, crop=None)
    for c in cols:
        t = toks(c); low = c.lower()
        if S["field"] is None and (low in ("field_id", "fieldid", "field", "plot_id", "farm_id", "parcel_id") or "field_id" in low): S["field"] = c
        if S["date"] is None and ("date" in t or low.endswith("_date")): S["date"] = c
        if S["year"] is None and t and t[-1] in ("year", "yr") and "date" not in t: S["year"] = c
        if S["lat"] is None and (t and t[0] in ("lat", "latitude") or low in ("lat", "latitude")): S["lat"] = c
        if S["lon"] is None and (t and t[0] in ("lon", "lng", "long", "longitude") or low in ("lon", "lng", "longitude")): S["lon"] = c
        if S["region"] is None and any(x in t for x in ("region", "state", "district", "zone", "county", "province", "village", "location", "site", "area")) \
                and not pd.api.types.is_numeric_dtype(df[c]):
            S["region"] = c
        if S["crop"] is None and any(x in t for x in ("crop", "variety", "cultivar")) and not pd.api.types.is_numeric_dtype(df[c]):
            S["crop"] = c
    return S


def assign_blocks(df, S, name):
    skip = {S["target"], S["id"], S["field"], S["date"], S["year"]}
    if not CFG["use_coords_as_features"]:
        skip |= {S["lat"], S["lon"]}
    blocks = {}
    for c in df.columns:
        if c in skip or c is None: continue
        is_num = pd.api.types.is_numeric_dtype(df[c])
        b = classify_block(c)
        if b is None:
            if S["crop"] == c or not is_num:
                b = "context"                          # crop type / categorical management / unknown categoricals
            elif S["region"] == c:
                b = "context"
            elif c in CFG["manual_blocks"].get(name, {}):
                b = CFG["manual_blocks"][name][c]
            else:
                b = CFG["unassigned_to"]; print(f"  [WARN] {name}: '{c}' matched no keyword -> '{b}'. Override via CFG['manual_blocks'].")
        blocks[c] = b
    for c, b in CFG["manual_blocks"].get(name, {}).items():
        if b == "drop": blocks.pop(c, None)
        else: blocks[c] = b
    if S["region"] and S["region"] in blocks:   # a region label identifies location; keep out of the features if it will define spatial splits
        blocks.pop(S["region"])
    return blocks
# %% 04 — Dataset preparation + Phase A audit (saved to audit/)
class Dataset:
    pass


def add_temporal_summaries(df, S, base_cols):
    """E5 block. CAUSAL expanding-window summaries per field (only observations up to the current date),
    so no row ever sees its own future. Computed from features only (never from the target)."""
    f, d = S["field"], S["date"]
    order = df.sort_values([f, d]).index
    g = df.loc[order].groupby(f, sort=False)
    out = {}
    for c in base_cols:
        x = df.loc[order, c].astype(float)
        gx = x.groupby(df.loc[order, f], sort=False)
        cnt = gx.cumcount() + 1
        mean = gx.expanding().mean().reset_index(level=0, drop=True)
        mx = gx.expanding().max().reset_index(level=0, drop=True)
        mn = gx.expanding().min().reset_index(level=0, drop=True)
        sd = gx.expanding().std().reset_index(level=0, drop=True).fillna(0)
        out[f"ts_{c}_mean"] = mean; out[f"ts_{c}_max"] = mx; out[f"ts_{c}_min"] = mn; out[f"ts_{c}_std"] = sd
        out[f"ts_{c}_range"] = mx - mn
        out[f"ts_{c}_cv"] = sd / (mean.abs() + 1e-9)
        # slope over time (days) via expanding covariance
        tt = (df.loc[order, d] - df.loc[order, d].min()).dt.days.astype(float)
        gt = tt.groupby(df.loc[order, f], sort=False)
        xt = x * tt
        sxy = xt.groupby(df.loc[order, f], sort=False).cumsum(); sx = gx.cumsum(); st = gt.cumsum()
        stt = (tt ** 2).groupby(df.loc[order, f], sort=False).cumsum()
        denom = cnt * stt - st ** 2
        out[f"ts_{c}_slope"] = ((cnt * sxy - sx * st) / denom.replace(0, np.nan)).fillna(0)
        # time (days) since first obs at which running max was reached; running AUC (trapezoid)
        is_max = (x >= mx - 1e-12)
        first_t = tt.groupby(df.loc[order, f], sort=False).transform("min")
        out[f"ts_{c}_t_to_max"] = (tt.where(is_max).groupby(df.loc[order, f], sort=False).ffill() - first_t).fillna(0)
        dtt = gt.diff().fillna(0)
        seg = ((x + gx.shift(1).fillna(x)) / 2) * dtt
        out[f"ts_{c}_auc"] = seg.groupby(df.loc[order, f], sort=False).cumsum()
    ts = pd.DataFrame(out).reindex(df.index)
    return ts.replace([np.inf, -np.inf], np.nan)


def prepare_dataset(name):
    tr, te, target, id_col = load_competition(name)
    ds = Dataset(); ds.name = name
    raw_n = len(tr)
    tr = tr.dropna(subset=[target]).reset_index(drop=True)
    n_dup = int(tr.drop(columns=[id_col] if id_col else []).duplicated().sum())
    if CFG["drop_exact_duplicates"] and n_dup:
        tr = tr.drop(tr.drop(columns=[id_col] if id_col else []).duplicated().to_numpy().nonzero()[0]).reset_index(drop=True)
    if CFG["max_rows"] and len(tr) > CFG["max_rows"]:
        tr = tr.sample(CFG["max_rows"], random_state=SEED).reset_index(drop=True)
    S = infer_schema(tr, target, id_col, name)
    if S["date"]:
        tr[S["date"]] = pd.to_datetime(tr[S["date"]], errors="coerce")
        if tr[S["date"]].notna().mean() < 0.9: S["date"] = None
    if S["date"]:
        tr["doy"] = tr[S["date"]].dt.dayofyear.astype(float); tr["month"] = tr[S["date"]].dt.month.astype(float)
    blocks = assign_blocks(tr, S, name)
    if S["date"]:
        blocks["doy"] = "context"; blocks["month"] = "context"
    y = tr[target].astype(float).to_numpy()

    # ---- group definitions -------------------------------------------------------------------
    grp_field = None
    if S["field"]: grp_field = pd.factorize(tr[S["field"]])[0]
    elif S["lat"] and S["lon"]:
        grp_field = pd.factorize(tr[S["lat"]].round(5).astype(str) + "_" + tr[S["lon"]].round(5).astype(str))[0]
    grp_spatial = None
    if S["lat"] and S["lon"]:
        xy = tr[[S["lat"], S["lon"]]].astype(float)
        xy = xy.fillna(xy.median())
        xy = (xy - xy.mean()) / xy.std().replace(0, 1)
        k = int(min(CFG["spatial_k"], max(2, tr[[S["lat"], S["lon"]]].drop_duplicates().shape[0] // 5)))
        grp_spatial = KMeans(n_clusters=k, n_init=5, random_state=SEED).fit_predict(xy.values)
    elif S["region"]:
        grp_spatial = pd.factorize(tr[S["region"]])[0]
    time_key = None
    if S["date"]: time_key = tr[S["date"]].astype("int64").to_numpy().astype(float)
    elif S["year"]: time_key = tr[S["year"]].astype(float).to_numpy()

    # ---- leakage / audit ---------------------------------------------------------------------
    audit = dict(dataset=name, source=str(SRC[name]), raw_rows=raw_n, rows_used=len(tr), target=target, id=id_col,
                 schema={k: v for k, v in S.items()}, exact_duplicates=n_dup,
                 target_stats=dict(mean=float(np.mean(y)), sd=float(np.std(y)), min=float(np.min(y)), max=float(np.max(y)),
                                   skew=float(sps.skew(y)), q01=float(np.quantile(y, .01)), q99=float(np.quantile(y, .99)),
                                   n_zero=int((y == 0).sum()), n_negative=int((y < 0).sum())))
    miss = tr.isna().mean().sort_values(ascending=False)
    audit["missing_top"] = {k: float(v) for k, v in miss.head(15).items() if v > 0}
    if te is not None:
        for key in [S["field"]]:
            if key and key in te.columns:
                audit["field_overlap_train_test"] = float(te[key].isin(tr[key]).mean())
    if grp_field is not None:
        cnt = pd.Series(grp_field).value_counts()
        audit["obs_per_field"] = dict(n_fields=int(len(cnt)), median=float(cnt.median()), max=int(cnt.max()))
        if cnt.median() > 1:
            wvar = pd.Series(y).groupby(grp_field).std().fillna(0)
            audit["target_within_field_sd_median"] = float(wvar.median())
            audit["target_constant_within_field"] = bool((wvar.median() < 1e-9))
    leak = {}
    num_feats = [c for c, b in blocks.items() if pd.api.types.is_numeric_dtype(tr[c])]
    for c in num_feats:
        r = sps.spearmanr(tr[c], y, nan_policy="omit")[0]
        if np.isfinite(r) and abs(r) >= CFG["leak_corr_threshold"]: leak[c] = float(r)
    audit["suspected_leakage_columns_removed"] = leak
    for c in leak: blocks.pop(c, None)
    const_cols = [c for c in list(blocks) if tr[c].nunique(dropna=True) <= 1]
    audit["constant_columns_removed"] = const_cols
    for c in const_cols: blocks.pop(c, None)
    audit["audit_checklist_hint"] = ("If target is constant within field and fields repeat, S1 random splits are leaky by construction; "
                                      "report S1 separately (H2).")
    # drop heavily-missing columns
    for c in [c for c in list(blocks) if tr[c].isna().mean() > 0.6]:
        blocks.pop(c); audit.setdefault("dropped_gt60pct_missing", []).append(c)

    # ---- temporal summaries (E5) -------------------------------------------------------------
    ts_cols = []
    if S["field"] and S["date"] and grp_field is not None and audit["obs_per_field"]["median"] >= 2:
        rs_like = [c for c, b in blocks.items() if b in ("remote",) and pd.api.types.is_numeric_dtype(tr[c])]
        wx_like = [c for c, b in blocks.items() if b == "weather" and pd.api.types.is_numeric_dtype(tr[c]) and "moist" in c.lower()]
        base = (rs_like + wx_like)[:8]
        if base:
            ts = add_temporal_summaries(tr, S, base)
            tr = pd.concat([tr, ts], axis=1)
            for c in ts.columns: blocks[c] = "temporal"
            ts_cols = list(ts.columns)
    audit["n_temporal_summary_features"] = len(ts_cols)
    json.dump(audit, open(OUT / "audit" / f"audit_{name}.json", "w"), indent=2, default=str)
    pd.Series({c: b for c, b in blocks.items()}, name="block").rename_axis("feature").to_csv(OUT / "audit" / f"feature_blocks_{name}.csv")

    ds.df, ds.S, ds.blocks, ds.y = tr, S, blocks, y
    ds.grp_field, ds.grp_spatial, ds.time_key, ds.audit = grp_field, grp_spatial, time_key, audit
    # climate stress index on whole data (robust z of rain/temperature columns) — used for S5 and stress-level reporting
    wx = [c for c, b in blocks.items() if b == "weather" and pd.api.types.is_numeric_dtype(tr[c])]
    sel = [c for c in wx if is_rain_amount(c) or any(k in c.lower() for k in ("temp", "tmax", "tmin"))] or wx
    if sel:
        z = []
        for c in sel:
            v = tr[c].astype(float); med = v.median(); mad = (v - med).abs().median() * 1.4826 or v.std() or 1.0
            z.append(((v - med) / mad).fillna(0).to_numpy())
        ds.stress = np.sqrt(np.mean(np.square(z), axis=0))
    else:
        ds.stress = None
    return ds


def print_dataset_summary(ds):
    print(f"\n===== {ds.name}: rows={len(ds.df)}  target='{ds.S['target']}'  mean={ds.y.mean():.2f} sd={ds.y.std():.2f} skew={sps.skew(ds.y):.2f}")
    print("Schema:", {k: v for k, v in ds.S.items() if v})
    bl = pd.Series(ds.blocks)
    for b in ["weather", "soil", "remote", "context", "temporal"]:
        cols = bl[bl == b].index.tolist()
        print(f"  [{b:9s}] {len(cols):3d}: {cols[:12]}{' ...' if len(cols) > 12 else ''}")
    a = ds.audit
    print("  duplicates:", a["exact_duplicates"], "| leakage cols:", list(a["suspected_leakage_columns_removed"]), "| missing(top):", dict(list(a["missing_top"].items())[:4]))
    if "obs_per_field" in a: print("  obs/field:", a["obs_per_field"], "| target constant within field:", a.get("target_constant_within_field"))
    print("  group defs: field-groups=", ds.grp_field is not None, "| spatial-groups=", ds.grp_spatial is not None, "| time key=", ds.time_key is not None,
          "| stress index=", ds.stress is not None)


# %% 05 — Feature builder (leak-safe: fitted on the fitting partition only) + preprocessing
RAIN_K = ("rain", "precip", "prcp")
TEMP_K = ("temp", "tmax", "tmin", "tavg", "t2m")
MOIST_K = ("moist",)
HUM_K = ("humid", "rh")


def _has(c, ks):
    low = c.lower(); t = toks(c)
    return any(k in low for k in ks if len(k) > 2) or any(k in t for k in ks if len(k) <= 2)


class FeatureBuilder:
    """raw frame -> frame with base features (median-imputed) + engineered agro-climatic features (E4)."""
    def __init__(self, ds):
        self.S, self.blocks = ds.S, ds.blocks

    def fit(self, raw):
        b = self.blocks
        self.base_num = [c for c, k in b.items() if k in ("weather", "soil", "remote") and pd.api.types.is_numeric_dtype(raw[c])]
        self.ts_cols = [c for c, k in b.items() if k == "temporal"]
        self.med = raw[self.base_num].median().fillna(0)
        self.sd = raw[self.base_num].std().replace(0, 1).fillna(1)
        wx = [c for c in self.base_num if b[c] == "weather"]
        self.rain = [c for c in wx if is_rain_amount(c)]
        self.temp = [c for c in wx if _has(c, TEMP_K)]
        self.moist = [c for c in wx if _has(c, MOIST_K)]
        self.hum = [c for c in wx if _has(c, HUM_K)]
        self.rs = [c for c in self.base_num if b[c] == "remote"]
        self.nit = [c for c in self.base_num if b[c] == "soil" and toks(c) and toks(c)[-1] in ("n", "nitrogen")]
        self.anom_cols = self.rain + self.temp + self.moist + self.rs[:3]
        crop = self.S["crop"]
        self.crop = crop if crop in raw.columns else None
        filled = raw[self.base_num].fillna(self.med)
        if self.crop:
            g = filled[self.anom_cols].groupby(raw[self.crop].astype(str))
            self.gmean, self.gstd = g.mean(), g.std().replace(0, np.nan)
        self.gm = filled[self.anom_cols].mean(); self.gs = filled[self.anom_cols].std().replace(0, 1).fillna(1)
        self.q90 = {c: float(filled[c].quantile(0.9)) for c in self.temp}
        # missing indicators for columns with >=1% (and <60%) missingness in the fitting data
        self.miss_ind = [c for c in self.base_num if 0.01 <= raw[c].isna().mean() < 0.6]
        return self

    def _anom(self, filled, raw, c):
        if self.crop is not None:
            key = raw[self.crop].astype(str)
            m = key.map(self.gmean[c]).fillna(self.gm[c]); s = key.map(self.gstd[c]).fillna(self.gs[c])
        else:
            m, s = self.gm[c], self.gs[c]
        s = s.replace(0, 1) if hasattr(s, 'replace') else (s or 1.0)
        return (filled[c] - m) / s

    def transform(self, raw):
        out = {}
        filled = raw[self.base_num].astype(float).fillna(self.med)
        for c in self.base_num: out[c] = filled[c]
        for c, k in self.blocks.items():
            if k in ("context",) and c in raw.columns: out[c] = raw[c]
        for c in self.ts_cols:
            if c in raw.columns: out[c] = raw[c]
        for c in self.miss_ind:
            out[f"{c}__isna"] = raw[c].isna().astype(float)
        for c in self.anom_cols:
            out[f"{c}__anom"] = self._anom(filled, raw, c)
        for c in self.temp:
            out[f"{c}__heat"] = (filled[c] - self.q90[c]).clip(lower=0)
        r0 = self.rain[0] if self.rain else None; t0 = self.temp[0] if self.temp else None
        m0 = self.moist[0] if self.moist else None; h0 = self.hum[0] if self.hum else None
        v0 = self.rs[0] if self.rs else None
        if r0 and m0: out["x_rain_x_moist"] = filled[r0] * filled[m0]
        if t0 and h0: out["x_temp_x_hum"] = filled[t0] * filled[h0]
        if t0 and m0: out["x_temp_x_moist"] = filled[t0] * filled[m0]
        if v0 and r0: out["x_rs_x_rain"] = filled[v0] * filled[r0]
        if self.nit and r0: out["x_N_x_rain"] = filled[self.nit[0]] * filled[r0]
        za = [out[f"{c}__anom"] for c in (self.rain + self.temp)]
        if za:
            out["stress_idx"] = np.sqrt(np.mean(np.square(np.vstack([z.to_numpy() for z in za])), axis=0))
            if self.rain and self.temp:
                out["dry_hot"] = ((out[f"{self.rain[0]}__anom"] < -0.5) & (out[f"{self.temp[0]}__anom"] > 0.5)).astype(float)
        return pd.DataFrame(out, index=raw.index)

    def block_of(self, col):
        if col in self.blocks: return self.blocks[col]
        if col.endswith("__isna"): return self.blocks.get(col[:-6], "context")
        return "engineered"


def is_rain_amount(c):
    """precipitation AMOUNT columns only (excludes probabilities and rain/snow type flags such as precipProbability, precipTypeIsRain)"""
    return _has(c, RAIN_K) and not any(x in c.lower() for x in ("prob", "type", "flag", "is_"))


def feature_sets(fb, cols):
    """E1..E5 as cumulative column lists (duplicates are dropped by the caller)."""
    blk = {c: fb.block_of(c) for c in cols}
    ctx = [c for c in cols if blk[c] == "context"]
    def sel(*bs): return [c for c in cols if blk[c] in bs]
    sets = {
        "E1": sel("weather", "context"),
        "E2": sel("weather", "soil", "context"),
        "E3": sel("weather", "soil", "remote", "context"),
        "E4": sel("weather", "soil", "remote", "context", "engineered"),
        "E5": sel("weather", "soil", "remote", "context", "engineered", "temporal"),
    }
    out, prev = {}, None
    for k, v in sets.items():
        if len(v) == 0 or (prev is not None and v == prev): continue
        out[k] = v; prev = v
    return out


class Prep:
    """median-impute numerics, one-hot categoricals (top-20 levels). Returns raw-unit matrix and a standardised view."""
    def fit(self, X):
        self.num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        self.cat = [c for c in X.columns if c not in self.num]
        self.med = X[self.num].median().fillna(0)
        self.levels = {c: list(X[c].astype(str).value_counts().index[:20]) for c in self.cat}
        Xr = self._raw(X)
        self.mu = Xr.mean(0); self.sd = Xr.std(0); self.sd[self.sd < 1e-9] = 1.0
        return self

    def _raw(self, X):
        parts = [X[self.num].astype(float).fillna(self.med).to_numpy(dtype=np.float32)]
        self.names = list(self.num)
        for c in self.cat:
            s = X[c].astype(str).to_numpy()
            for v in self.levels[c]:
                parts.append((s == v).astype(np.float32)[:, None]); self.names.append(f"{c}={v}")
        return np.hstack(parts)

    def transform(self, X):
        Xr = self._raw(X)
        return Xr, ((Xr - self.mu) / self.sd).astype(np.float32)


# %% 06 — Phase D: leakage-safe split definitions (S1..S5) with inner fit/val/calibration partitions
def make_splits(ds):
    n = len(ds.df); idx = np.arange(n); rng = np.random.RandomState(SEED); splits, notes = {}, {}
    k = CFG["n_folds"]
    splits["S1_random"] = [(a, b) for a, b in KFold(k, shuffle=True, random_state=SEED).split(idx)]
    if ds.grp_field is not None and len(np.unique(ds.grp_field)) >= k * 2:
        perm = rng.permutation(ds.grp_field.max() + 1)[ds.grp_field]      # random fold assignment of whole fields
        splits["S2_group"] = [(a, b) for a, b in GroupKFold(k).split(idx, groups=perm)]
    else:
        notes["S2_group"] = "unavailable: no field identifier / too few fields"
    if ds.time_key is not None and len(np.unique(ds.time_key)) >= 4:
        tk = ds.time_key; cuts = np.linspace(0.5, 1.0, CFG["t_folds"] + 1); folds = []
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            a_, b_ = np.quantile(tk, lo), np.quantile(tk, hi)
            tr_i = idx[tk <= a_]; te_i = idx[(tk > a_) & (tk <= b_)] if hi < 1 else idx[tk > a_]
            if len(tr_i) >= 100 and len(te_i) >= 30: folds.append((tr_i, te_i))
        if folds: splits["S3_temporal"] = folds
        else: notes["S3_temporal"] = "unavailable: time key too coarse"
    else:
        notes["S3_temporal"] = "unavailable: no date/year column"
    if ds.grp_spatial is not None and len(np.unique(ds.grp_spatial)) >= 3:
        ng = len(np.unique(ds.grp_spatial)); kk = min(k, ng)
        perm = rng.permutation(ng)[pd.factorize(ds.grp_spatial)[0]]
        splits["S4_spatial"] = [(a, b) for a, b in GroupKFold(kk).split(idx, groups=perm)]
    else:
        notes["S4_spatial"] = "unavailable: no coordinates / region column"
    if ds.stress is not None:
        thr = np.quantile(ds.stress, 1 - CFG["ood_frac"])
        te_i = idx[ds.stress >= thr]; tr_i = idx[ds.stress < thr]
        if ds.grp_field is not None:                      # keep fields intact: remove from train any field present in the OOD test
            bad = np.isin(ds.grp_field[tr_i], np.unique(ds.grp_field[te_i])); tr_i = tr_i[~bad]
        if len(tr_i) >= 100 and len(te_i) >= 30: splits["S5_ood"] = [(tr_i, te_i)]
    else:
        notes["S5_ood"] = "unavailable: no weather columns for a stress index"
    return splits, notes


def partition_train(train_idx, split_name, ds, seed):
    """Inner partition of a training set -> fit / val (early stopping) / cal (conformal). Mirrors the outer split logic."""
    rng = np.random.RandomState(seed); fv, fc = CFG["split_fracs"]["val"], CFG["split_fracs"]["cal"]
    n = len(train_idx)
    if split_name == "S3_temporal":
        order = train_idx[np.argsort(ds.time_key[train_idx], kind="stable")]
        a, b = int(n * (1 - fv - fc)), int(n * (1 - fc))
        return order[:a], order[a:b], order[b:]
    grp = None
    if split_name == "S2_group": grp = ds.grp_field[train_idx]
    elif split_name == "S4_spatial":
        grp = ds.grp_spatial[train_idx]
    elif split_name == "S5_ood" and ds.grp_field is not None: grp = ds.grp_field[train_idx]
    if grp is not None and len(np.unique(grp)) >= 6:
        ug = rng.permutation(np.unique(grp)); sizes = pd.Series(grp).value_counts().reindex(ug).to_numpy(); cs = np.cumsum(sizes) / n
        g_fit = ug[cs <= 1 - fv - fc]; g_val = ug[(cs > 1 - fv - fc) & (cs <= 1 - fc)]; g_cal = ug[cs > 1 - fc]
        if len(g_fit) and len(g_val) and len(g_cal):
            return train_idx[np.isin(grp, g_fit)], train_idx[np.isin(grp, g_val)], train_idx[np.isin(grp, g_cal)]
    perm = rng.permutation(train_idx); a, b = int(n * (1 - fv - fc)), int(n * (1 - fc))
    return perm[:a], perm[a:b], perm[b:]
# %% 07 — Models (Phase E), metrics (Phase H), ensemble, Optuna (Phase G), conformal helpers (Phase I), perturbations (Phase J)
def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def reg_metrics(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    r = rmse(y, p); mae = float(np.mean(np.abs(y - p)))
    ss_res = float(np.sum((y - p) ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
    return dict(RMSE=r, MAE=mae, R2=(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
                NRMSE=r / abs(y.mean()) if y.mean() != 0 else np.nan)


class W:                                    # prediction wrapper: some models use the standardised matrix, others the raw-unit one
    def __init__(self, est, kind): self.est, self.kind = est, kind
    def predict(self, Xr, Xs): return np.asarray(self.est.predict(Xs if self.kind == "s" else Xr), dtype=np.float64)


class TorchMLP:
    def __init__(self, epochs, seed, hidden=(256, 128, 64), lr=2e-3, wd=1e-4, drop=0.1, bs=1024, patience=10):
        self.epochs, self.seed, self.h, self.lr, self.wd, self.drop, self.bs, self.pat = epochs, seed, hidden, lr, wd, drop, bs, patience
        self.dev = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"

    def fit(self, X, y, Xv, yv):
        torch.manual_seed(self.seed)
        self.ym, self.ys = float(y.mean()), float(y.std() or 1.0)
        Xt = torch.tensor(X, device=self.dev); yt = torch.tensor(((y - self.ym) / self.ys).astype(np.float32), device=self.dev)
        Xvt = torch.tensor(Xv, device=self.dev); yvt = torch.tensor(((yv - self.ym) / self.ys).astype(np.float32), device=self.dev)
        layers, d = [], X.shape[1]
        for h in self.h: layers += [torch.nn.Linear(d, h), torch.nn.ReLU(), torch.nn.Dropout(self.drop)]; d = h
        layers.append(torch.nn.Linear(d, 1))
        self.net = torch.nn.Sequential(*layers).to(self.dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
        best, best_state, bad, n = 1e18, None, 0, len(Xt)
        for ep in range(self.epochs):
            self.net.train(); perm = torch.randperm(n, device=self.dev)
            for i in range(0, n, self.bs):
                b = perm[i:i + self.bs]; opt.zero_grad()
                loss = torch.nn.functional.mse_loss(self.net(Xt[b]).squeeze(1), yt[b]); loss.backward(); opt.step()
            self.net.eval()
            with torch.no_grad(): v = torch.nn.functional.mse_loss(self.net(Xvt).squeeze(1), yvt).item()
            sched.step(v)
            if v < best - 1e-5: best, bad, best_state = v, 0, {k: t.detach().clone() for k, t in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.pat: break
        if best_state is not None: self.net.load_state_dict(best_state)
        return self

    def predict(self, X):
        self.net.eval()
        with torch.no_grad(): out = self.net(torch.tensor(X, device=self.dev)).squeeze(1).cpu().numpy()
        return out * self.ys + self.ym


def xgb_kwargs(**over):
    kw = dict(n_estimators=CFG["n_est"], learning_rate=0.03, max_depth=8, subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
              objective="reg:squarederror", random_state=SEED, tree_method="hist", early_stopping_rounds=50, n_jobs=N_CPU)
    if XGB_V >= (2, 0): kw["device"] = CFG["xgb_device"]
    elif CFG["xgb_device"] == "cuda": kw["tree_method"] = "gpu_hist"
    kw.update(over); return kw


MODEL_ORDER = ["Mean", "Linear", "Ridge", "ElasticNet", "RF", "ExtraTrees", "XGBoost", "LightGBM", "CatBoost", "MLP"]
ENS_MEMBERS = ["Ridge", "RF", "ExtraTrees", "XGBoost", "LightGBM", "CatBoost", "MLP"]


def active_models():
    m = list(MODEL_ORDER)
    if not CFG["run_elasticnet"]: m.remove("ElasticNet")
    if not (CFG["run_mlp"] and torch is not None): m.remove("MLP")
    return m


FAILED = {}                                  # model name -> number of folds where it failed (reported at the end)


def _fit_one(nm, Xr, Xs, y, Xrv, Xsv, yv, seed, sub):
    if nm == "Mean": return DummyRegressor().fit(Xr, y), "r"
    if nm == "Linear": return LinearRegression().fit(Xs.astype(np.float64), y), "s"
    if nm == "Ridge": return RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Xs.astype(np.float64), y), "s"
    if nm == "ElasticNet":
        # float64 + precompute=False: float32 rounding made sklearn's Gram-matrix check fail on the Kaggle Python 3.13 image
        return ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8], n_alphas=25, cv=3, max_iter=1500, precompute=False, n_jobs=1,
                            random_state=seed).fit(Xs.astype(np.float64), y), "s"
    if nm == "RF":
        return RandomForestRegressor(n_estimators=CFG["rf_est"], max_features=0.33, min_samples_leaf=2, n_jobs=-1, random_state=seed).fit(Xr[sub], y[sub]), "r"
    if nm == "ExtraTrees":
        return ExtraTreesRegressor(n_estimators=CFG["rf_est"], max_features=0.7, min_samples_leaf=2, n_jobs=-1, random_state=seed).fit(Xr[sub], y[sub]), "r"
    if nm == "XGBoost":
        est = xgb.XGBRegressor(**xgb_kwargs(random_state=seed)); est.fit(Xr, y, eval_set=[(Xrv, yv)], verbose=False); return est, "r"
    if nm == "LightGBM":
        est = lgb.LGBMRegressor(n_estimators=CFG["n_est"], learning_rate=0.03, num_leaves=31, subsample=0.8, subsample_freq=1,
                                colsample_bytree=0.8, min_child_samples=10, random_state=seed, n_jobs=N_CPU, verbose=-1)
        est.fit(Xr, y, eval_set=[(Xrv, yv)], callbacks=[lgb.early_stopping(50, verbose=False)]); return est, "r"
    if nm == "CatBoost":
        est = CatBoostRegressor(iterations=CFG["n_est"], depth=8, learning_rate=0.03, loss_function="RMSE", verbose=False, random_seed=seed,
                                task_type=CFG["catboost_task"], early_stopping_rounds=50, thread_count=N_CPU)
        est.fit(Xr, y, eval_set=(Xrv, yv), use_best_model=True); return est, "r"
    if nm == "MLP":
        return TorchMLP(CFG["mlp_epochs"], seed).fit(Xs, y, Xsv, yv), "s"
    raise ValueError(nm)


def fit_models(Xr, Xs, y, Xrv, Xsv, yv, seed, names=None):
    """A model that raises is logged and skipped for that fold (never aborts a multi-hour run). Failures are summarised at the end."""
    names = names or active_models(); out, times = {}, {}
    cap = CFG["rf_max_rows"]
    sub = np.random.RandomState(seed).choice(len(y), cap, replace=False) if len(y) > cap else slice(None)
    for nm in names:
        t = time.time()
        try:
            est, kind = _fit_one(nm, Xr, Xs, y, Xrv, Xsv, yv, seed, sub)
            out[nm] = W(est, kind); times[nm] = time.time() - t
        except Exception as e:
            FAILED[nm] = FAILED.get(nm, 0) + 1
            print(f"  [WARN] model {nm} failed and was skipped for this fold: {type(e).__name__}: {str(e)[:160]}", flush=True)
            if nm in ("XGBoost", "Mean") : raise                 # core models must not fail silently
    return out, times


def ens_weights(pred_val, y_val, members):
    members = [m for m in members if m in pred_val]
    A = np.column_stack([pred_val[m] for m in members])
    w, _ = nnls(A, y_val)
    w = w / w.sum() if w.sum() > 0 else np.ones(len(members)) / len(members)
    return dict(zip(members, w))


def ens_predict(preds, w): return sum(w[m] * preds[m] for m in w)


# ---------- Optuna (Phase G): tuned ONLY on the fitting/validation partitions of each outer fold (nested; test never seen) ----------
def tune_xgb(Xr, y, Xrv, yv, seed, timeout_s):
    def objective(trial):
        p = dict(max_depth=trial.suggest_int("max_depth", 3, 12), learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                 subsample=trial.suggest_float("subsample", 0.6, 1.0), colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                 min_child_weight=trial.suggest_int("min_child_weight", 1, 20), reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 30, log=True),
                 n_estimators=min(CFG["n_est"], 1500))
        m = xgb.XGBRegressor(**xgb_kwargs(random_state=seed, early_stopping_rounds=30, **p))
        m.fit(Xr, y, eval_set=[(Xrv, yv)], verbose=False)
        return rmse(yv, m.predict(Xrv))
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=CFG["trials"], timeout=timeout_s)
    return study.best_params, float(study.best_value), len(study.trials)


# ---------- conformal (Phase I) ----------
def conf_q(scores, alpha):
    n = len(scores); level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def interval_stats(y, lo, hi, alpha):
    cover = (y >= lo) & (y <= hi); w = hi - lo
    return dict(PICP=float(cover.mean()), MPIW=float(w.mean()), nominal=1 - alpha,
                winkler=float(np.mean(w + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi))))


def split_conformal(pred_cal, y_cal, pred_te, alpha):
    q = conf_q(np.abs(y_cal - pred_cal), alpha); return pred_te - q, pred_te + q


def norm_conformal(pred_cal, sig_cal, y_cal, pred_te, sig_te, alpha):
    q = conf_q(np.abs(y_cal - pred_cal) / sig_cal, alpha); return pred_te - q * sig_te, pred_te + q * sig_te


def cqr_calibrate(lo_c, hi_c, y_c, lo_t, hi_t, alpha):
    q = conf_q(np.maximum(lo_c - y_c, y_c - hi_c), alpha); return lo_t - q, hi_t + q


def member_spread(preds, names):
    return np.std(np.column_stack([preds[m] for m in names]), axis=1)


# ---------- perturbations (Phase J) ----------
def build_perturbations(raw_te, fb, seed, reps):
    rng = np.random.RandomState(seed); out = [("clean", 0.0, 0, raw_te)]
    base = [c for c in fb.base_num if raw_te[c].nunique() > 2]
    for p in (0.05, 0.10, 0.20, 0.30):
        for r in range(reps):
            d = raw_te.copy(); mask = rng.rand(len(d), len(fb.base_num)) < p
            vals = np.array(d[fb.base_num].astype(float), dtype=float); vals[mask] = np.nan; d[fb.base_num] = vals; out.append(("missing", p, r, d))
    for p in (0.05, 0.10, 0.15):
        for r in range(reps):
            d = raw_te.copy()
            for c in base: d[c] = d[c].astype(float) + rng.normal(0, p * fb.sd[c], len(d))
            out.append(("noise", p, r, d))
    if fb.rain:
        for lv in (-0.1, -0.2, -0.3):
            d = raw_te.copy()
            for c in fb.rain: d[c] = d[c].astype(float) * (1 + lv)
            out.append(("rain", lv, 0, d))
    if fb.temp:
        for lv in (1.0, 2.0, 3.0):
            d = raw_te.copy()
            for c in fb.temp: d[c] = d[c].astype(float) + lv
            out.append(("temp", lv, 0, d))
    if fb.rain and fb.temp:
        for lr_, lt in ((-0.1, 1.0), (-0.2, 2.0), (-0.3, 3.0)):
            d = raw_te.copy()
            for c in fb.rain: d[c] = d[c].astype(float) * (1 + lr_)
            for c in fb.temp: d[c] = d[c].astype(float) + lt
            out.append(("combined", lt, 0, d))              # level = +dT; paired rainfall change = -10%*dT
    return out
# %% 08 — Core experiment engine: one fold = E1..E5 × all models + ensemble + Optuna + conformal inputs + stress tests + SHAP inputs
SPLIT_ORDER = ["S2_group", "S3_temporal", "S4_spatial", "S5_ood", "S1_random"]    # priority order if the time budget runs out
PARTS = ("val", "cal", "test")


def run_fold(ds, split_name, k, tr, te, deadline_h):
    ck = OUT / "ckpt" / ds.name / f"{split_name}_{k}.pkl"
    ck.parent.mkdir(parents=True, exist_ok=True)
    sig = f"{CFG['profile']}|{CFG['n_est']}|{CFG['rf_est']}|{CFG['trials']}|{CFG['n_folds']}|{len(ds.df)}|{sorted(CFG['robust_models'])}|{CFG['run_mlp']}"
    if ck.exists():
        try:
            c = pickle.load(open(ck, "rb"))
            if c.get("sig") == sig: return c
            print(f"  checkpoint {ck.name} was made with a different configuration/data → recomputing")
        except Exception:
            print(f"  checkpoint {ck.name} unreadable → recomputing")
    t_fold = time.time(); seed = SEED + k; raw = ds.df
    fit_i, val_i, cal_i = partition_train(tr, split_name, ds, seed)
    fb = FeatureBuilder(ds).fit(raw.iloc[fit_i])
    R = {"fit": raw.iloc[fit_i], "val": raw.iloc[val_i], "cal": raw.iloc[cal_i], "test": raw.iloc[te]}
    F = {nm: fb.transform(v) for nm, v in R.items()}
    y = {"fit": ds.y[fit_i], "val": ds.y[val_i], "cal": ds.y[cal_i], "test": ds.y[te]}
    cols = list(F["fit"].columns); fsets = feature_sets(fb, cols); final = list(fsets)[-1]
    chunk = dict(dataset=ds.name, split=split_name, fold=k, fsets={a: b for a, b in fsets.items()}, final=final,
                 test_idx=te, cal_idx=cal_i, val_idx=val_i, fit_idx=fit_i, metrics=[], preds={}, cqr={}, tuned={}, ens_w={}, robust=[], shap=None)
    do_rob = split_name in CFG["robust_splits"]
    perts = build_perturbations(R["test"], fb, seed, CFG["rob_reps"]) if do_rob else []
    pert_F = [fb.transform(p[3]) for p in perts]
    for fs, fcols in fsets.items():
        prep = Prep().fit(F["fit"][fcols]); M = {nm: prep.transform(F[nm][fcols]) for nm in F}
        models, times = fit_models(M["fit"][0], M["fit"][1], y["fit"], M["val"][0], M["val"][1], y["val"], seed)
        preds = {nm: {pt: models[nm].predict(*M[pt]) for pt in PARTS} for nm in models}
        w = ens_weights({m: preds[m]["val"] for m in ENS_MEMBERS if m in preds}, y["val"], ENS_MEMBERS)
        preds["Ensemble"] = {pt: ens_predict({m: preds[m][pt] for m in w}, w) for pt in PARTS}; chunk["ens_w"][fs] = w
        if fs == final and CFG["trials"] > 0 and "XGBoost" in models:
            t = time.time(); left_s = max(30.0, (deadline_h - elapsed_h()) * 3600 * 0.08)
            bp, bv, nt = tune_xgb(M["fit"][0], y["fit"], M["val"][0], y["val"], seed, min(900.0, left_s))
            est = xgb.XGBRegressor(**xgb_kwargs(random_state=seed, **{**bp, "n_estimators": min(CFG["n_est"], 1500)}))
            est.fit(M["fit"][0], y["fit"], eval_set=[(M["val"][0], y["val"])], verbose=False)
            models["XGB_Tuned"] = W(est, "r"); times["XGB_Tuned"] = time.time() - t
            try: est.save_model(str(ck.with_name(f"{split_name}_{k}_xgb_tuned.json")))
            except Exception: pass
            preds["XGB_Tuned"] = {pt: models["XGB_Tuned"].predict(*M[pt]) for pt in PARTS}
            chunk["tuned"] = dict(params=bp, val_rmse=bv, n_trials=nt, seed=seed, best_iteration=int(getattr(est, "best_iteration", -1)))
        for nm, pr in preds.items():
            chunk["metrics"].append(dict(dataset=ds.name, split=split_name, fold=k, fset=fs, model=nm, n_features=len(prep.names),
                                         n_fit=len(fit_i), n_test=len(te), fit_s=times.get(nm, np.nan), **reg_metrics(y["test"], pr["test"])))
        chunk["preds"].update({(fs, nm): {pt: v.astype(np.float32) for pt, v in pr.items()} for nm, pr in preds.items()})
        # ---- quantile-regression intervals (CQR) on the final feature set ----
        if fs == final:
            qa = CFG["alpha"]; q = {}
            for tag, a in (("lo", qa / 2), ("hi", 1 - qa / 2)):
                m = lgb.LGBMRegressor(objective="quantile", alpha=a, n_estimators=CFG["cqr_est"], learning_rate=0.05, num_leaves=31,
                                      min_child_samples=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, n_jobs=N_CPU, verbose=-1)
                m.fit(M["fit"][0], y["fit"]); q[tag] = {pt: m.predict(M[pt][0]) for pt in ("cal", "test")}
            chunk["cqr"][fs] = {f"{tag}_{pt}": np.minimum(q["lo"][pt], q["hi"][pt]) if tag == "lo" else np.maximum(q["lo"][pt], q["hi"][pt])
                                for tag in ("lo", "hi") for pt in ("cal", "test")}
        # ---- Phase J stress tests ----
        if do_rob and fs != "E5":
            rob = [m for m in CFG["robust_models"] if m in models or m == "Ensemble"]
            need = {m for m in rob if m != "Ensemble"} | (set(w) if "Ensemble" in rob else set())
            for (kind, level, rep, _), Fp in zip(perts, pert_F):
                Xr, Xs = prep.transform(Fp[fcols]); pp = {m: models[m].predict(Xr, Xs) for m in need if m in models}
                if "Ensemble" in rob: pp["Ensemble"] = ens_predict(pp, w)
                for m in rob:
                    chunk["robust"].append(dict(dataset=ds.name, split=split_name, fold=k, fset=fs, model=m, kind=kind, level=level, rep=rep,
                                                **{a: b for a, b in reg_metrics(y["test"], pp[m]).items() if a in ("RMSE", "MAE", "R2")}))
        # ---- SHAP inputs (native XGBoost TreeSHAP on the final tuned/baseline XGBoost) ----
        if fs == final and split_name in CFG["shap_splits"]:
            mdl = (models.get("XGB_Tuned") or models.get("XGBoost"))
            if mdl is not None:
                chunk["shap"] = shap_inputs(mdl, M["test"][0], y["test"], prep.names, k == 0, seed)
        del models; gc.collect()
    chunk["fold_s"] = time.time() - t_fold; chunk["sig"] = sig
    pickle.dump(chunk, open(ck, "wb"))
    return chunk


def shap_inputs(mdl, Xte, yte, names, first_fold, seed):
    rng = np.random.RandomState(seed); n = len(Xte)
    sub = rng.choice(n, min(CFG["shap_rows"], n), replace=False)
    booster = mdl.est.get_booster(); dm = xgb.DMatrix(Xte[sub])
    contrib = booster.predict(dm, pred_contribs=True)
    out = dict(names=list(names), mean_abs=np.abs(contrib[:, :-1]).mean(0), sub=sub, X=Xte[sub].astype(np.float32),
               shap=contrib[:, :-1].astype(np.float32), base=float(contrib[:, -1].mean()), pred=mdl.predict(Xte[sub], None), y=yte[sub])
    if first_fold:
        ii = rng.choice(len(sub), min(CFG["inter_rows"], len(sub)), replace=False)
        try:
            inter = booster.predict(xgb.DMatrix(Xte[sub][ii]), pred_interactions=True)
            m = np.abs(inter[:, :-1, :-1]).mean(0); np.fill_diagonal(m, 0); out["inter"] = m.astype(np.float32)
        except Exception as e:
            out["inter"] = None; print("  interaction values skipped:", repr(e)[:120])
        try:
            pi = permutation_importance(mdl.est, Xte[sub][:CFG["perm_rows"]], yte[sub][:CFG["perm_rows"]], scoring="neg_root_mean_squared_error",
                                        n_repeats=3, random_state=seed, n_jobs=1)
            out["perm"] = pi.importances_mean
        except Exception as e:
            out["perm"] = None; print("  permutation importance skipped:", repr(e)[:120])
    return out


def run_dataset(ds, splits, deadline_h):
    chunks, last = [], 0.0
    for sp in SPLIT_ORDER:
        if sp not in splits: continue
        for k, (tr, te) in enumerate(splits[sp]):
            if elapsed_h() + 1.25 * last > deadline_h and not (OUT / "ckpt" / ds.name / f"{sp}_{k}.pkl").exists():
                log(f"  [{ds.name}] time budget reached — skipping {sp} fold {k} and later folds"); break
            existed = (OUT / "ckpt" / ds.name / f"{sp}_{k}.pkl").exists(); _t_chk = time.time(); c = run_fold(ds, sp, k, tr, te, deadline_h)
            if not existed or time.time() - _t_chk > 2: last = max(last * 0.5, c["fold_s"] / 3600)
            chunks.append(c)
            log(f"  [{ds.name}] {sp} fold {k + 1}/{len(splits[sp])} done: train={len(tr)} test={len(te)} ({c['fold_s']:.0f}s; final set={c['final']})")
    return chunks


# %% 09 — Pre-flight: prepare datasets, show feature blocks, benchmark one fold and PROJECT total runtime
DS = {}
for _name in CFG["datasets"]:
    try:
        DS[_name] = prepare_dataset(_name); print_dataset_summary(DS[_name])
    except AssertionError as e:
        print(f"[skip] {_name}: {e}")
SPLITS, SPLIT_NOTES = {}, {}
for _n, _d in DS.items():
    SPLITS[_n], SPLIT_NOTES[_n] = make_splits(_d)
    print(f"\n{_n} splits:", {k: f"{len(v)} fold(s), test sizes {[len(t[1]) for t in v][:6]}" for k, v in SPLITS[_n].items()})
    if SPLIT_NOTES[_n]: print("   not available:", SPLIT_NOTES[_n])
json.dump({n: v for n, v in SPLIT_NOTES.items()}, open(OUT / "audit" / "split_availability.json", "w"), indent=2)


def estimate_runtime():
    ref_name = list(DS)[0]; ref = DS[ref_name]
    sp = "S2_group" if "S2_group" in SPLITS[ref_name] else list(SPLITS[ref_name])[0]
    tr, te = SPLITS[ref_name][sp][0]
    fit_i, val_i, cal_i = partition_train(tr, sp, ref, SEED)
    fb = FeatureBuilder(ref).fit(ref.df.iloc[fit_i]); F = {n: fb.transform(ref.df.iloc[i]) for n, i in (("fit", fit_i), ("val", val_i), ("test", te))}
    fs = feature_sets(fb, list(F["fit"].columns)); final = list(fs)[-1]; cols = fs[final]
    prep = Prep().fit(F["fit"][cols]); M = {n: prep.transform(F[n][cols]) for n in F}
    t = time.time(); models, ft = fit_models(M["fit"][0], M["fit"][1], ref.y[fit_i], M["val"][0], M["val"][1], ref.y[val_i], SEED); t_models = time.time() - t
    t = time.time(); [models[m].predict(*M["test"]) for m in models]; t_pred = time.time() - t
    n_ref = len(fit_i); print(f"\n--- Pre-flight benchmark on {ref_name} ({n_ref} fit rows, {M['fit'][0].shape[1]} features, final set {final}) ---")
    print({k: f"{v:.1f}s" for k, v in ft.items()}); print(f"one (fold × feature-set) = {t_models:.1f}s fit, {t_pred:.2f}s predict-all")
    t_xgb = ft.get("XGBoost", 5.0); n_pert = 1 + 7 * CFG["rob_reps"] + 9
    total = {}
    for name, d in DS.items():
        scale = len(d.df) / max(len(ref.df), 1); folds = sum(len(v) for v in SPLITS[name].values()); n_fs = len(fs)
        rob_folds = sum(len(SPLITS[name].get(s, [])) for s in CFG["robust_splits"])
        t_fit = folds * n_fs * t_models * scale
        t_opt = folds * CFG["trials"] * t_xgb * 1.6 * scale
        t_rob = rob_folds * max(n_fs - 1, 1) * n_pert * t_pred * scale * 0.7
        t_misc = folds * (2 + 0.00005 * len(d.df))
        total[name] = (t_fit + t_opt + t_rob + t_misc) / 3600
        print(f"{name}: {folds} folds × {n_fs} feature sets  →  models {t_fit / 60:.0f} min | Optuna {t_opt / 60:.0f} min | stress tests {t_rob / 60:.0f} min | other {t_misc / 60:.0f} min  ≈ {total[name]:.2f} h")
    tot = sum(total.values()) + 0.35
    print(f"\nPROJECTED TOTAL ≈ {tot:.2f} h (incl. ~0.35 h for D3/D4, statistics, figures, export).  Budget: {CFG['time_budget_h']} h of 8 h quota.")
    if tot > CFG["time_budget_h"] - 0.3:
        print("⚠ Projection exceeds the budget. Options: lower CFG['trials'], set CFG['n_folds']=3, CFG['run_mlp']=False, "
              "CFG['robust_models']=[...fewer...], CFG['rf_est']=150, or run only D1 first (CFG['datasets']=['D1']).")
    return tot


PROJECTED_H = estimate_runtime()
# %% 10 — RUN THE EXPERIMENTS (long cell; checkpointed per fold — safe to re-run after a crash)
CHUNKS = {}
_exp_budget = CFG["time_budget_h"] - CFG["finalize_reserve_h"]; _t_start = elapsed_h()
for _i, (_name, _d) in enumerate(DS.items()):
    _deadline = _t_start + (_exp_budget - _t_start) * (_i + 1) / len(DS)       # unused time of D1 rolls over to D2
    log(f"=== Running {_name} (deadline at {_deadline:.2f} h) ===")
    CHUNKS[_name] = run_dataset(_d, SPLITS[_name], _deadline)
log("Experiments finished.")
if FAILED: print("[WARN] models skipped because they raised errors (count of folds):", FAILED)

# %% 11 — Phase H/N: metric tables (Tables 3–5) and the H2 leakage check
def ms(x, nd=3): return f"{np.mean(x):.{nd}f} ± {np.std(x, ddof=1) if len(x) > 1 else 0:.{nd}f}"


METRICS = pd.DataFrame([r for n in CHUNKS for c in CHUNKS[n] for r in c["metrics"]])
save_table(METRICS, "metrics_by_fold", index=False)
FINAL = {n: (CHUNKS[n][0]["final"] if CHUNKS[n] else None) for n in CHUNKS}
MODEL_SHOW = ["Mean", "Linear", "Ridge", "ElasticNet", "RF", "ExtraTrees", "XGBoost", "LightGBM", "CatBoost", "MLP", "XGB_Tuned", "Ensemble"]


def table_summary(metric="RMSE"):
    g = METRICS.groupby(["dataset", "split", "fset", "model"])[metric].apply(lambda s: ms(s.values)).reset_index()
    return g


T3 = METRICS[METRICS.apply(lambda r: r.fset == FINAL[r.dataset], axis=1)]
T3 = T3.groupby(["dataset", "split", "model"]).agg(RMSE=("RMSE", ms), MAE=("MAE", ms), R2=("R2", ms), NRMSE=("NRMSE", ms), folds=("fold", "nunique")).reset_index()
T3["m_order"] = T3.model.map({m: i for i, m in enumerate(MODEL_SHOW)}); T3 = T3.sort_values(["dataset", "split", "m_order"]).drop(columns="m_order")
save_table(T3, "Table3_baseline_performance_final_feature_set", index=False)
T4 = METRICS[METRICS.model.isin(["Ridge", "RF", "XGBoost", "LightGBM", "CatBoost", "Ensemble"])]
T4 = T4.groupby(["dataset", "split", "model", "fset"]).RMSE.apply(lambda s: ms(s.values)).unstack("fset").reset_index()
save_table(T4, "Table4_ablation_E1_E5_RMSE", index=False)
T5 = METRICS[METRICS.apply(lambda r: r.fset == FINAL[r.dataset], axis=1)].groupby(["dataset", "model", "split"]).RMSE.apply(lambda s: ms(s.values)).unstack("split").reset_index()
save_table(T5, "Table5_split_generalisation_RMSE", index=False)
print("Table 3 (final feature set):"); display(T3.head(40)) if "display" in globals() else print(T3.head(40).to_string())
print("\nTable 4 (ablation, RMSE mean ± sd across folds):"); print(T4.to_string(index=False))

# H2: random vs leakage-safe splits
H2 = []
for n in CHUNKS:
    m = METRICS[(METRICS.dataset == n)]
    m = m[m.fset == FINAL[n]]
    if "S1_random" not in set(m.split): continue
    for mod in MODEL_SHOW:
        a = m[(m.model == mod) & (m.split == "S1_random")].RMSE.values
        for sp in ("S2_group", "S3_temporal", "S4_spatial", "S5_ood"):
            b = m[(m.model == mod) & (m.split == sp)].RMSE.values
            if len(a) and len(b):
                p = sps.ttest_ind(b, a, equal_var=False).pvalue if len(a) > 1 and len(b) > 1 else np.nan
                H2.append(dict(dataset=n, model=mod, vs_split=sp, RMSE_random=a.mean(), RMSE_split=b.mean(), ratio=b.mean() / a.mean(), welch_p_exploratory=p))
H2 = save_table(pd.DataFrame(H2), "H2_random_vs_leakage_safe", index=False)
print("\nH2 — error inflation relative to the random split (ratio > 1 means random splits were optimistic):")
print(H2[H2.model.isin(["XGBoost", "Ensemble"])].to_string(index=False))

# %% 12 — Phase N: paired statistics (cluster bootstrap by field, Wilcoxon on absolute errors, effect size, Holm correction)
def paired_arrays(name, split, A, B):
    (fa, ma), (fb_, mb) = A, B; idx, pa, pb = [], [], []
    for c in sorted([c for c in CHUNKS[name] if c["split"] == split], key=lambda c: c["fold"]):
        if (fa, ma) in c["preds"] and (fb_, mb) in c["preds"]:
            idx.append(c["test_idx"]); pa.append(c["preds"][(fa, ma)]["test"].astype(float)); pb.append(c["preds"][(fb_, mb)]["test"].astype(float))
    if not idx: return None
    idx = np.concatenate(idx); return idx, DS[name].y[idx], np.concatenate(pa), np.concatenate(pb)


def paired_stats(name, split, A, B, label):
    pa_ = paired_arrays(name, split, A, B)
    if pa_ is None: return None
    idx, y, pa, pb = pa_; ea, eb = y - pa, y - pb
    gid = DS[name].grp_field[idx] if DS[name].grp_field is not None else idx
    codes = pd.factorize(gid)[0]; G = codes.max() + 1
    sa = np.bincount(codes, weights=ea ** 2, minlength=G); sb = np.bincount(codes, weights=eb ** 2, minlength=G); cn = np.bincount(codes, minlength=G)
    rng = np.random.RandomState(SEED); d = np.empty(CFG["boot"])
    for b in range(CFG["boot"]):
        s = rng.randint(0, G, G); d[b] = np.sqrt(sb[s].sum() / cn[s].sum()) - np.sqrt(sa[s].sum() / cn[s].sum())
    diff = np.abs(eb) - np.abs(ea)
    try: p = sps.wilcoxon(diff).pvalue
    except Exception: p = np.nan
    ra, rb = rmse(y, pa), rmse(y, pb)
    return dict(dataset=name, split=split, comparison=label, baseline=f"{A[0]}/{A[1]}", challenger=f"{B[0]}/{B[1]}", n=len(y), n_clusters=int(G),
                RMSE_baseline=ra, RMSE_challenger=rb, delta_RMSE=rb - ra, rel_change_pct=100 * (rb - ra) / ra,
                CI95_lo=float(np.percentile(d, 2.5)), CI95_hi=float(np.percentile(d, 97.5)), wilcoxon_p=p,
                cohen_dz=float(diff.mean() / (diff.std(ddof=1) or np.nan)) * -1)      # positive dz = challenger has smaller absolute errors


def holm(p):
    p = np.asarray(p, float); o = np.argsort(np.where(np.isnan(p), 2, p)); adj = np.empty_like(p); m = np.sum(~np.isnan(p)); run = 0
    for r, i in enumerate(o):
        if np.isnan(p[i]): adj[i] = np.nan; continue
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


STATS = []
for n in CHUNKS:
    f = FINAL[n]
    if f is None: continue
    sp_all = sorted({c["split"] for c in CHUNKS[n]}); fs_list = list(CHUNKS[n][0]["fsets"])
    for sp in sp_all:
        for mod in ("XGBoost", "Ensemble"):
            for a, b in zip(fs_list[:-1], fs_list[1:]):
                STATS.append(paired_stats(n, sp, (a, mod), (b, mod), f"{mod}: {a} → {b}"))
            if len(fs_list) > 2: STATS.append(paired_stats(n, sp, (fs_list[0], mod), (f, mod), f"{mod}: {fs_list[0]} → {f}"))
        for a, b in (("XGBoost", "XGB_Tuned"), ("XGBoost", "Ensemble"), ("RF", "XGBoost"), ("Ridge", "XGBoost"), ("CatBoost", "XGBoost"), ("XGB_Tuned", "Ensemble")):
            STATS.append(paired_stats(n, sp, (f, a), (f, b), f"final set: {a} → {b}"))
STATS = pd.DataFrame([s for s in STATS if s])
if len(STATS):
    STATS["wilcoxon_p_holm"] = holm(STATS.wilcoxon_p.values)
    STATS["CI_excludes_0"] = (STATS.CI95_lo > 0) | (STATS.CI95_hi < 0)
save_table(STATS, "Table_paired_statistics", index=False)
print("Paired statistics: delta_RMSE<0 means the challenger is better. Showing feature-set comparisons for XGBoost:")
print(STATS[STATS.comparison.str.startswith("XGBoost:")][["dataset", "split", "comparison", "RMSE_baseline", "RMSE_challenger", "rel_change_pct", "CI95_lo", "CI95_hi", "wilcoxon_p_holm"]].round(4).to_string(index=False))

# %% 13 — Phase I: uncertainty — split-conformal, normalised conformal (ensemble spread) and CQR (LightGBM quantiles)
ALPHAS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
CONF_ROWS, CONF_SUM, CONF_STRESS, CONF_CAL = {}, [], [], []
for n in CHUNKS:
    ds = DS[n]; f = FINAL[n]
    if f is None: continue
    if ds.stress is not None: q1, q2 = np.quantile(ds.stress, [1 / 3, 2 / 3])
    for sp in sorted({c["split"] for c in CHUNKS[n]}, key=lambda s: SPLIT_ORDER.index(s)):
        acc = {}
        for c in sorted([c for c in CHUNKS[n] if c["split"] == sp], key=lambda c: c["fold"]):
            te, cal = c["test_idx"], c["cal_idx"]; yc, yt = ds.y[cal], ds.y[te]
            P = lambda m, pt: c["preds"][(f, m)][pt].astype(float)
            members = [m for m in ENS_MEMBERS if (f, m) in c["preds"]]
            base_m = "XGB_Tuned" if (f, "XGB_Tuned") in c["preds"] else "XGBoost"
            sig_c = member_spread({m: P(m, "cal") for m in members}, members); sig_t = member_spread({m: P(m, "test") for m in members}, members)
            eps = 0.1 * np.median(sig_c) + 1e-9
            for a in ALPHAS:
                res = {"SplitConf-Ensemble": split_conformal(P("Ensemble", "cal"), yc, P("Ensemble", "test"), a),
                       f"SplitConf-{base_m}": split_conformal(P(base_m, "cal"), yc, P(base_m, "test"), a),
                       "NormConf-Ensemble": norm_conformal(P("Ensemble", "cal"), sig_c + eps, yc, P("Ensemble", "test"), sig_t + eps, a)}
                if a == CFG["alpha"] and f in c["cqr"]:
                    q = c["cqr"][f]; res["CQR-LightGBM"] = cqr_calibrate(q["lo_cal"], q["hi_cal"], yc, q["lo_test"], q["hi_test"], a)
                for meth, (lo, hi) in res.items():
                    acc.setdefault((meth, a), []).append(pd.DataFrame(dict(idx=te, y=yt, lo=lo, hi=hi, fold=c["fold"],
                                                                         pred=P("Ensemble" if "Ensemble" in meth else base_m, "test") if not meth.startswith("CQR") else (lo + hi) / 2)))
        for (meth, a), parts in acc.items():
            D = pd.concat(parts, ignore_index=True)
            if ds.stress is not None: D["stress"] = ds.stress[D.idx.values]
            cover = ((D.y >= D.lo) & (D.y <= D.hi)).values
            CONF_CAL.append(dict(dataset=n, split=sp, method=meth, nominal=1 - a, empirical=float(cover.mean())))
            if abs(a - CFG["alpha"]) < 1e-9:
                CONF_ROWS[(n, sp, meth)] = D
                w = (D.hi - D.lo).values; err = np.abs(D.y - D.pred).values
                st = interval_stats(D.y.values, D.lo.values, D.hi.values, a)
                rho = sps.spearmanr(w, err)[0] if np.std(w) > 1e-12 else np.nan
                fold_cov = D.groupby("fold").apply(lambda g: ((g.y >= g.lo) & (g.y <= g.hi)).mean())
                CONF_SUM.append(dict(dataset=n, split=sp, method=meth, n=len(D), **st, PICP_gap=st["PICP"] - (1 - a), PICP_fold_sd=float(fold_cov.std()) if len(fold_cov) > 1 else np.nan,
                                     spearman_width_vs_abs_err=rho, mean_abs_err=float(err.mean())))
                if ds.stress is not None:
                    lab = pd.cut(D.stress, [-np.inf, q1, q2, np.inf], labels=["low", "mid", "high"])
                    for lv in ["low", "mid", "high"]:
                        g = D[lab == lv]
                        if len(g): CONF_STRESS.append(dict(dataset=n, split=sp, method=meth, stress_level=lv, n=len(g),
                                                           PICP=float(((g.y >= g.lo) & (g.y <= g.hi)).mean()), MPIW=float((g.hi - g.lo).mean()),
                                                           mean_abs_err=float(np.abs(g.y - g.pred).mean())))
CONF_SUM, CONF_STRESS, CONF_CAL = pd.DataFrame(CONF_SUM), pd.DataFrame(CONF_STRESS), pd.DataFrame(CONF_CAL)
save_table(CONF_SUM, "Table7_uncertainty_calibration", index=False); save_table(CONF_STRESS, "Table7b_coverage_by_climate_stress", index=False); save_table(CONF_CAL, "calibration_curve_data", index=False)
print("Table 7 — 90% prediction intervals (nominal coverage 0.90):"); print(CONF_SUM.round(4).to_string(index=False))

# %% 14 — Phase J: robustness aggregation (Table 6) and Phase K: SHAP aggregation (Table 8)
ROB = pd.DataFrame([r for n in CHUNKS for c in CHUNKS[n] for r in c["robust"]])
if len(ROB):
    ROBG = ROB.groupby(["dataset", "split", "fset", "model", "kind", "level"])[["RMSE", "MAE", "R2"]].mean().reset_index()
    clean = ROBG[ROBG.kind == "clean"].set_index(["dataset", "split", "fset", "model"])["RMSE"]
    ROBG["RMSE_clean"] = [clean.get((r.dataset, r.split, r.fset, r.model), np.nan) for r in ROBG.itertuples()]
    ROBG["RMSE_ratio"] = ROBG.RMSE / ROBG.RMSE_clean
    save_table(ROBG, "Table6_robustness_all", index=False)
    T6 = ROBG[(ROBG.kind != "clean") & (ROBG.model.isin(["XGBoost", "Ensemble", "Ridge"]))].pivot_table(index=["dataset", "split", "fset", "model"], columns=["kind", "level"], values="RMSE_ratio")
    save_table(T6, "Table6_robustness_RMSE_ratio_vs_clean"); print("Table 6 (RMSE ratio vs clean test set; 1.00 = no degradation):"); print(T6.round(3).to_string())
else:
    ROBG = pd.DataFrame(); print("No stress-test results (robust_splits not run).")


def block_of_name(nm, ds):
    if nm in ds.blocks: return ds.blocks[nm]
    if nm.endswith("__isna"): return ds.blocks.get(nm[:-6], "context")
    if "=" in nm: return "context"
    if nm.startswith("ts_"): return "temporal"
    return "engineered"


SHAP_IMP, SHAP_FIRST = {}, {}
for n in CHUNKS:
    for c in CHUNKS[n]:
        if c["shap"] is None: continue
        s = pd.Series(c["shap"]["mean_abs"], index=c["shap"]["names"]); SHAP_IMP.setdefault((n, c["split"]), []).append(s)
        if c["fold"] == 0: SHAP_FIRST[(n, c["split"])] = c["shap"]
SHAP_IMP = {k: pd.concat(v, axis=1).fillna(0).mean(axis=1).sort_values(ascending=False) for k, v in SHAP_IMP.items()}
T8, SHAP_CMP = [], []
for (n, sp), s in SHAP_IMP.items():
    for rank, (feat, val) in enumerate(s.head(15).items(), 1):
        T8.append(dict(dataset=n, split=sp, rank=rank, feature=feat, block=block_of_name(feat, DS[n]), mean_abs_shap=val, share_pct=100 * val / s.sum()))
    ref = SHAP_IMP.get((n, "S1_random"))
    if ref is not None and sp != "S1_random":
        both = s.index.intersection(ref.index)
        SHAP_CMP.append(dict(dataset=n, split=sp, spearman_rank_vs_random=sps.spearmanr(s[both], ref[both])[0], top10_overlap=len(set(s.head(10).index) & set(ref.head(10).index))))
T8 = save_table(pd.DataFrame(T8), "Table8_shap_top_drivers", index=False); SHAP_CMP = save_table(pd.DataFrame(SHAP_CMP), "shap_importance_stability_vs_random", index=False)
if len(T8):
    blk = T8.assign(blk=T8.block).groupby(["dataset", "split", "blk"]).share_pct.sum().unstack("blk").round(1)
    print("Share of top-15 SHAP importance by information block (%):"); print(blk.to_string())
    print("\nTop drivers:"); print(T8[T8["rank"] <= 8][["dataset", "split", "rank", "feature", "block", "share_pct"]].round(2).to_string(index=False))
    if len(SHAP_CMP): print("\nImportance stability vs. random split:"); print(SHAP_CMP.round(3).to_string(index=False))

# %% 15 — Decision gates (automated where possible; Gate 5 needs YOUR agronomic judgement)
GATES = {}
for n in CHUNKS:
    a = DS[n].audit; f = FINAL[n]; g = {}
    notes = []
    if a["suspected_leakage_columns_removed"]: notes.append(f"removed suspected leakage columns {list(a['suspected_leakage_columns_removed'])}")
    if a.get("target_constant_within_field"): notes.append("target is constant within field → S1 random split leaks by construction")
    g["Gate1_audit"] = ("REVIEW" if notes else "PASS", "; ".join(notes) or "no leakage flags; confirm units and target semantics manually")
    safe = [s for s in ("S2_group", "S3_temporal", "S4_spatial") if s in SPLITS[n] and any(c["split"] == s for c in CHUNKS[n])]
    g["Gate2_leakage_safe_split"] = ("PASS" if safe else "FAIL", f"available: {safe}")
    st = STATS[(STATS.dataset == n) & STATS.comparison.str.startswith("XGBoost:")] if len(STATS) else pd.DataFrame()
    fl = list(CHUNKS[n][0]["fsets"]) if CHUNKS[n] else []
    if len(st) and len(fl) > 1:
        lab = f"XGBoost: {fl[0]} → {fl[-1]}" if len(fl) > 2 else f"XGBoost: {fl[0]} → {fl[1]}"
        s2 = st[st.comparison == lab]; shift = s2[s2.split.isin(safe + ["S5_ood"])]
        win = shift[(shift.CI95_hi < 0)]; gain = (-shift.rel_change_pct).clip(lower=0).mean() if len(shift) else 0
        ok = len(shift) > 0 and len(win) >= max(1, math.ceil(len(shift) / 2)) and gain >= 3
        g["Gate3_multimodal_gain"] = ("PASS" if ok else "FAIL", f"{lab}: significant improvement (95% CI < 0) in {len(win)}/{len(shift)} shift-aware splits; mean gain {gain:.1f}%"
                                      + ("" if ok else " → do NOT force the multimodal story; pivot to uncertainty / shift analysis (plan §25)"))
        s3 = s2[s2.split.isin(["S3_temporal", "S4_spatial", "S5_ood"])]
        g["Gate6_survives_shift"] = (("PASS" if len(s3) and (s3.delta_RMSE < 0).mean() >= 2 / 3 else "FAIL" if len(s3) else "N/A"),
                                     f"E-level gain has the same sign in {(s3.delta_RMSE < 0).sum()}/{len(s3)} temporal/spatial/OOD splits")
    cs = CONF_SUM[(CONF_SUM.dataset == n)] if len(CONF_SUM) else pd.DataFrame()
    if len(cs):
        ref_sp = (safe or ["S1_random"])[0]; r = cs[cs.split == ref_sp]
        cov_ok = r[r.method.str.startswith("SplitConf")].PICP_gap.abs().le(0.05).any()
        adapt = r[r.method.isin(["NormConf-Ensemble", "CQR-LightGBM"])].spearman_width_vs_abs_err.max()
        g["Gate4_uncertainty"] = ("PASS" if cov_ok and adapt >= 0.2 else "REVIEW", f"split {ref_sp}: split-conformal coverage within ±0.05 of nominal={cov_ok}; best adaptive width–error Spearman={adapt:.2f}")
    g["Gate5_agronomic_plausibility"] = ("MANUAL", "Inspect Table 8 / Fig 8-9: are the top drivers (rainfall/temperature/moisture/vegetation indices/N) agronomically sensible and signed correctly?")
    GATES[n] = g
json.dump(GATES, open(OUT / "tables" / "decision_gates.json", "w"), indent=2)
for n, g in GATES.items():
    print(f"\n== Decision gates — {n} ==")
    for k, (s, m) in g.items(): print(f"  {k:32s} {s:7s} {m}")
# %% 16 — Phases L & M: D3 (India crop production, contextual) and D4 (crop-recommendation agronomic envelopes) — never row-merged with D1/D2
D3_RES, D4_RES = {}, {}
if SRC.get("D3"):
    d3 = pd.read_csv(SRC["D3"]); d3.columns = [sanitize(c) for c in d3.columns]
    n0 = len(d3); d3 = d3.dropna(subset=["Area", "Production"]); d3 = d3[d3["Area"] > 0].copy()
    d3["yield_proxy"] = d3["Production"] / d3["Area"]                      # t/ha only if Production in tonnes and Area in ha (verify on the Kaggle page)
    hi = d3.yield_proxy.quantile(0.999); d3_clean = d3[d3.yield_proxy <= hi]
    maize = d3_clean[d3_clean["Crop"].astype(str).str.lower().eq("maize")]
    yr = maize.groupby("Crop_Year").yield_proxy.median()
    tau, p_tau = sps.kendalltau(yr.index, yr.values) if len(yr) > 3 else (np.nan, np.nan)
    ts = sps.theilslopes(yr.values, yr.index.values)[0] if len(yr) > 3 else np.nan
    seas = maize.groupby(maize["Season"].astype(str).str.strip()).yield_proxy.agg(["count", "median", "mean", "std"]).round(3)
    state = maize.groupby("State_Name").yield_proxy.agg(["count", "median"]).query("count>=30").sort_values("median", ascending=False)
    D3_RES = dict(rows_raw=n0, rows_valid=len(d3), maize_rows=len(maize), kendall_tau=float(tau), kendall_p=float(p_tau), theil_sen_slope_per_year=float(ts))
    save_table(seas, "D3_maize_season_summary"); save_table(state, "D3_maize_state_medians"); save_table(yr.to_frame("median_yield_proxy"), "D3_maize_year_trend")
    print("D3:", D3_RES); print(seas)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(yr.index, yr.values, marker="o"); ax[0].set(title="India maize: median yield proxy by year (D3)", xlabel="Year", ylabel="Production/Area")
    _g = maize.groupby(maize["Season"].astype(str).str.strip()).yield_proxy
    ax[1].boxplot([v.values for _, v in _g], labels=[k for k, _ in _g]); ax[1].set(title="By season", ylabel="yield proxy")
    state.head(12)["median"][::-1].plot.barh(ax=ax[2]); ax[2].set(title="Top states (median, n≥30)")
    save_fig("D3_india_maize_context")
    print("NOTE: D3 has no weather/soil variables, so it can only support season/trend/regional context — it cannot confirm climatic drivers.")
else:
    print("D3 not attached — skipped.")

if SRC.get("D4"):
    d4 = pd.read_csv(SRC["D4"]); d4.columns = [sanitize(c) for c in d4.columns]
    vars_ = [c for c in ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"] if c in d4.columns]
    env = d4.groupby("label")[vars_].quantile([0.1, 0.5, 0.9]).unstack(); env.columns = [f"{a}_q{int(b * 100)}" for a, b in env.columns]
    save_table(env, "D4_crop_envelopes")
    D4_RES["labels"] = sorted(d4["label"].unique().tolist())
    for n, ds in DS.items():                                                   # envelope-deviation score only when crop AND variables AND units are compatible
        crop_col = ds.S["crop"]; vals = ds.df[crop_col].astype(str).str.lower().unique().tolist() if crop_col else ["maize"] if n == "D1" else []
        cand = [v for v in vals if v in set(d4["label"].str.lower())]
        m = {"temperature": [c for c in ds.df.columns if _has(c, TEMP_K) and ds.blocks.get(c) == "weather"],
             "rainfall": [c for c in ds.df.columns if is_rain_amount(c) and ds.blocks.get(c) == "weather"],
             "humidity": [c for c in ds.df.columns if _has(c, HUM_K) and ds.blocks.get(c) == "weather"]}
        used, z = [], []
        for dv, cols_ in m.items():
            if not cols_ or dv not in d4.columns or not cand: continue
            ref = d4[d4.label.str.lower().isin(cand)][dv]; col = cols_[0]; ratio = ds.df[col].median() / (ref.median() or np.nan)
            if not (0.2 <= ratio <= 5):
                print(f"  [D4→{n}] units of '{col}' look incompatible with D4 '{dv}' (median ratio {ratio:.2f}) — envelope score skipped for this variable"); continue
            sd = ref.std() or 1.0; z.append(((ds.df[col] - ref.median()) / sd).abs().to_numpy()); used.append(f"{col}~{dv}")
        if z and len(cand):
            score = np.mean(z, axis=0); D4_RES[n] = dict(variables=used, crops=cand, spearman_with_yield=float(sps.spearmanr(score, ds.y, nan_policy="omit")[0]))
            print(f"  [D4→{n}] envelope deviation vs {cand}, variables {used}: Spearman with yield = {D4_RES[n]['spearman_with_yield']:.3f}")
        else:
            print(f"  [D4→{n}] no compatible crop label/variables → D4 stays purely contextual (as the plan prescribes).")
    if "maize" in set(d4.label.str.lower()):
        fig, ax = plt.subplots(1, len(vars_), figsize=(2.6 * len(vars_), 3.4))
        for a_, v in zip(np.atleast_1d(ax), vars_):
            a_.boxplot([d4[d4.label.str.lower() == "maize"][v]], labels=["maize"]); a_.set_title(v)
        plt.suptitle("D4 agronomic envelope: maize"); save_fig("D4_maize_envelope")
else:
    print("D4 not attached — skipped.")

# %% 17 — Tables 1–2 and Figures 1–10
def draw_flow(ax, boxes, arrows):
    for (x, y, w, h, t, c) in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=c, ec="k", lw=1)); ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8, wrap=True)
    for (x1, y1, x2, y2) in arrows: ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->"))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")


fig, ax = plt.subplots(figsize=(13, 5.5))
draw_flow(ax, [(0.2, 4.2, 2.4, 1.2, "Weather\n(rain, temp, humidity, moisture)", "#cfe8ff"), (0.2, 2.5, 2.4, 1.2, "Soil / management", "#e6d3b3"),
               (0.2, 0.8, 2.4, 1.2, "Remote sensing\n(NDVI, GNDVI, NDWI, SAVI …)", "#c9f0c9"), (3.4, 2.5, 2.2, 1.2, "Agro-climatic\nfeature engineering\n(+ causal temporal summaries)", "#fff2b3"),
               (6.2, 2.5, 2.0, 1.2, "Learners: Ridge, RF, ET,\nXGB, LGBM, CatBoost, MLP\n→ ensemble", "#f3cfe9"), (8.9, 4.2, 2.8, 1.2, "Conformal / CQR\nprediction intervals", "#ffd9c2"),
               (8.9, 2.5, 2.8, 1.2, "Stress & shift tests\n(missing, noise, rain↓, temp↑,\ntemporal / spatial / OOD)", "#ffd9c2"), (8.9, 0.8, 2.8, 1.2, "SHAP global / local /\ninteraction + permutation", "#ffd9c2")],
          [(2.6, 4.8, 3.4, 3.4), (2.6, 3.1, 3.4, 3.1), (2.6, 1.4, 3.4, 2.8), (5.6, 3.1, 6.2, 3.1), (8.2, 3.1, 8.9, 4.7), (8.2, 3.1, 8.9, 3.1), (8.2, 3.1, 8.9, 1.4)])
ax.set_title("Fig. 1 — Multisource, explainable, uncertainty-aware yield-prediction framework"); save_fig("Fig1_framework")

fig, ax = plt.subplots(figsize=(12, 4.6))
d1c = "#cfe8ff" if "D1" in DS else "#eeeeee"; d2c = "#c9f0c9" if "D2" in DS else "#eeeeee"
draw_flow(ax, [(0.3, 3.8, 2.8, 1.4, "D1 AgriYield 2025\n(PRIMARY, field-level)", d1c), (0.3, 1.8, 2.8, 1.4, ("D2 wheat (Kaggle shaikasif89/\nwheat-yeild): independent\nvalidation" if CFG["d2_source"] == "wheat" else "D2 kaggle-cup-2025-01\n(independent multimodal\nvalidation)"), d2c),
               (4.8, 2.8, 2.6, 1.4, "Separate experiments\n(NO row-level merge)", "#fff2b3"), (8.4, 3.8, 3.2, 1.4, "D3 India production\n(context: season / trend / state)", "#e6d3b3"),
               (8.4, 1.8, 3.2, 1.4, "D4 crop envelopes\n(agronomic plausibility check)", "#e6d3b3")],
          [(3.1, 4.5, 4.8, 3.9), (3.1, 2.5, 4.8, 3.4), (7.4, 3.5, 8.4, 4.5), (7.4, 3.5, 8.4, 2.5)])
ax.set_title("Fig. 2 — Data-fusion strategy: joined vs. external/contextual sources"); save_fig("Fig2_data_fusion")

T1, T2 = [], []
for n, ds in DS.items():
    a = ds.audit
    T1.append(dict(dataset=n, source=a["source"], rows=a["rows_used"], target=a["target"], target_mean=a["target_stats"]["mean"], target_sd=a["target_stats"]["sd"],
                   target_skew=a["target_stats"]["skew"], fields=a.get("obs_per_field", {}).get("n_fields"), date_col=ds.S["date"], coords=bool(ds.S["lat"]),
                   n_features=len(ds.blocks), access_date=str(dt.date.today())))
    for c, b in ds.blocks.items():
        s = ds.df[c]; num = pd.api.types.is_numeric_dtype(s)
        T2.append(dict(dataset=n, feature=c, block=b, dtype=str(s.dtype), missing_pct=100 * s.isna().mean(), mean=s.mean() if num else np.nan, sd=s.std() if num else np.nan))
save_table(pd.DataFrame(T1), "Table1_dataset_characteristics", index=False); save_table(pd.DataFrame(T2), "Table2_data_dictionary", index=False)

for n, ds in DS.items():
    # Fig 3: target distribution + association
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    ax[0].hist(ds.y, bins=50, color="#4c78a8"); ax[0].set(title=f"{n}: target distribution (skew={sps.skew(ds.y):.2f})", xlabel=ds.S["target"])
    num = [c for c in ds.blocks if pd.api.types.is_numeric_dtype(ds.df[c]) and not c.startswith("ts_")]
    cor = pd.Series({c: sps.spearmanr(ds.df[c], ds.y, nan_policy="omit")[0] for c in num}).dropna()
    top = cor.reindex(cor.abs().sort_values(ascending=False).head(15).index)
    ax[1].barh(top.index[::-1], top.values[::-1], color=["#59a14f" if v > 0 else "#e15759" for v in top.values[::-1]]); ax[1].set(title="Top |Spearman| associations with target")
    save_fig(f"Fig3_target_and_associations_{n}")
    if not CHUNKS.get(n): continue
    fsl = list(CHUNKS[n][0]["fsets"]); splits_here = [s for s in SPLIT_ORDER if any(c["split"] == s for c in CHUNKS[n])]
    # Fig 4: E-levels
    fig, axs = plt.subplots(1, len(splits_here), figsize=(4.2 * len(splits_here), 4), sharey=False); axs = np.atleast_1d(axs)
    for a_, sp in zip(axs, splits_here):
        for j, mod in enumerate(["XGBoost", "Ensemble"]):
            g = METRICS[(METRICS.dataset == n) & (METRICS.split == sp) & (METRICS.model == mod)].groupby("fset").RMSE.agg(["mean", "std"]).reindex(fsl)
            a_.bar(np.arange(len(fsl)) + 0.38 * j, g["mean"], 0.36, yerr=g["std"].fillna(0), label=mod, capsize=3)
        a_.set_xticks(np.arange(len(fsl)) + 0.19); a_.set_xticklabels(fsl); a_.set_title(sp); a_.set_ylabel("RMSE")
    axs[0].legend(); fig.suptitle(f"Fig. 4 — Information-level ablation ({n})"); save_fig(f"Fig4_ablation_{n}")
    # Fig 5: model comparison across splits
    fig, ax = plt.subplots(figsize=(12, 4.4)); mods = [m for m in MODEL_SHOW if m != "Mean"]
    for j, sp in enumerate(splits_here):
        g = METRICS[(METRICS.dataset == n) & (METRICS.split == sp) & (METRICS.fset == FINAL[n])].groupby("model").RMSE.mean().reindex(mods)
        ax.bar(np.arange(len(mods)) + j * 0.8 / len(splits_here), g.values, 0.8 / len(splits_here), label=sp)
    ax.set_xticks(np.arange(len(mods)) + 0.4); ax.set_xticklabels(mods, rotation=40, ha="right"); ax.set_ylabel("RMSE"); ax.legend(); ax.set_title(f"Fig. 5 — Models × splits at {FINAL[n]} ({n})")
    save_fig(f"Fig5_models_vs_splits_{n}")
    # Fig 6: robustness curves
    if len(ROBG):
        r = ROBG[(ROBG.dataset == n) & (ROBG.model == "XGBoost") & (ROBG.kind != "clean")]
        kinds = [k for k in ["missing", "noise", "rain", "temp", "combined"] if k in set(r.kind)]
        if kinds:
            fig, axs = plt.subplots(1, len(kinds), figsize=(3.6 * len(kinds), 3.6)); axs = np.atleast_1d(axs)
            for a_, kd in zip(axs, kinds):
                for fs_ in sorted(set(r.fset)):
                    g = r[(r.kind == kd) & (r.fset == fs_)].groupby("level").RMSE_ratio.mean(); a_.plot(g.index, g.values, marker="o", label=fs_)
                a_.axhline(1, color="grey", ls=":"); a_.set_title(kd); a_.set_xlabel("level"); a_.set_ylabel("RMSE / clean RMSE")
            axs[0].legend(title="feature set"); fig.suptitle(f"Fig. 6 — Robustness of XGBoost ({n}, group split)"); save_fig(f"Fig6_robustness_{n}")
    # Fig 7: calibration + stress
    cc = CONF_CAL[CONF_CAL.dataset == n]
    if len(cc):
        sp_ref = splits_here[0]; fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        for meth, g in cc[cc.split == sp_ref].groupby("method"): ax[0].plot(g.nominal, g.empirical, marker="o", label=meth)
        ax[0].plot([0.5, 1], [0.5, 1], "k:"); ax[0].set(title=f"Coverage calibration ({sp_ref})", xlabel="nominal", ylabel="empirical")
        ax[0].legend(fontsize=7)
        for meth in ["SplitConf-Ensemble", "NormConf-Ensemble", "CQR-LightGBM"]:
            g = CONF_STRESS[(CONF_STRESS.dataset == n) & (CONF_STRESS.split == sp_ref) & (CONF_STRESS.method == meth)]
            if len(g): ax[1].plot(g.stress_level, g.PICP, marker="o", label=meth)
        ax[1].axhline(1 - CFG["alpha"], color="k", ls=":"); ax[1].set(title="PICP by climate-stress tertile", ylabel="coverage"); ax[1].legend(fontsize=7)
        for meth, mk in [("NormConf-Ensemble", "o"), ("CQR-LightGBM", "s")]:
            D = CONF_ROWS.get((n, sp_ref, meth))
            if D is not None:
                w = (D.hi - D.lo); e = (D.y - D.pred).abs(); b = pd.qcut(w.rank(method="first"), 10, labels=False)
                ax[2].plot(w.groupby(b).mean(), e.groupby(b).mean(), marker=mk, label=meth)
        ax[2].set(title="Mean |error| vs. interval width (deciles)", xlabel="mean width", ylabel="mean |error|"); ax[2].legend(fontsize=7)
        fig.suptitle(f"Fig. 7 — Uncertainty ({n})"); save_fig(f"Fig7_uncertainty_{n}")
    # Fig 8/9: SHAP
    key = next(((n, s) for s in ["S2_group", "S3_temporal", "S1_random", "S5_ood"] if (n, s) in SHAP_IMP), None)
    if key:
        imp = SHAP_IMP[key].head(15)[::-1]; cols = {"weather": "#4c78a8", "soil": "#b07a3b", "remote": "#59a14f", "engineered": "#f28e2b", "temporal": "#9c755f", "context": "#bab0ab"}
        fig, ax = plt.subplots(figsize=(7, 5)); ax.barh(imp.index, imp.values, color=[cols.get(block_of_name(f_, ds), "#999") for f_ in imp.index])
        ax.set(title=f"Fig. 8 — Mean |SHAP| ({n}, {key[1]})"); handles = [plt.Rectangle((0, 0), 1, 1, color=v) for v in cols.values()]; ax.legend(handles, cols.keys(), fontsize=7); save_fig(f"Fig8_shap_global_{n}")
        sh = SHAP_FIRST.get(key)
        if sh is not None and shap is not None:
            try:
                shap.summary_plot(sh["shap"], sh["X"], feature_names=sh["names"], show=False, max_display=15); plt.title(f"SHAP beeswarm ({n}, {key[1]})"); save_fig(f"Fig8b_shap_beeswarm_{n}")
            except Exception as e: print("beeswarm skipped:", repr(e)[:100])
        if sh is not None:
            names = sh["names"]; topf = [names.index(f_) for f_ in SHAP_IMP[key].head(3).index if f_ in names]
            fig, axs = plt.subplots(1, len(topf) + 1, figsize=(4.2 * (len(topf) + 1), 3.8)); axs = np.atleast_1d(axs)
            for a_, j in zip(axs, topf):
                partner = None
                if sh.get("inter") is not None: partner = int(np.argmax(sh["inter"][j]))
                c_ = sh["X"][:, partner] if partner is not None else None
                sc = a_.scatter(sh["X"][:, j], sh["shap"][:, j], c=c_, s=6, cmap="viridis"); a_.set(xlabel=names[j], ylabel="SHAP", title=f"vs {names[partner]}" if partner is not None else "")
            if sh.get("inter") is not None:
                m = sh["inter"]; iu = np.dstack(np.unravel_index(np.argsort(-np.triu(m).ravel())[:8], m.shape))[0]
                axs[-1].barh([f"{names[i]} × {names[j]}" for i, j in iu][::-1], [m[i, j] for i, j in iu][::-1]); axs[-1].set_title("Top SHAP interactions")
            fig.suptitle(f"Fig. 9 — Dependence & interaction ({n})"); save_fig(f"Fig9_shap_dependence_{n}")
            order = np.argsort(sh["pred"]); picks = {"low": order[int(0.05 * len(order))], "medium": order[len(order) // 2], "high": order[int(0.95 * len(order)) - 1]}
            fig, axs = plt.subplots(1, 3, figsize=(15, 3.8))
            for a_, (lab, i) in zip(axs, picks.items()):
                v = pd.Series(sh["shap"][i], index=names); v = v.reindex(v.abs().sort_values(ascending=False).head(8).index)[::-1]
                a_.barh(v.index, v.values, color=["#59a14f" if x > 0 else "#e15759" for x in v.values]); a_.set_title(f"{lab}-yield case: pred={sh['pred'][i]:.1f}, observed={sh['y'][i]:.1f}")
            fig.suptitle("Local SHAP case studies"); save_fig(f"Fig9b_shap_local_{n}")
    # Fig 10: error / uncertainty map or subgroup analysis
    D = CONF_ROWS.get((n, splits_here[0], "CQR-LightGBM"))
    if D is None: D = CONF_ROWS.get((n, splits_here[0], "NormConf-Ensemble"))
    if D is not None:
        D = D.copy(); D["abs_err"] = (D.y - D.pred).abs(); D["width"] = D.hi - D.lo
        if ds.S["lat"] and ds.S["lon"]:
            fig, ax = plt.subplots(1, 2, figsize=(12, 4.6)); lat, lon = ds.df[ds.S["lat"]].values[D.idx.values], ds.df[ds.S["lon"]].values[D.idx.values]
            for a_, col, t in zip(ax, ["abs_err", "width"], ["Absolute error", "Interval width"]):
                s = a_.scatter(lon, lat, c=D[col], s=6, cmap="magma"); plt.colorbar(s, ax=a_); a_.set(title=t, xlabel="longitude", ylabel="latitude")
        else:
            grp = ds.S["crop"] or ds.S["region"]; fig, ax = plt.subplots(figsize=(7, 4))
            if grp:
                lab = ds.df[grp].astype(str).values[D.idx.values]; D.assign(g=lab).boxplot(column="abs_err", by="g", ax=ax); plt.suptitle("")
            ax.set_title("Absolute error by subgroup")
        fig.suptitle(f"Fig. 10 — Error/uncertainty structure ({n}, {splits_here[0]})"); save_fig(f"Fig10_error_map_{n}")

# %% 18 — Reproducibility export (config, environment, splits, feature lists, Optuna trials, final models) + zip
env = dict(python=platform.python_version(), platform=platform.platform(), gpu_available=HAS_GPU, cpu_count=N_CPU, seed=SEED,
           numpy=np.__version__, pandas=pd.__version__, xgboost=xgb.__version__, lightgbm=lgb.__version__, optuna=optuna.__version__,
           shap=getattr(shap, "__version__", None), torch=getattr(torch, "__version__", None))
try:
    import sklearn, catboost; env.update(sklearn=sklearn.__version__, catboost=catboost.__version__)
except Exception: pass
try: env["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
except Exception: pass
json.dump(env, open(OUT / "repro" / "environment.json", "w"), indent=2)
json.dump({k: (v if not isinstance(v, Path) else str(v)) for k, v in CFG.items()}, open(OUT / "repro" / "config.json", "w"), indent=2, default=str)
json.dump(dict(access_date=str(dt.date.today()), sources={k: str(v) for k, v in SRC.items()}, kaggle_ids=dict(
    D1="competitions/agriyield-2025", D2=("shaikasif89/wheat-yeild" if CFG["d2_source"] == "wheat" else "competitions/kaggle-cup-2025-01"), D3="abhinand05/crop-production-in-india", D4="siddharthss/crop-recommendation-dataset")),
    open(OUT / "repro" / "datasets.json", "w"), indent=2)
tuned_rows = []
for n in CHUNKS:
    if CHUNKS[n]: json.dump(CHUNKS[n][0]["fsets"], open(OUT / "repro" / f"feature_sets_{n}.json", "w"), indent=2)
    arrs = {}
    for c in CHUNKS[n]:
        for part in ("fit", "val", "cal", "test"): arrs[f"{c['split']}_f{c['fold']}_{part}"] = c[f"{part}_idx"]
        if c["tuned"]: tuned_rows.append(dict(dataset=n, split=c["split"], fold=c["fold"], **c["tuned"]["params"], val_rmse=c["tuned"]["val_rmse"], n_trials=c["tuned"]["n_trials"]))
    np.savez_compressed(OUT / "repro" / f"split_indices_{n}.npz", **arrs)
pd.DataFrame(tuned_rows).to_csv(OUT / "tables" / "optuna_best_params_per_fold.csv", index=False)
ens_rows = [dict(dataset=n, split=c["split"], fold=c["fold"], fset=fs, **w) for n in CHUNKS for c in CHUNKS[n] for fs, w in c["ens_w"].items()]
pd.DataFrame(ens_rows).to_csv(OUT / "tables" / "ensemble_weights.csv", index=False)
shutil.copy(OUT / "audit" / "split_availability.json", OUT / "repro" / "split_availability.json")
zip_path = shutil.make_archive(str(Path(CFG["out_dir"]).parent / "agri_results"), "zip", CFG["out_dir"])
log(f"ALL DONE. Total elapsed {elapsed_h():.2f} h. Results zipped at {zip_path}")
print(f"Projected {PROJECTED_H:.2f} h vs actual {elapsed_h():.2f} h")
