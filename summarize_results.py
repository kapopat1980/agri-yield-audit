#!/usr/bin/env python3
"""Recompute the headline numbers and the paper's result tables from the saved CSV outputs.

Usage:  python scripts/summarize_results.py [results_dir]   (default: results)
Writes: <results_dir>/paper_tables.md  and prints the headline statistics.
Only reads CSV files produced by the notebook; it does not need the raw Kaggle data.
"""
import sys
from pathlib import Path

import pandas as pd

R = Path(sys.argv[1] if len(sys.argv) > 1 else "results") / "tables"
OUT = R.parent / "paper_tables.md"
MODELS = ["Mean", "Linear", "Ridge", "ElasticNet", "RF", "ExtraTrees", "XGBoost", "LightGBM", "CatBoost", "MLP", "XGB_Tuned", "Ensemble"]
NAMES = {"Mean": "Mean predictor", "Linear": "Linear (OLS)", "ElasticNet": "Elastic Net", "RF": "Random forest", "ExtraTrees": "Extra Trees",
         "XGB_Tuned": "XGBoost (Optuna-tuned)"}
RANGE = {"D1": 1499.8, "D2": 79.9994}          # observed target range (Table 1 of the paper)
SD = {"D1": 436.51, "D2": 23.083}              # target SD (Table 1 of the paper)
out = []


def md(df):
    """Markdown table without needing the optional `tabulate` package."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |")
    return "\n".join(lines)


def headline():
    m = pd.read_csv(R / "metrics_by_fold.csv")
    base = m[m.model == "Mean"].set_index(["dataset", "split", "fold", "fset"]).RMSE
    n = m[m.model != "Mean"].copy()
    n["rel"] = [r.RMSE / base[(r.dataset, r.split, r.fold, r.fset)] for r in n.itertuples()]
    print("== Headline statistics (non-Mean fits vs. the mean predictor on the same fold and feature set) ==")
    for d, g in n.groupby("dataset"):
        best = g.loc[g.rel.idxmin()]
        print(f"{d}: fits={len(g)}  share with RMSE below mean predictor={100 * (g.rel < 1).mean():.1f}%  "
              f"best relative RMSE={best.rel:.5f} ({best.model}, {best.fset}, {best.split}, fold {best.fold})  "
              f"R2 range=[{g.R2.min():.4f}, {g.R2.max():.4f}]")
    med = n.groupby(["dataset", "model"]).rel.median().unstack(0)
    print("\nMedian RMSE relative to the mean predictor (1.0 = identical):\n", med.round(4).to_string())
    s = pd.read_csv(R / "Table_paired_statistics.csv")
    print(f"\nPaired comparisons: {len(s)}; bootstrap CI excludes 0 in {int(((s.CI95_lo > 0) | (s.CI95_hi < 0)).sum())}; "
          f"Holm-significant (Wilcoxon) {int((s.wilcoxon_p_holm < 0.05).sum())}; "
          f"median |relative change|={s.rel_change_pct.abs().median():.3f}%")
    print("Holm-significant comparisons:", s[s.wilcoxon_p_holm < 0.05].comparison.unique().tolist())


def table(title, df):
    out.append(f"\n**{title}**\n")
    out.append(md(df))


def main():
    headline()
    t3 = pd.read_csv(R / "Table3_baseline_performance_final_feature_set.csv")
    cols = [("D1", "S1_random"), ("D1", "S2_group"), ("D1", "S5_ood"),
            ("D2", "S1_random"), ("D2", "S2_group"), ("D2", "S4_spatial"), ("D2", "S5_ood")]
    rows = []
    for mod in MODELS:
        row = {"Model": NAMES.get(mod, mod)}
        for d, s in cols:
            r = t3[(t3.dataset == d) & (t3.split == s) & (t3.model == mod)]
            row[f"{d} {s.split('_')[0]}"] = r.RMSE.iloc[0].replace(" ± ", "±") if len(r) else "-"
        rows.append(row)
    table("Table 3. Test RMSE (mean ± SD over folds), final feature set E4", pd.DataFrame(rows))

    t4 = pd.read_csv(R / "Table4_ablation_E1_E5_RMSE.csv")
    table("Table 4. RMSE by cumulative feature set (XGBoost and Ensemble)", t4[t4.model.isin(["XGBoost", "Ensemble"])].fillna("n/a"))

    s = pd.read_csv(R / "Table_paired_statistics.csv")
    x = s[s.comparison.isin(["XGBoost: E1 → E4", "Ensemble: E1 → E4"])]
    table("Table 5. Paired comparison E1 → E4 (negative favours the richer feature set)",
          x[["dataset", "split", "comparison", "rel_change_pct", "CI95_lo", "CI95_hi", "wilcoxon_p_holm", "cohen_dz"]].round(4))

    c = pd.read_csv(R / "Table7_uncertainty_calibration.csv")
    c = c[c.method.isin(["SplitConf-Ensemble", "NormConf-Ensemble", "CQR-LightGBM"])].copy()
    c["MPIW_pct_of_range"] = [100 * r.MPIW / RANGE[r.dataset] for r in c.itertuples()]
    table("Table 6. 90% prediction intervals",
          c[["dataset", "split", "method", "PICP", "MPIW_pct_of_range", "spearman_width_vs_abs_err"]].round(4))

    rb = pd.read_csv(R / "Table6_robustness_all.csv")
    rb = rb[rb.kind != "clean"]
    table("Table 7. Range of RMSE ratio (perturbed / clean)", rb.groupby(["dataset", "model"]).RMSE_ratio.agg(["min", "max"]).round(4).reset_index())

    t8 = pd.read_csv(R / "Table8_shap_top_drivers.csv")
    t8 = t8[(t8.split == "S2_group") & (t8["rank"] <= 6)].copy()
    t8["mean_abs_shap_pct_of_target_SD"] = [100 * r.mean_abs_shap / SD[r.dataset] for r in t8.itertuples()]
    table("Table 8. Top SHAP features (group split)",
          t8[["dataset", "rank", "feature", "block", "mean_abs_shap", "mean_abs_shap_pct_of_target_SD", "share_pct"]].round(3))
    OUT.write_text("\n".join(out) + "\n")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
