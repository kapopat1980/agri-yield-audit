#!/usr/bin/env python3
"""Audit check: moment-implied uniform bounds (mean +/- sqrt(3)*SD) and noise bounds for correlations.

For a variable that is uniform on [a, b], mean +/- sqrt(3)*SD recovers (a, b). Round-number implied endpoints for the target and for
every predictor are unusual for field data and are a prompt to inspect histograms and the dataset documentation.
This is evidence of consistency with uniform draws, not proof of how a file was generated.

Needs only results/tables/Table2_data_dictionary.csv (written by the notebook).
Usage: python scripts/audit_moments.py [results_dir]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(sys.argv[1] if len(sys.argv) > 1 else "results") / "tables"
t = pd.read_csv(R / "Table2_data_dictionary.csv")
t["implied_lo"] = t["mean"] - np.sqrt(3) * t["sd"]
t["implied_hi"] = t["mean"] + np.sqrt(3) * t["sd"]
print(t[["dataset", "feature", "block", "mean", "sd", "implied_lo", "implied_hi"]].round(3).to_string(index=False))
for d, n in (("D1", 10000), ("D2", 200000)):
    print(f"{d}: n={n}  95% noise bound for a correlation = {1.96 / np.sqrt(n - 3):.4f}")
