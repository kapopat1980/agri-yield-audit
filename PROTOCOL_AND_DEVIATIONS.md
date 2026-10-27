# Protocol and deviations

The experiments followed a written protocol with research questions RQ1-RQ4, hypotheses H1-H4 and six decision gates. Deviations and outcomes:

| Item | Plan | What happened |
|---|---|---|
| D2 | Competition `kaggle-cup-2025-01` (field, date, coordinates, satellite indices) | Not accessible (invitation required). Substituted the Kaggle wheat dataset, which has no soil, date or repeated observations |
| Group split (S2) | Leakage-safe split by field | Degenerate: every field (D1) or coordinate pair (D2) occurs once, so S2 behaves like a random split |
| Temporal split (S3), E5 temporal features | Planned | Not possible: no date or repeated observations |
| Spatial split (S4) | Planned | D2 only (D1 has no coordinates or region) |
| Gate 3 (multimodal gain) | Required gain to keep the multimodal narrative | Failed in both datasets; the manuscript was reframed as an audit / negative-result paper, as the protocol prescribes |
| Control experiments | Not in the protocol | Label-permutation and shuffled-feature controls were not run |
| Software | scikit-learn `squared=False` in the original snippet | Replaced by a manual RMSE (the argument was removed in newer scikit-learn) |
| Linear models | float32 inputs | ElasticNet now runs in float64 with `precompute=False` after a Gram-matrix validation error on the Kaggle image; a failing model is skipped instead of aborting the run |

Decision-gate outcomes are in `results/tables/decision_gates.json`.
