
**Table 3. Test RMSE (mean ± SD over folds), final feature set E4**

| Model | D1 S1 | D1 S2 | D1 S5 | D2 S1 | D2 S2 | D2 S4 | D2 S5 |
|---|---|---|---|---|---|---|---|
| Mean predictor | 436.587±5.698 | 436.514±3.647 | 436.532±0.000 | 23.083±0.030 | 23.083±0.063 | 23.084±0.043 | 23.028±0.000 |
| Linear (OLS) | 437.001±5.708 | 436.933±3.879 | 438.719±0.000 | 23.087±0.030 | 23.088±0.063 | 23.086±0.041 | 23.031±0.000 |
| Ridge | 436.907±5.732 | 436.739±3.955 | 438.634±0.000 | 23.086±0.030 | 23.086±0.062 | 23.086±0.041 | 23.030±0.000 |
| Elastic Net | 436.609±5.674 | 436.547±3.692 | 437.537±0.000 | 23.084±0.030 | 23.083±0.063 | 23.084±0.043 | 23.028±0.000 |
| Random forest | 444.474±5.311 | 445.133±4.172 | 445.360±0.000 | 23.184±0.030 | 23.190±0.066 | 23.193±0.043 | 23.152±0.000 |
| Extra Trees | 446.396±5.499 | 447.129±3.885 | 450.055±0.000 | 23.194±0.027 | 23.188±0.061 | 23.194±0.047 | 23.132±0.000 |
| XGBoost | 437.326±6.148 | 436.800±3.466 | 437.646±0.000 | 23.084±0.031 | 23.083±0.063 | 23.086±0.045 | 23.028±0.000 |
| LightGBM | 436.875±5.782 | 437.102±2.934 | 436.288±0.000 | 23.084±0.030 | 23.084±0.062 | 23.084±0.043 | 23.028±0.000 |
| CatBoost | 436.810±5.894 | 436.692±3.485 | 438.686±0.000 | 23.083±0.031 | 23.084±0.063 | 23.084±0.042 | 23.028±0.000 |
| MLP | 436.968±5.446 | 437.266±3.817 | 437.953±0.000 | 23.087±0.029 | 23.089±0.062 | 23.087±0.040 | 23.028±0.000 |
| XGBoost (Optuna-tuned) | 437.235±5.720 | 437.336±3.275 | 438.409±0.000 | 23.085±0.031 | 23.085±0.064 | 23.087±0.042 | 23.033±0.000 |
| Ensemble | 436.921±5.802 | 437.555±3.000 | 437.680±0.000 | 23.085±0.030 | 23.087±0.063 | 23.086±0.042 | 23.028±0.000 |

**Table 4. RMSE by cumulative feature set (XGBoost and Ensemble)**

| dataset | split | model | E1 | E2 | E3 | E4 |
|---|---|---|---|---|---|---|
| D1 | S1_random | Ensemble | 437.017 ± 5.760 | 436.814 ± 5.675 | 437.167 ± 5.908 | 436.921 ± 5.802 |
| D1 | S1_random | XGBoost | 436.818 ± 5.784 | 437.163 ± 5.994 | 436.844 ± 5.623 | 437.326 ± 6.148 |
| D1 | S2_group | Ensemble | 436.761 ± 3.716 | 437.386 ± 3.935 | 437.633 ± 3.838 | 437.555 ± 3.000 |
| D1 | S2_group | XGBoost | 436.465 ± 3.794 | 436.918 ± 4.019 | 437.504 ± 3.276 | 436.800 ± 3.466 |
| D1 | S5_ood | Ensemble | 439.104 ± 0.000 | 437.634 ± 0.000 | 437.480 ± 0.000 | 437.680 ± 0.000 |
| D1 | S5_ood | XGBoost | 436.758 ± 0.000 | 436.542 ± 0.000 | 436.040 ± 0.000 | 437.646 ± 0.000 |
| D2 | S1_random | Ensemble | 23.085 ± 0.032 | n/a | 23.084 ± 0.031 | 23.085 ± 0.030 |
| D2 | S1_random | XGBoost | 23.084 ± 0.031 | n/a | 23.084 ± 0.031 | 23.084 ± 0.031 |
| D2 | S2_group | Ensemble | 23.086 ± 0.063 | n/a | 23.085 ± 0.062 | 23.087 ± 0.063 |
| D2 | S2_group | XGBoost | 23.086 ± 0.065 | n/a | 23.084 ± 0.064 | 23.083 ± 0.063 |
| D2 | S4_spatial | Ensemble | 23.085 ± 0.043 | n/a | 23.086 ± 0.042 | 23.086 ± 0.042 |
| D2 | S4_spatial | XGBoost | 23.084 ± 0.043 | n/a | 23.084 ± 0.043 | 23.086 ± 0.045 |
| D2 | S5_ood | Ensemble | 23.028 ± 0.000 | n/a | 23.030 ± 0.000 | 23.028 ± 0.000 |
| D2 | S5_ood | XGBoost | 23.028 ± 0.000 | n/a | 23.028 ± 0.000 | 23.028 ± 0.000 |

**Table 5. Paired comparison E1 → E4 (negative favours the richer feature set)**

| dataset | split | comparison | rel_change_pct | CI95_lo | CI95_hi | wilcoxon_p_holm | cohen_dz |
|---|---|---|---|---|---|---|---|
| D1 | S1_random | XGBoost: E1 → E4 | 0.1172 | 0.0267 | 0.9895 | 1.0 | -0.0139 |
| D1 | S1_random | Ensemble: E1 → E4 | -0.0218 | -0.5551 | 0.3704 | 1.0 | 0.0045 |
| D1 | S2_group | XGBoost: E1 → E4 | 0.0762 | -0.0817 | 0.7409 | 1.0 | -0.0108 |
| D1 | S2_group | Ensemble: E1 → E4 | 0.181 | 0.3156 | 1.2665 | 0.6194 | -0.0239 |
| D1 | S5_ood | XGBoost: E1 → E4 | 0.2032 | -0.3369 | 2.0174 | 1.0 | -0.0248 |
| D1 | S5_ood | Ensemble: E1 → E4 | -0.3244 | -2.3737 | -0.4943 | 1.0 | 0.039 |
| D2 | S1_random | XGBoost: E1 → E4 | -0.0001 | -0.0009 | 0.0007 | 1.0 | 0.0002 |
| D2 | S1_random | Ensemble: E1 → E4 | -0.0007 | -0.0015 | 0.0011 | 1.0 | 0.0003 |
| D2 | S2_group | XGBoost: E1 → E4 | -0.0103 | -0.0038 | -0.001 | 1.0 | 0.0065 |
| D2 | S2_group | Ensemble: E1 → E4 | 0.0033 | -0.001 | 0.0024 | 1.0 | -0.0006 |
| D2 | S4_spatial | XGBoost: E1 → E4 | 0.0085 | 0.0002 | 0.0036 | 1.0 | -0.0036 |
| D2 | S4_spatial | Ensemble: E1 → E4 | 0.0049 | -0.0002 | 0.0025 | 1.0 | -0.004 |
| D2 | S5_ood | XGBoost: E1 → E4 | 0.001 | -0.0011 | 0.0017 | 1.0 | -0.0067 |
| D2 | S5_ood | Ensemble: E1 → E4 | -0.0012 | -0.003 | 0.0024 | 1.0 | -0.0011 |

**Table 6. 90% prediction intervals**

| dataset | split | method | PICP | MPIW_pct_of_range | spearman_width_vs_abs_err |
|---|---|---|---|---|---|
| D1 | S2_group | SplitConf-Ensemble | 0.8949 | 89.8416 | -0.0034 |
| D1 | S2_group | NormConf-Ensemble | 0.9003 | 138.9217 | -0.0088 |
| D1 | S2_group | CQR-LightGBM | 0.8952 | 89.5824 | -0.004 |
| D1 | S5_ood | SplitConf-Ensemble | 0.897 | 90.0661 |  |
| D1 | S5_ood | NormConf-Ensemble | 0.9385 | 144.8182 | 0.0167 |
| D1 | S5_ood | CQR-LightGBM | 0.7495 | 75.1773 | 0.012 |
| D1 | S1_random | SplitConf-Ensemble | 0.9034 | 90.4326 | 0.0043 |
| D1 | S1_random | NormConf-Ensemble | 0.9015 | 141.4858 | -0.0123 |
| D1 | S1_random | CQR-LightGBM | 0.9 | 89.9875 | -0.0019 |
| D2 | S2_group | SplitConf-Ensemble | 0.899 | 89.8307 | 0.0005 |
| D2 | S2_group | NormConf-Ensemble | 0.8989 | 130.1461 | -0.0005 |
| D2 | S2_group | CQR-LightGBM | 0.8989 | 89.8265 | -0.0011 |
| D2 | S4_spatial | SplitConf-Ensemble | 0.8994 | 89.8474 | -0.0011 |
| D2 | S4_spatial | NormConf-Ensemble | 0.9002 | 129.0432 | 0.0015 |
| D2 | S4_spatial | CQR-LightGBM | 0.8989 | 89.8532 | -0.0031 |
| D2 | S5_ood | SplitConf-Ensemble | 0.8992 | 89.8301 |  |
| D2 | S5_ood | NormConf-Ensemble | 0.9105 | 139.5159 | -0.0015 |
| D2 | S5_ood | CQR-LightGBM | 0.8842 | 88.305 | -0.0072 |
| D2 | S1_random | SplitConf-Ensemble | 0.899 | 89.8111 | -0.0025 |
| D2 | S1_random | NormConf-Ensemble | 0.8968 | 129.6444 | -0.0042 |
| D2 | S1_random | CQR-LightGBM | 0.8985 | 89.7637 | -0.0005 |

**Table 7. Range of RMSE ratio (perturbed / clean)**

| dataset | model | min | max |
|---|---|---|---|
| D1 | CatBoost | 0.9995 | 1.0012 |
| D1 | Ensemble | 0.999 | 1.0063 |
| D1 | LightGBM | 0.9987 | 1.0015 |
| D1 | MLP | 0.9994 | 1.0256 |
| D1 | RF | 0.9922 | 1.0067 |
| D1 | Ridge | 0.9998 | 1.0037 |
| D1 | XGBoost | 0.9991 | 1.0026 |
| D2 | CatBoost | 1.0 | 1.0001 |
| D2 | Ensemble | 1.0 | 1.0001 |
| D2 | LightGBM | 1.0 | 1.0001 |
| D2 | MLP | 0.9999 | 1.0003 |
| D2 | RF | 0.9997 | 1.0021 |
| D2 | Ridge | 1.0 | 1.0003 |
| D2 | XGBoost | 1.0 | 1.0001 |

**Table 8. Top SHAP features (group split)**

| dataset | rank | feature | block | mean_abs_shap | mean_abs_shap_pct_of_target_SD | share_pct |
|---|---|---|---|---|---|---|
| D1 | 1 | x_rs_x_rain | engineered | 16.026 | 3.671 | 11.049 |
| D1 | 2 | soil_ph | soil | 15.12 | 3.464 | 10.424 |
| D1 | 3 | sand_pct | soil | 14.612 | 3.347 | 10.074 |
| D1 | 4 | rainfall | weather | 13.927 | 3.191 | 9.602 |
| D1 | 5 | stress_idx | engineered | 13.73 | 3.145 | 9.466 |
| D1 | 6 | organic_matter | soil | 13.6 | 3.116 | 9.377 |
| D2 | 1 | windBearing | weather | 0.098 | 0.424 | 5.74 |
| D2 | 2 | x_rs_x_rain | engineered | 0.09 | 0.392 | 5.306 |
| D2 | 3 | humidity | weather | 0.088 | 0.382 | 5.179 |
| D2 | 4 | x_temp_x_hum | engineered | 0.079 | 0.344 | 4.659 |
| D2 | 5 | temperatureMin | weather | 0.079 | 0.341 | 4.626 |
| D2 | 6 | dewPoint | weather | 0.077 | 0.334 | 4.526 |
