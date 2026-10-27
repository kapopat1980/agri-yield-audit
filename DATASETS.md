# Data sources

Access date: 23 October 2026. Raw files are not redistributed here. Check each source's licence and, for D1, the competition rules before reuse or publication.

| ID | Role | Kaggle source | URL |
|---|---|---|---|
| D1 | Primary modelling dataset | Competition `agriyield-2025` ("AgriYield: Predict Crop Yield from Soil, Weather") | https://www.kaggle.com/competitions/agriyield-2025/data |
| D2 | Second modelling dataset (substitute) | Dataset `shaikasif89/wheat-yeild` ("Wheat Yield Prediction") | https://www.kaggle.com/datasets/shaikasif89/wheat-yeild |
| D3 | Context only (India production) | Dataset `abhinand05/crop-production-in-india` | https://www.kaggle.com/abhinand05/crop-production-in-india |
| D4 | Context only (crop envelopes) | Dataset `siddharthss/crop-recommendation-dataset` | https://www.kaggle.com/siddharthss/crop-recommendation-dataset |

## What is and is not known

- **D1.** The competition page describes the task as forecasting maize yield from soil, climate and satellite data. The files used had 10,000 training rows, seven predictors, no date, coordinate or region column, and one row per `field_id`. Units and generation method were not available to the authors.
- **D2.** The Kaggle page subtitle is "Leveraging Data for Predictive Analytics in Indian Agriculture". The file has 200,000 rows, latitude and longitude, 18 weather columns in a Dark Sky-style naming scheme, NDVI, DayInSeason and the target `Yield`. No date, soil or field-identifier column; each coordinate pair occurs once. Units, collection method and provenance were not available to the authors.
- **D2 substitution.** The protocol designated the competition *kaggle-cup-2025-01*, which required an invitation, so the wheat dataset was used instead.
- **Distribution check.** For both targets and every continuous predictor tabulated, mean ± √3·SD recovers round-number bounds, which is consistent with independent uniform draws (`python scripts/audit_moments.py`). This does not establish how the files were produced.
- **D3 and D4** were used only for context and were never merged row-wise with D1 or D2. D4's page states that it was built by augmenting rainfall, climate and fertiliser data for India.

## Kaggle paths in the reported run

```
D1  /kaggle/input/competitions/agriyield-2025
D2  /kaggle/input/datasets/shaikasif89/wheat-yeild  (copied by the notebook to /kaggle/working/D2_wheat/train.csv)
D3  /kaggle/input/datasets/abhinand05/crop-production-in-india/crop_production.csv
D4  /kaggle/input/datasets/siddharthss/crop-recommendation-dataset/Crop_recommendation.csv
```
