- # 🩺 Stroke Prediction — README

- ## Stroke Prediction UI
   - View Notebook in Google Colabs to display Plotly-based EBM plots: [Notebook](https://colab.research.google.com/drive/1Pxv_PBF13RxKFvxLJ4y-CfiuXESRCeoy?usp=sharing)
   - Browser UI for use: [Stroke Prediction App](https://stroke-prediction.azurewebsites.net/) (Disabled due to free-trial on Azure ending)
   - Provides form and CSV upload for input data to the deployed model.

- ## 📌 Project Overview

- Stroke prediction analysis and FastAPI service to serve a calibrated LinearSVC pipeline for prediction generation.
- Notebook: `stroke_prediction.ipynb` — full EDA, statistical inference, model development and serialization steps.
- Service & utilities: `scripts/` (FastAPI app and helper scripts).
- Data used: `healthcare-dataset-stroke-data.csv` (Kaggle: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).

- ## Requirements
- Python >= 3.12
- Poetry for dependency / environment management (poetry is used instead of pip)

- ## Quick setup (Windows)
1. Install Python 3.12+ and Poetry (if not installed):
   - Install Python 3.12 from python.org or via the Microsoft Store.
   - Install Poetry (recommended):  
     py -3.12 -m pip install --user poetry
2. Create environment and install dependencies:
   - From the repo root:
     ```powershell
     poetry install
     ```
   - To use a specific python executable with poetry:
     poetry env use C:\Path\To\Python312\python.exe
3. Activate shell (optional):
  ```powershell
  poetry shell
  ```
- ## Core dependencies (managed by Poetry)
- scikit-learn, pandas, numpy, matplotlib, seaborn, statsmodels, xgboost, imbalanced-learn, lightgbm, interpret, eli5, PyALE, missingno, category_encoders, fastapi, uvicorn, dill, pytest
(See pyproject.toml / poetry.lock in repo for exact versions.)

- ## Files & layout
- stroke_prediction.ipynb — main analysis and training notebook.
- .github/
  - workflows/
     - main_stroke-prediction.yml - Add or update the Azure App Service build and deployment workflow config (automatically added)
- scripts/
  - app.py — FastAPI service (endpoints: `/health`, `/load-model`, `/predict`, `/ui`).
  - templates/index.html — small browser UI for predictions / CSV upload.
  - validate_model.py — CLI to run model against CSV.
- data/
  - healthcare-dataset-stroke-data.csv — original dataset used for EDA and modeling.
- models/
  - lsvc_calibrated.dill — serialized production model (dill).
  - lsvc_calibrated_metadata.json — metadata (feature list, model_version).
- tests/
  - test_health.py — simple health endpoint test.

Data & input schema
- Source CSV: `healthcare-dataset-stroke-data.csv`
- Key features used by deployed model (input fields expected by service):
  - gender (Male | Female)
  - age (numeric)
  - hypertension (0 | 1)
  - heart_disease (0 | 1)
  - ever_married (Yes | No)
  - work_type (Private | Self-employed | Govt_job | other)
  - Residence_type (Urban | Rural)
  - avg_glucose_level (numeric)
  - bmi (numeric)
  - smoking_status (never smoked | ever_smoked | unknown)
- Target: `stroke` (0/1)
- The model metadata file `models/lsvc_calibrated_metadata.json` contains the feature order used by the pipeline — the FastAPI app aligns incoming data to that list before inference.

- ## Run the FastAPI service (development)
- From repo root:
  poetry run uvicorn scripts.app:app --host 0.0.0.0 --port 8000 --reload
- Or (if inside poetry shell):
  uvicorn scripts.app:app --host 0.0.0.0 --port 8000 --reload

- ## Predict endpoint (example)
- POST /predict
- JSON payload example:
  {
    "records": [
      {
        "gender": "Male",
        "age": 45,
        "hypertension": 0,
        "heart_disease": 0,
        "ever_married": "No",
        "work_type": "Private",
        "Residence_type": "Urban",
        "avg_glucose_level": 85.0,
        "bmi": 25.0,
        "smoking_status": "never smoked"
      }
    ]
  }
- Response (example): probabilities array, chosen threshold and recommendation for clinical checks.

- ## Model artifacts & serialization
- Final model serialized in notebook:
  - `models/lsvc_calibrated.dill`
  - `models/lsvc_calibrated_metadata.json` (keys: features, model_version, random_state, etc.)
- If model file is not checked into the repo, the service will attempt to download it if `MODEL_BLOB_URL` is set in the environment.

- ## Testing
- Run tests with Poetry:
  poetry run pytest -q

- ## Operational notes
- Feature alignment: the API uses metadata['features'] to enforce column order and missing-value handling before inference.
- Threshold tuning: recommended to store chosen threshold in metadata or app configuration for consistent behavior.

- ## Key findings
- Dominant predictors: **age** (strongest).
- Binary risk contributors: hypertension, heart disease, smoking_status.
- Class imbalance: stroke prevalence ~ 4–5% → optimize for recall (sensitivity) and average precision (AP).
- Best deployed model: calibrated **LinearSVC** (threshold tuning to meet recall > 0.8 as project objective).
- Typical performance (noted in notebook experiments): AUC ≈ 0.82–0.84, recall often > 0.8 for tuned SVC with calibration, precision trade-offs apply.

- ## How to retrain & update model
- Retrain in `stroke_prediction.ipynb`.
- Serialize with dill and update `models/lsvc_calibrated.dill` and `models/lsvc_calibrated_metadata.json` (update metadata features list and model_version).
- Redeploy artifact or update `MODEL_BLOB_URL` and call POST /load-model on the running service.

- ## 📚 References

* [Stroke Prediction Data](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) - Kaggle Dataset
* [Statsmodels Documentation](https://www.statsmodels.org/) - Statistical modeling in Python
* [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html) - Machine learning preprocessing and validation
* [Interpret Documentation](https://github.com/interpretml/interpret/blob/main/README.md) - Interpretable Machine Learning Model
* [missingno](https://github.com/ResidentMario/missingno/blob/master/README.md) - Quick Visual Summary Of The Dataset Completeness 
* [Imbalanced-learn Documentation](https://imbalanced-learn.org/) - Tools for imbalanced classification problems - Kaggle Dataset
* [PyALE](https://pypi.org/project/PyALE/) - ALE Plot Generation
* [eli5](https://pypi.org/project/eli5/) - Debug Machine Learning Classifiers And Explain Their Predictions
