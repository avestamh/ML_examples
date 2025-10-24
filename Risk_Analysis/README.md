# Bankruptcy Prediction (Supplier Risk Analysis)

A small project for experimenting with supplier bankruptcy / financial instability prediction using tabular models (LightGBM) and neural approaches (CNN/LSTM). The repository contains data generation utilities, training scripts, pre-trained model artifacts, and a simple app/dashboard to visualize predictions.

## Contents

- `app.py` — Simple web app / dashboard to view predictions (entry point for demo).
- `train.py` — Training orchestration for models.
- `train_lightGB_CNN.py` — Training pipeline for LightGBM + CNN ensemble.
- `generate_data.py` — Generate mock/train data used for experiments.
- `generate_new_data_for_test.py` — Generate example new supplier data for inference/testing.
- `lightgbm_model.py`, `lstm_model.py`, `cnn_model.py` — Model definitions / wrappers.
- `final_model.txt`, `final_model_lgb.txt`, `final_model_type.txt` — Saved model files / metadata.
- `cnn_model.pth` — Saved PyTorch CNN weights.
- `ensemble_weights.npy` — Ensemble weights used to combine model outputs.
- `supplier_financial_data.csv` — Example dataset used for training.
- `supplier_financial_data_predictions.csv`, `new_supplier_predictions.csv` — Example prediction outputs.
- `dashboard.py` — Alternative dashboard implementation (Dash/Flask components).
- `README.md`, `requirements.txt` — This file and dependency list.

## Quick start

1. Create a Python 3.8+ virtual environment and activate it (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

If you prefer, you can install the core dependencies manually:

```bash
pip install pandas numpy requests lightgbm torch torchvision dash flask scikit-learn
```

3. (Optional) Generate mock data used by training and examples:

```bash
python generate_data.py
```

4. Train models (example):

```bash
python train.py
# or for the LightGBM+CNN ensemble
python train_lightGB_CNN.py
```

5. Run the app/dashboard to view predictions:

```bash
python app.py
# then open http://127.0.0.1:8050 or the port printed by the app
```

6. Run inference on new supplier data:

```bash
python generate_new_data_for_test.py
# or run a prediction script that loads model files (e.g. use `lightgbm_model.py`/`cnn_model.py` wrappers)
```

## Files and models

- Pretrained artifacts included for convenience: `cnn_model.pth`, `final_model_lgb.txt`, and `final_model.txt`.
- Prediction outputs are stored in `supplier_financial_data_predictions.csv` and `new_supplier_predictions.csv`.

## Notes, assumptions, and troubleshooting

- The repository mixes experimental code (not production-ready) and demo dashboards. Expect quick scripts in the root that show how the pieces connect.
- If a script fails due to a missing dependency, install it into the active venv and rerun.
- If memory / GPU issues occur when loading `cnn_model.pth`, ensure `torch` is installed for your platform and try forcing CPU usage (set device to `cpu` in the model loader).

## How I checked the code (quick diagnostics)

- Ran a syntax/compile check across all `.py` files in the repo; no syntax errors were detected.

## Contributing

If you'd like changes or a README section expanded (examples, API docs, tests), open an issue or ask — I can add runnable examples or tests.

## License

Specify a license if this project will be shared. No license is included by default.

---

If you'd like, I can also:
- add a small `scripts/` folder with convenience commands to train and run the app,
- add a `Makefile` or `tox` configuration for reproducible runs, or
- generate minimal unit tests for critical modules (`lightgbm_model.py`, `cnn_model.py`).
# Bankruptcy_prediction

This repository is for predicting company bankruptcy using mock data.

## Implementation Details

### Data Acquisition & Processing
- Load mock financial and risk data (debt-to-equity, Z-score, news alerts).
- Integrate balance sheet data (from OpenBB or a CSV source).
- Extract economic indicators (inflation, GDP, currency rates) from a mock API.

### Bankruptcy Prediction Model
- Utilize a LightGBM or CNN model for bankruptcy risk analysis.
- Train on historical financial data with a binary target (bankrupt: 1, not bankrupt: 0).

### Real-Time Risk Detection
- Process Bloomberg-style alerts for supplier risk analysis.
- Use NLP to extract risks from economic/news alerts.

### Supplier Stability Dashboard
- Create a Flask/Dash dashboard to display supplier risk (color-coded indicators).

### Required Libraries
```bash
pip install pandas numpy requests openai lightgbm torch torchvision transformers dash flask
```
### How to run everythin
```bash
python generate_data.py
python train.py
python app.py