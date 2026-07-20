# Sentry — AI-Powered Financial Fraud Detection System

An end-to-end fraud detection project: a trained machine learning model, a REST
API that serves it, and a dashboard website to monitor transactions and test
the model live.

## What's inside

```
fraud_detection_project/
├── data/
│   ├── generate_data.py      # builds a synthetic transaction dataset
│   └── transactions.csv      # generated dataset (20,000 transactions, ~3% fraud)
├── model/
│   ├── train_model.py        # trains and evaluates the model
│   ├── model.pkl             # trained RandomForestClassifier
│   ├── scaler.pkl            # feature scaler
│   ├── features.pkl          # feature name order
│   ├── metrics.json          # accuracy / precision / recall / F1 / ROC-AUC
│   ├── confusion_matrix.png
│   └── feature_importance.png
├── backend/
│   └── app.py                # Flask REST API that serves the model
├── frontend/
│   ├── index.html            # dashboard website
│   └── app.js                # dashboard logic (fetches from the API)
└── requirements.txt
```

## How it works

1. **Data** — `generate_data.py` creates realistic transaction records (amount,
   distance from home, distance from the last transaction, spend ratio,
   retailer history, chip/PIN/online usage, time of day) with the same
   statistical patterns real fraud datasets have: fraud is rare (~3%), and it
   correlates with large amounts, unfamiliar locations, and card-not-present
   purchases.
2. **Model** — `train_model.py` trains a `RandomForestClassifier` (scikit-learn)
   on that data, evaluates it on a held-out test set, and saves the trained
   model plus evaluation charts.
3. **API** — `backend/app.py` is a Flask server that loads the trained model
   and exposes it over HTTP: `/api/predict` scores a single transaction,
   `/api/transactions` returns a scored sample feed, `/api/metrics` returns
   the evaluation numbers.
4. **Website** — `frontend/index.html` is the dashboard: a live "risk pulse"
   monitor, a scored transaction feed, an interactive "check a transaction"
   tool with a live risk gauge, and a model performance panel. It calls the
   Flask API — and if the API isn't running, it automatically falls back to
   an equivalent scoring model that runs directly in the browser, so the demo
   always works even without a backend.

## Running it yourself

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Regenerate the data and retrain the model
The trained model is already included, but you can rebuild it from scratch:
```bash
cd data
python generate_data.py

cd ../model
python train_model.py
```

### 3. Start the API
```bash
cd backend
python app.py
```
This starts the API at `http://localhost:5000`.

### 4. Open the website
Just open `frontend/index.html` in your browser (double-click it, or
right-click → Open With → your browser). It will automatically detect and
connect to the API running on `localhost:5000`.

> No backend running? The site still works — it switches to "offline demo
> mode" and scores transactions using an equivalent model in JavaScript.

## Model performance

| Metric | Score |
|---|---|
| Accuracy | 99.98% |
| Precision | 100% |
| Recall | 99.3% |
| F1 score | 99.7% |
| ROC AUC | 1.00 |

These are very high because the synthetic training data has cleanly
separable fraud patterns. On real, messier data you should expect meaningfully
lower — and more realistic — numbers, especially recall.

## Extending this project

Ideas if you want to take this further for a class project or portfolio piece:
- Swap in a real, anonymized dataset (e.g. Kaggle's "Credit Card Fraud
  Detection" dataset) instead of the synthetic one.
- Add an `IsolationForest` or `AutoEncoder` for unsupervised anomaly
  detection, and compare it against the supervised RandomForest.
- Persist scored transactions to a real database instead of sampling from
  the CSV.
- Add authentication to the API and deploy it (e.g. Render, Railway, or a
  small VM) so the dashboard works from anywhere, not just localhost.
- Add SHAP values for per-transaction explainability instead of the
  rule-based "what drove this score" list.

## Disclaimer

This project uses **synthetic data** for educational purposes. It is a
demonstration of the fraud-detection pipeline (data → model → API → UI), not
a production-ready system. Real fraud detection systems handle sensitive
financial data and need proper security, compliance (PCI-DSS), monitoring,
and much larger, real-world training data.
