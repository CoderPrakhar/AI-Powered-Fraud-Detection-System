"""
app.py
------
Flask REST API that serves the trained fraud detection model to the website.

Endpoints:
  GET  /api/health          -> health check
  GET  /api/metrics         -> model performance metrics
  GET  /api/transactions    -> recent sample transactions (for the dashboard table)
  POST /api/predict         -> predict fraud probability for a single transaction

Run:
  pip install -r ../requirements.txt
  python app.py
  Then open frontend/index.html in your browser (it calls http://localhost:5000)
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

app = Flask(__name__)


@app.after_request
def add_cors_headers(response):
    # Allow the frontend (served from file:// or a different port) to call this API
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
FEATURES = joblib.load(os.path.join(MODEL_DIR, "features.pkl"))

with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
    METRICS = json.load(f)

transactions_df = pd.read_csv(os.path.join(DATA_DIR, "transactions.csv"))


def score_transaction(payload: dict):
    row = [[float(payload.get(feat, 0)) for feat in FEATURES]]
    scaled = scaler.transform(row)
    proba = float(model.predict_proba(scaled)[0][1])
    pred = int(proba >= payload.get("threshold", 0.5))

    if proba >= 0.75:
        risk = "High"
    elif proba >= 0.4:
        risk = "Medium"
    else:
        risk = "Low"

    return {
        "fraud_probability": round(proba, 4),
        "is_fraud": bool(pred),
        "risk_level": risk,
    }


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model_loaded": True})


@app.route("/api/metrics")
def metrics():
    return jsonify(METRICS)


@app.route("/api/transactions")
def transactions():
    limit = int(request.args.get("limit", 25))
    sample = transactions_df.sample(n=min(limit, len(transactions_df)), random_state=None)
    scaled = scaler.transform(sample[FEATURES])
    probs = model.predict_proba(scaled)[:, 1]

    records = []
    for (_, row), prob in zip(sample.iterrows(), probs):
        records.append({
            "transaction_id": row["transaction_id"],
            "amount": row["amount"],
            "distance_from_home": row["distance_from_home"],
            "online_order": bool(row["online_order"]),
            "used_chip": bool(row["used_chip"]),
            "actual_fraud": bool(row["fraud"]),
            "predicted_fraud_probability": round(float(prob), 4),
            "predicted_fraud": bool(prob >= 0.5),
        })
    records.sort(key=lambda r: r["predicted_fraud_probability"], reverse=True)
    return jsonify(records)


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    missing = [f for f in FEATURES if f not in payload]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400
    result = score_transaction(payload)
    return jsonify(result)


if __name__ == "__main__":
    print("Fraud Detection API starting on http://localhost:5000")
    app.run(debug=True, port=5000, use_reloader=False)
