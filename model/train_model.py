"""
train_model.py
---------------
Trains a fraud detection model on transactions.csv and saves it as model.pkl
so the Flask API (backend/app.py) can load it and serve live predictions.

Model: RandomForestClassifier
- Handles non-linear patterns well
- Gives feature importance (explainability - important for fraud review teams)
- class_weight='balanced' to handle the fraud/legit imbalance

Run: python train_model.py
Outputs:
  - model.pkl              (trained model)
  - scaler.pkl             (feature scaler)
  - metrics.json           (accuracy/precision/recall/f1/roc_auc)
  - confusion_matrix.png
  - feature_importance.png
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_score, recall_score, f1_score, accuracy_score
)

FEATURES = [
    "amount", "distance_from_home", "distance_from_last_transaction",
    "ratio_to_median_purchase", "repeat_retailer", "used_chip",
    "used_pin", "online_order", "hour_of_day",
]

df = pd.read_csv("../data/transactions.csv")
X = df[FEATURES]
y = df["fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

metrics = {
    "accuracy": round(accuracy_score(y_test, y_pred), 4),
    "precision": round(precision_score(y_test, y_pred), 4),
    "recall": round(recall_score(y_test, y_pred), 4),
    "f1_score": round(f1_score(y_test, y_pred), 4),
    "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    "n_train": len(X_train),
    "n_test": len(X_test),
    "fraud_in_test": int(y_test.sum()),
}

print("=== Fraud Detection Model - Evaluation ===")
print(json.dumps(metrics, indent=2))
print("\nClassification report:\n", classification_report(y_test, y_pred, digits=3))

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4.5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Legit", "Fraud"]); ax.set_yticklabels(["Legit", "Fraud"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                 color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=14)
plt.colorbar(im)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

# Feature importance plot
importances = model.feature_importances_
order = np.argsort(importances)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.barh(np.array(FEATURES)[order], importances[order], color="#2563eb")
ax.set_title("Feature Importance")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(FEATURES, "features.pkl")

print("\nSaved: model.pkl, scaler.pkl, features.pkl, metrics.json, confusion_matrix.png, feature_importance.png")
