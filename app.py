from flask import Flask, request, jsonify
import os
import pandas as pd

from train_model import train_and_save, load_model

MODEL_PATH = "model.joblib"
CSV_PATH = "covid_toy - covid_toy.csv"

app = Flask(__name__)


@app.before_first_request
def startup():
    if not os.path.exists(MODEL_PATH):
        train_and_save(CSV_PATH, MODEL_PATH)


@app.route("/", methods=["GET"])
def root():
    return jsonify(message="RandomForest COVID Toy API. POST /predict with patient JSON.")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    if not isinstance(data, dict):
        return jsonify(error="JSON object expected"), 400
    model = load_model(MODEL_PATH)
    df = pd.DataFrame([data])
    preds = model.predict(df)
    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(df)[:, 1]
    return jsonify(
        prediction=("Yes" if int(preds[0]) == 1 else "No"),
        probability=(float(probs[0]) if probs is not None else None),
    )


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    data = request.get_json(force=True)
    if not isinstance(data, list):
        return jsonify(error="JSON array expected"), 400
    model = load_model(MODEL_PATH)
    df = pd.DataFrame(data)
    preds = model.predict(df)
    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(df)[:, 1]
    results = []
    for i, p in enumerate(preds):
        results.append(
            {
                "prediction": ("Yes" if int(p) == 1 else "No"),
                "probability": (float(probs[i]) if probs is not None else None),
            }
        )
    return jsonify(results)


@app.route("/train", methods=["POST"])
def retrain():
    out = train_and_save(CSV_PATH, MODEL_PATH)
    return jsonify(status="trained", accuracy=out.get("accuracy"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
