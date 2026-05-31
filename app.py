from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

from database.db import (
    save_prediction,
    get_prediction_history
)

app = Flask(__name__)

# Load trained model
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")


# --------------------------------------------------
# Prediction Function
# --------------------------------------------------
def predict_ticket(text):

    text_vector = vectorizer.transform([text])

    prediction = model.predict(text_vector)[0]

    confidence = 0
    top_predictions = []

    if hasattr(model, "predict_proba"):

        probs = model.predict_proba(text_vector)[0]

        confidence = round(np.max(probs) * 100, 2)

        top_indices = np.argsort(probs)[::-1][:3]

        for idx in top_indices:

            top_predictions.append({
                "category": model.classes_[idx],
                "confidence": round(probs[idx] * 100, 2)
            })

    return prediction, confidence, top_predictions


# --------------------------------------------------
# Home Page
# --------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------------------------
# Prediction API
# --------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    ticket_text = data.get("text", "").strip()

    if ticket_text == "":
        return jsonify({
            "error": "Ticket text cannot be empty"
        }), 400

    prediction, confidence, top_predictions = predict_ticket(ticket_text)

    # Save prediction to SQLite
    save_prediction(
        ticket_text,
        prediction,
        confidence
    )

    return jsonify({
        "ticket": ticket_text,
        "prediction": prediction,
        "confidence": confidence,
        "top_predictions": top_predictions
    })


# --------------------------------------------------
# History API
# --------------------------------------------------
@app.route("/history")
def history():

    records = get_prediction_history()

    return jsonify(records)


# --------------------------------------------------
# Run Application
# --------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)