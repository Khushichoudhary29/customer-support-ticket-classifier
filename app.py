from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load Model
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")


# --------------------------------------------------
# Prediction Function
# --------------------------------------------------
def predict_ticket(text):

    text_vector = vectorizer.transform([text])

    prediction = model.predict(text_vector)[0]

    confidence = None
    top_predictions = []

    # If model supports probability prediction
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
# API Endpoint
# --------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    ticket_text = data.get("text", "")

    if ticket_text.strip() == "":
        return jsonify({
            "error": "Ticket text cannot be empty"
        }), 400

    prediction, confidence, top_predictions = predict_ticket(ticket_text)

    return jsonify({
        "ticket": ticket_text,
        "prediction": prediction,
        "confidence": confidence,
        "top_predictions": top_predictions
    })


# --------------------------------------------------
# Run App
# --------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)