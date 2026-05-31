from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")


def predict_ticket(text):

    text_vector = vectorizer.transform([text])

    prediction = model.predict(text_vector)

    return prediction[0]


@app.route("/")
def home():

    return "Customer Support Ticket Classifier API Running"


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    ticket_text = data.get("text", "")

    prediction = predict_ticket(ticket_text)

    return jsonify({
        "ticket": ticket_text,
        "prediction": prediction
    })


if __name__ == "__main__":
    app.run(debug=True)