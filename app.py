from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)


# Load trained model
model = joblib.load("model_joblib.pkl")


@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "status": "API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    input_df = pd.DataFrame([{
        "ApplicantIncome": data["income"],
        "Credit_History": data["credit_history"],
        "Self_Employed": data["self_employed"],
        "Property_Area": data["property_area"]
    }])


    # Prediction
    prediction = model.predict(input_df)[0]


    # Probability
    probability = model.predict_proba(input_df)[0][1]


    return jsonify({
        "prediction": "Approve" if prediction == 1 else "Reject",
        "probability": round(float(probability), 3)
    })


if __name__ == "__main__":

    app.run(
        debug=True
    )
    