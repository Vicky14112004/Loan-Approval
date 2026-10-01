import streamlit as st
import requests


st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="💰",
    layout="centered"
)


st.title("💰 Loan Approval Prediction")

st.write(
    "Enter the applicant details to predict loan approval."
)


# -----------------------------------
# Applicant Income
# -----------------------------------

income = st.number_input(
    "Applicant Income",
    min_value=0,
    value=5000
)


# -----------------------------------
# Credit History
# -----------------------------------

credit_history = st.selectbox(
    "Credit History",
    [1.0, 0.0]
)


# -----------------------------------
# Self Employed
# -----------------------------------

self_employed = st.selectbox(
    "Self Employed",
    ["Yes", "No"]
)


# -----------------------------------
# Property Area
# -----------------------------------

property_area = st.selectbox(
    "Property Area",
    [
        "Urban",
        "Semiurban",
        "Rural"
    ]
)


# -----------------------------------
# Prediction
# -----------------------------------

if st.button("🔍 Predict Loan Approval"):

    data = {

        "income": income,

        "credit_history": credit_history,

        "self_employed": self_employed,

        "property_area": property_area

    }


    try:

        response = requests.post(
            "http://127.0.0.1:5000/predict",
            json=data
        )


        result = response.json()


        st.subheader("Prediction Result")


        if result["prediction"] == "Approve":

            st.success("✅ Loan Approved")

        else:

            st.error("❌ Loan Rejected")


        probability = result["probability"]


        st.write(
            f"Approval Probability: "
            f"**{probability * 100:.1f}%**"
        )


        st.progress(probability)


    except Exception as e:

        st.error(
            "Unable to connect to Flask API. "
            "Make sure the Flask server is running."
        )