import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Load model, encoders, and test datasets ---
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)
with open("X_test.pkl", "rb") as f:
    X_test = pickle.load(f)
with open("y_test.pkl", "rb") as f:
    y_test = pickle.load(f)

st.title("Employee Salary Prediction App")

# --- Sidebar Input ---
def user_input_features():
    age = st.sidebar.slider("Age", 18, 65, 30)
    fnlwgt = st.sidebar.number_input("Final Weight (fnlwgt)", 10000, 500000, 50000, step=1000)
    educational_num = st.sidebar.slider("Educational Number", 1, 16, 10)
    workclass = st.sidebar.selectbox("Workclass", encoders['workclass'].classes_)
    marital_status = st.sidebar.selectbox("Marital Status", encoders['marital-status'].classes_)
    occupation = st.sidebar.selectbox("Occupation", encoders['occupation'].classes_)
    relationship = st.sidebar.selectbox("Relationship", encoders['relationship'].classes_)
    race = st.sidebar.selectbox("Race", encoders['race'].classes_)
    gender = st.sidebar.selectbox("Gender", encoders['gender'].classes_)
    native_country = st.sidebar.selectbox("Native Country", encoders['native-country'].classes_)
    capital_gain = st.sidebar.number_input("Capital Gain", 0, 99999, 0, step=1000)
    capital_loss = st.sidebar.number_input("Capital Loss", 0, 4356, 0, step=100)
    hours_per_week = st.sidebar.slider("Hours per Week", 1, 99, 40)

    data = {
        "age": age,
        "workclass": workclass,
        "fnlwgt": fnlwgt,
        "educational-num": educational_num,
        "marital-status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "gender": gender,
        "capital-gain": capital_gain,
        "capital-loss": capital_loss,
        "hours-per-week": hours_per_week,
        "native-country": native_country,
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()
st.subheader("Your Input")
st.write(input_df)

# --- Encode input for prediction ---
encoded_input = input_df.copy()
for col in encoders:
    encoded_input[col] = encoders[col].transform(encoded_input[col].astype(str))

# --- Ensure order and dtypes match test set ---
encoded_input = encoded_input[X_test.columns]
for col in X_test.columns:
    encoded_input[col] = encoded_input[col].astype(X_test[col].dtype)

# --- Row-matching function ---
def is_record_in_test(encoded_row, X_test):
    # Compare DataFrames with matching column order/dtype
    matches = (X_test == encoded_row.values[0]).all(axis=1)
    idxs = np.where(matches)[0]
    return idxs[0] if len(idxs) > 0 else -1

if st.button("Predict Salary"):
    pred = model.predict(encoded_input)[0]
    proba = model.predict_proba(encoded_input)[0]

    idx = is_record_in_test(encoded_input, X_test)
    pred_str = ">50K" if pred == 1 else "<=50K"

    if idx != -1:
        actual_label = y_test.iloc[idx] if hasattr(y_test, 'iloc') else y_test[idx]
        actual_label_str = ">50K" if (actual_label in [1, '>50K']) else "<=50K"
        st.success("✔ This record is found in your test set!")
        st.write(f"**Model Prediction:** {pred_str}  \n**Actual Label:** {actual_label_str}")
        st.write("Prediction Probability:", {">50K": proba[1], "<=50K": proba[0]})
    else:
        st.info("This exact record is NOT in your test set, so only a prediction is shown.")
        st.write(f"**Model Prediction:** {pred_str}")
        st.write("Prediction Probability:", {">50K": proba[1], "<=50K": proba[0]})
