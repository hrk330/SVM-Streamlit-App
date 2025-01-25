import streamlit as st # type: ignore
import pandas as pd # type: ignore
import joblib # type: ignore

# Load the trained SVM model and scaler
model = joblib.load("svm_model.pkl")  # Trained SVM model
scaler = joblib.load("scaler.pkl")    # Saved scaler used for preprocessing

# Define the Streamlit application
st.title("SVM Model Deployment")
st.write("This is a web-based application to make predictions using the trained Support Vector Machine model.")

# Collect user input
st.header("Input Features")
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", min_value=18, max_value=60, value=30, step=1)
estimated_salary = st.number_input("Estimated Salary", min_value=15000, max_value=150000, value=50000, step=1000)

# Preprocess the input
if st.button("Predict"):
    # Convert Gender to numerical
    gender_encoded = 1 if gender == "Male" else 0

    # Combine inputs into a DataFrame
    input_data = pd.DataFrame({
        "Gender": [gender_encoded],
        "Age": [age],
        "EstimatedSalary": [estimated_salary]
    })

    # Scale the input data using the preloaded scaler
    input_data_scaled = scaler.transform(input_data)

    # Make predictions
    prediction = model.predict(input_data_scaled)

    # Display the result
    if prediction[0] == 1:
        st.success("The prediction is: **Purchased**")
    else:
        st.warning("The prediction is: **Not Purchased**")
