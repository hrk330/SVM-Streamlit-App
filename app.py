import streamlit as st  # Streamlit for web app
import pandas as pd  # Pandas for data handling
import joblib  # Joblib for loading model and scaler
from PIL import Image  # For adding images or logos

# Load the trained SVM model and scaler
model = joblib.load("svm_model.pkl")  # Trained SVM model
scaler = joblib.load("scaler.pkl")    # Saved scaler used for preprocessing

# Set custom page configuration
st.set_page_config(
    page_title="SVM Model Predictor",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add a header with a logo
st.markdown(
    """
    <div style="text-align: center;">
        <img src="https://images.pexels.com/photos/373543/pexels-photo-373543.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1" alt="App Image" style="width:100%; height:25vh; object-fit:cover;">
    </div>
    """,
    unsafe_allow_html=True
)


st.title("Welcome to the SVM Model Prediction App")
st.markdown("### **Discover insights with your trained machine learning model** 🚀")
st.markdown("This web application uses a trained Support Vector Machine model to predict user behavior based on provided inputs.")

# Add sections with expander
with st.expander("🧾 About the App"):
    st.write("""
    This app is built using **Streamlit** and allows users to:
    - Input their details (Gender, Age, and Estimated Salary).
    - Get predictions on whether a user is likely to purchase a product.
    """)

# Add user input fields in the sidebar
st.sidebar.header("🔍 Enter User Details:")
gender = st.sidebar.radio("Select Gender:", ["Male", "Female"], index=0)
age = st.sidebar.slider("Select Age:", min_value=18, max_value=60, value=30, step=1)
estimated_salary = st.sidebar.number_input("Enter Estimated Salary (in $):", min_value=15000, max_value=150000, value=50000, step=1000)

# Display user inputs
st.markdown("### **Input Summary**")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Gender", value=gender)
with col2:
    st.metric(label="Age", value=age)
with col3:
    st.metric(label="Estimated Salary", value=f"${estimated_salary}")

# Predict button
if st.button("✨ Predict Now"):
    # Convert Gender to numerical
    gender_encoded = 1 if gender == "Male" else 0

    # Combine inputs into a DataFrame
    input_data = pd.DataFrame({
        "Gender": [gender_encoded],
        "Age": [age],
        "EstimatedSalary": [estimated_salary]
    })

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Make predictions
    prediction = model.predict(input_data_scaled)

    # Display the result
    st.markdown("### **Prediction Result**")
    if prediction[0] == 1:
        st.success("🎉 The user is likely to **purchase** the product!")
    else:
        st.warning("⚠️ The user is **unlikely** to purchase the product.")

# Footer with additional styling
st.markdown(
    """
    <hr style="border: 1px solid #f1f1f1;">
    <p style="text-align: center;">
        Built using <a href="https://streamlit.io/" target="_blank">Streamlit</a>.
    </p>
    """,
    unsafe_allow_html=True
)
