import streamlit as st 
import pandas as pd 
import joblib 
import os
#Page configuration
st.set_page_config(page_title="Customer Churn Predictor", page_icon="🏦", layout="centered")
#Load model and features dynamically using absolute paths
@st.cache_resource 
def load_assets(): 
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
    model_path = os.path.join(BASE_DIR, 'churn_model.pkl') 
    features_path = os.path.join(BASE_DIR, 'model_features.pkl')
    model = joblib.load(model_path)
    features = joblib.load(features_path)
    return model, features
model, features = load_assets()
st.subheader("Customer Profile") 
col1, col2 = st.columns(2)
with col1: 
    age = st.slider("Age", 18, 90, 40) 
    geography = st.selectbox("Country", ["France", "Germany", "Spain"]) 
    gender = st.selectbox("Gender", ["Male", "Female"]) 
    credit_score = st.number_input("Credit Score", 300, 850, 650) 
    tenure = st.slider("Tenure (Years)", 0, 10, 5)
with col2: 
    balance = st.number_input("Account Balance ($)", 0.0, 250000.0, 50000.0) 
    num_products = st.selectbox("Number of Products", [1, 2, 3, 4]) 
    has_card = st.radio("Has Credit Card?", ["Yes", "No"]) 
    is_active = st.radio("Is Active Member?", ["Yes", "No"]) 
    salary = st.number_input("Estimated Salary ($)", 0.0, 200000.0, 75000.0)
#Process inputs into dynamic feature DataFrame
input_data = {
     'CreditScore': credit_score, 
     'Age': age, 
     'Tenure': tenure, 
     'Balance': balance, 
     'NumOfProducts': num_products, 
     'HasCrCard': 1 if has_card == "Yes" else 0, 
     'IsActiveMember': 1 if is_active == "Yes" else 0, 
     'EstimatedSalary': salary, 
     'Geography_Germany': 1 if geography == "Germany" else 0, 
     'Geography_Spain': 1 if geography == "Spain" else 0, 
     'Gender_Male': 1 if gender == "Male" else 0 }
#Create base DataFrame
input_df = pd.DataFrame([input_data])
#Automatically fill missing columns with 0 and align column order exactly
for col in features: 
    if col not in input_df.columns: 
        input_df[col] = 0
input_df = input_df[features]
st.divider()
if st.button("Predict Churn Risk", type="primary"): 
    # Probabilistic prediction using tuned threshold (0.35) 
    prob = model.predict_proba(input_df)[0][1] 
    prediction = 1 if prob >= 0.35 else 0
    st.subheader("Prediction Result")
    if prediction == 1: 
        st.error(f"⚠️ High Churn Risk! Probability: {prob*100:.1f}%") 
        st.write("Recommendation: Assign retention offer or engagement outreach immediately.")
    else:
         st.success(f"✅ Low Churn Risk. Probability: {prob*100:.1f}%") 
         st.write("Recommendation: Account is stable. Regular engagement applies.")    