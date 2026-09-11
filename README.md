# 🏦 Bank Customer Churn Predictor

An end-to-end Machine Learning web application designed to predict bank customer churn. This project helps financial institutions identify customers at high risk of leaving, enabling proactive retention strategies.

🔗 **Live App:** [View Live Streamlit App](https://bank-churn--predictor-nzd9hkaiffrgnd4jg2bjfk.streamlit.app/)

---

## 🚀 Features
* **🔮 Single Prediction Dashboard:** Input custom customer details (Credit Score, Age, Tenure, Balance, Geography, Gender, etc.) to get an instant churn probability score.
* **📊 Model Performance Metrics:** Transparent breakdown of model evaluation metrics (Accuracy, Precision, Recall, F1 Score).
* **💡 Feature Importance Visualization:** Visual charts highlighting which factors drive customers to leave the bank.

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Web Framework:** Streamlit
* **Data Manipulation & Modeling:** Pandas, NumPy, Scikit-Learn / XGBoost
* **Model Serialization:** Joblib
* **Deployment:** Streamlit Cloud & GitHub

---

## 📂 Project Structure
```text
bank-churn--predictor/
│
├── main.py              # Main Streamlit application file
├── app.py               # Secondary app reference/sync file
├── churn_model.pkl      # Trained machine learning model
├── model_features.pkl   # Saved feature alignment structure
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
