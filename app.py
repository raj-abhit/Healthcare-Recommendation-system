import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# -------------------------------------------------------
# Load model + feature order safely
# -------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")

    # Ensure feature_order.json exists and is valid
    if not os.path.exists("feature_order.json"):
        st.error("❌ feature_order.json not found. Please regenerate it in your notebook.")
        st.stop()

    with open("feature_order.json", "r") as f:
        try:
            feature_order = json.load(f)
        except json.JSONDecodeError:
            st.error("❌ feature_order.json is invalid. Please re-save it as a valid JSON list of feature names.")
            st.stop()

    # Handle case where feature_order is wrong (e.g., only [0])
    if not isinstance(feature_order, list) or len(feature_order) < 2:
        st.error("⚠️ feature_order.json seems incorrect. Regenerate it from your notebook using:")
        st.code(
            """import json
FEATURES = list(X.columns)
with open("feature_order.json", "w") as f:
    json.dump(FEATURES, f)
print("✅ feature_order.json saved successfully!")"""
        )
        st.stop()

    return model, feature_order


model, feature_order = load_artifacts()

# -------------------------------------------------------
# App title and intro
# -------------------------------------------------------
st.set_page_config(page_title="Personalized Healthcare Recommendations", layout="centered")

st.title("🧬 Personalized Blood Health Prediction")
st.write(
    "This interactive app uses a trained Machine Learning model to predict whether a person is **Healthy** or may have a **Health Risk**, based on blood test parameters."
)
st.markdown("---")

# -------------------------------------------------------
# Collect input features
# -------------------------------------------------------
st.subheader("🩸 Input Blood Parameters")
st.write("Adjust the feature values below:")

user_data = {}
for feature in feature_order:
    user_data[feature] = st.number_input(f"{feature}", value=0.0, step=0.1)

# Convert to DataFrame
input_df = pd.DataFrame([user_data])[feature_order]

st.markdown("---")

# -------------------------------------------------------
# Predict button
# -------------------------------------------------------
if st.button("🔍 Predict Health Condition"):
    with st.spinner("Analyzing your blood parameters... ⏳"):
        prediction = model.predict(input_df)[0]
        st.success("✅ Model and feature order loaded successfully!")

    # Display prediction results
    st.markdown("## 🧠 Predicted Health Status:")
    if prediction == 1:
        st.success("🩺 The model predicts a **Health Risk**. Please consult a doctor for further evaluation.")
    else:
        st.info("💚 The model predicts that the person is **Healthy**.")

    # Interpretation note
    st.markdown(
        """
        ---
        ### ℹ️ Interpretation Notes:
        - Prediction **1** → Higher risk, needs medical attention  
        - Prediction **0** → Generally healthy  
        - Always consult healthcare professionals for accurate diagnosis and treatment.
        """
    )

# -------------------------------------------------------
# Footer
# -------------------------------------------------------
st.markdown("---")
st.caption("Built with ❤️ by Abhit Raj | Powered by Streamlit & Scikit-learn")
