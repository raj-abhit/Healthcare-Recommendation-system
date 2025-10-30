# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Blood Donation Predictor",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #ff4b4b;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .donor-positive {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
    }
    .donor-negative {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        border-radius: 5px;
    }
    .recommendation-item {
        background-color: #e7f3ff;
        padding: 0.5rem 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        border-left: 4px solid #007bff;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_features():
    """Load the trained model and feature information"""
    try:
        model = joblib.load('blood_donation_recommendation_model.pkl')
        with open('model_features.json', 'r') as f:
            feature_info = json.load(f)
        return model, feature_info
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'blood_donation_recommendation_model.pkl' and 'model_features.json' are in the same directory.")
        return None, None

def generate_recommendations(prediction, probability, features):
    """Generate personalized health recommendations"""
    recency = features['Recency']
    frequency = features['Frequency']
    monetary = features['Monetary']
    time_since_first = features['Time']
    
    recommendations = []
    
    if prediction == 1:  # Likely donor
        recommendations.append("🎯 **You are likely to be a regular blood donor!** Keep up this life-saving habit.")
        
        if recency <= 3:
            recommendations.append("💪 **Recent donor detected** - Maintain adequate iron levels with iron-rich foods.")
        else:
            recommendations.append("🩸 **Consider donating soon** to maintain your regular donation schedule.")
            
        if frequency > 10:
            recommendations.append("🏆 **Experienced donor** - Regular ferritin level checks are recommended.")
        
        if frequency >= 5:
            recommendations.append("⭐ **Frequent donor** - Ensure proper hydration and nutrition between donations.")
            
    else:  # Non-donor or irregular donor
        recommendations.append("🌟 **Potential donor identified** - Here's how you can start your donation journey:")
        
        if recency > 12:
            recommendations.append("📅 **Long time since last donation** - Consider scheduling a donation appointment.")
        elif recency > 6:
            recommendations.append("🔄 **Getting back on track** - Regular donations help maintain blood supply.")
            
        if frequency == 0:
            recommendations.append("🚀 **First-time donor potential** - Blood donation is safe and saves lives!")
        elif frequency < 3:
            recommendations.append("📈 **Occasional donor** - Consider becoming a regular donor for greater impact.")
    
    # General health recommendations for all
    recommendations.extend([
        "🍎 **Nutrition**: Eat iron-rich foods (spinach, lentils, red meat) and vitamin C for better absorption",
        "💧 **Hydration**: Drink plenty of water before and after donation",
        "😴 **Rest**: Ensure adequate sleep before donating",
        "🏃 **Exercise**: Regular physical activity improves blood circulation",
        "🩺 **Health Check**: Free health screening with every donation",
        "📱 **Stay Connected**: Download blood donation apps for reminders"
    ])
    
    # Specific timing recommendations
    if recency < 3:
        recommendations.append("⏳ **Next donation**: You can donate again in about 2-3 months")
    elif recency < 6:
        recommendations.append("📋 **Eligibility check**: Verify you meet donation criteria")
    
    return recommendations

def calculate_donation_metrics(features):
    """Calculate additional donation metrics"""
    donation_rate = features['Frequency'] / (features['Time'] + 1e-6)
    avg_interval = features['Time'] / (features['Frequency'] + 1e-6)
    
    metrics = {
        'Donation Rate (per month)': donation_rate,
        'Average Interval (months)': avg_interval,
        'Total Blood Donated (liters)': features['Monetary'] / 1000,
        'Lifesaving Potential': f"Up to {features['Frequency'] * 3} lives potentially saved"
    }
    
    return metrics

def main():
    # Header
    st.markdown('<h1 class="main-header">🩸 Personalized Blood Donation Predictor</h1>', unsafe_allow_html=True)
    
    # Sidebar for information
    with st.sidebar:
        st.header("About This App")
        st.markdown("""
        This AI-powered application predicts blood donation likelihood and provides 
        personalized health recommendations using machine learning.
        
        **How it works:**
        - Enter your donation history
        - Get instant prediction
        - Receive personalized recommendations
        - Track your donation impact
        
        **Model Features:**
        - Recency (months since last donation)
        - Frequency (total donations)
        - Monetary (total blood donated)
        - Time (months since first donation)
        """)
        
        st.header("Blood Donation Facts")
        st.markdown("""
        🔸 One donation can save up to 3 lives
        🔸 Healthy adults can donate every 56 days
        🔸 Blood cannot be manufactured
        🔸 Only 3% of age-eligible people donate yearly
        """)
    
    # Load model
    model, feature_info = load_model_and_features()
    if model is None:
        st.stop()
    
    # Main content - Two columns layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📊 Enter Your Donation Information")
        
        with st.form("prediction_form"):
            # Input fields
            recency = st.slider(
                "Months since last donation (Recency)",
                min_value=0,
                max_value=50,
                value=6,
                help="How many months since your last blood donation"
            )
            
            frequency = st.number_input(
                "Total number of donations (Frequency)",
                min_value=0,
                max_value=100,
                value=5,
                help="Total number of times you've donated blood"
            )
            
            monetary = st.number_input(
                "Total blood donated in c.c. (Monetary)",
                min_value=0,
                max_value=50000,
                value=12500,
                step=500,
                help="Total volume of blood donated (1 donation ≈ 450-500 c.c.)"
            )
            
            time_since_first = st.number_input(
                "Months since first donation (Time)",
                min_value=0,
                max_value=200,
                value=24,
                help="Months since your first blood donation"
            )
            
            # Calculate derived features
            donation_rate = frequency / (time_since_first + 1e-6)
            recent_donor = 1 if recency <= 3 else 0
            experienced_donor = 1 if frequency > 5 else 0
            avg_donation_interval = time_since_first / (frequency + 1e-6)
            
            submitted = st.form_submit_button("🔍 Predict Donation Likelihood")
    
    with col2:
        if submitted:
            st.header("🎯 Prediction Results")
            
            # Prepare input data
            input_data = {
                'Recency': recency,
                'Frequency': frequency,
                'Monetary': monetary,
                'Time': time_since_first,
                'Donation_Rate': donation_rate,
                'Recent_Donor': recent_donor,
                'Experienced_Donor': experienced_donor,
                'Avg_Donation_Interval': avg_donation_interval
            }
            
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data], columns=feature_info['feature_names'])
            
            # Make prediction
            try:
                prediction = model.predict(input_df)[0]
                probability = model.predict_proba(input_df)[0][1]
                
                # Display prediction
                st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                
                if prediction == 1:
                    st.markdown(f"""
                    <div class="donor-positive">
                        <h3>✅ Likely Blood Donor</h3>
                        <p><strong>Confidence:</strong> {probability:.2%}</p>
                        <p>You show strong patterns of a regular blood donor!</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="donor-negative">
                        <h3>📋 Potential Irregular Donor</h3>
                        <p><strong>Confidence:</strong> {probability:.2%}</p>
                        <p>There's opportunity to become a more regular donor!</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Display metrics
                st.subheader("📈 Your Donation Metrics")
                metrics = calculate_donation_metrics(input_data)
                
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("Donation Rate", f"{metrics['Donation Rate (per month)']:.2f}/month")
                    st.metric("Total Blood Donated", f"{metrics['Total Blood Donated (liters)']:.1f} L")
                
                with metric_col2:
                    st.metric("Average Interval", f"{metrics['Average Interval (months)']:.1f} months")
                    st.metric("Impact", metrics['Lifesaving Potential'])
                
                # Generate and display recommendations
                st.subheader("💡 Personalized Recommendations")
                recommendations = generate_recommendations(prediction, probability, input_data)
                
                for i, recommendation in enumerate(recommendations, 1):
                    st.markdown(f'<div class="recommendation-item">{recommendation}</div>', unsafe_allow_html=True)
                
                # Next steps
                st.subheader("🔄 Next Steps")
                if prediction == 1:
                    st.success("**Maintain Your Impact:** Continue regular donations and encourage others!")
                else:
                    st.info("**Start Your Journey:** Find a local blood drive and schedule your first donation!")
                    
                # Share button
                if st.button("📢 Share Your Achievement"):
                    st.balloons()
                    st.success("Thank you for being part of the lifesaving community! 🎉")
            
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
        
        else:
            # Show placeholder when no prediction yet
            st.info("👈 Enter your donation information and click 'Predict' to get started!")
            
            # Sample predictions
            st.subheader("💡 Sample Scenarios")
            
            sample_cases = [
                {"label": "Regular Donor", "recency": 2, "frequency": 12, "monetary": 30000, "time": 36},
                {"label": "New Donor", "recency": 1, "frequency": 2, "monetary": 1000, "time": 6},
                {"label": "Lapsed Donor", "recency": 18, "frequency": 8, "monetary": 20000, "time": 60}
            ]
            
            for case in sample_cases:
                with st.expander(f"Example: {case['label']}"):
                    st.write(f"• Last donation: {case['recency']} months ago")
                    st.write(f"• Total donations: {case['frequency']}")
                    st.write(f"• Blood donated: {case['monetary']} c.c.")
                    st.write(f"• Donating for: {case['time']} months")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit | Machine Learning Model: Random Forest</p>
        <p>Remember: Your donation can save lives! 🩸</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()