import streamlit as st
import requests
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="Churn Prediction | Khoshaba Odesho",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Premium Look ---
st.markdown("""
<style>
    /* Main Colors: Navy Blue (#1E3A5F), Gold (#D4AF37) */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A5F;
        text-align: center;
        letter-spacing: -0.5px;
    }
    .sub-title {
        font-size: 1rem;
        text-align: center;
        color: #6B7B8D;
        margin-top: -10px;
        margin-bottom: 2rem;
    }
    .risk-high { color: #E74C3C; font-size: 1.8rem; font-weight: 800; }
    .risk-low { color: #27AE60; font-size: 1.8rem; font-weight: 800; }
    .risk-medium { color: #F39C12; font-size: 1.8rem; font-weight: 800; }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Form Styling */
    .stForm {
        background-color: #F8F9FA;
        border: 1px solid #E1E4E8;
        border-radius: 10px;
        padding: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    /* Recommendation Box */
    .rec-box {
        background: #ffffff;
        border-left: 5px solid #1E3A5F;
        padding: 1.5rem;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar (Your Branding) ---
with st.sidebar:
    st.image("assets/logo.jpg", width=160)
    st.markdown("---")
    st.markdown("### Khoshaba Odesho")
    st.caption("MLOps Engineer | Data Scientist")
    st.markdown("---")
    st.markdown("#### 🌐 Live APIs")
    st.markdown("- [Backend API](https://assyriana-churn-api.hf.space/docs)")
    st.markdown("- [GitHub Repo](https://github.com/Assyrian91/mlops-churn-pipeline)")
    st.markdown("---")
    st.caption("Built with Python • Scikit-learn • FastAPI • Docker • Streamlit")

# --- Main Layout ---
st.markdown('<div class="main-title">🔄 Telecom Customer Churn Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Identify at-risk customers and generate proactive retention strategies</div>', unsafe_allow_html=True)

# API URL
API_URL = "https://assyriana-churn-api.hf.space"

# --- Input Form (Doesn't re-run until button is clicked) ---
with st.form("prediction_form"):
    st.subheader("📋 Customer Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Demographics**")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        
    with col2:
        st.markdown("**Services**")
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        
    with col3:
        st.markdown("**Billing & Support**")
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])

    col4, col5 = st.columns(2)
    with col4:
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=0.05)
    with col5:
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=monthly_charges * tenure * 0.8, step=0.05)

    # Submit Button
    submit_button = st.form_submit_button("🔍 Analyze Churn Risk", type="primary", use_container_width=True)

# --- Prediction Logic ---
if submit_button:
    payload = {
        "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": phone, "MultipleLines": multiple_lines,
        "InternetService": internet, "OnlineSecurity": online_security, "OnlineBackup": online_backup,
        "DeviceProtection": device_protection, "TechSupport": tech_support, "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies, "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": monthly_charges, "TotalCharges": total_charges
    }

    with st.spinner("Analyzing customer data..."):
        try:
            response = requests.post(f"{API_URL}/predict", json=payload)
            result = response.json()
            probability = result["churn_probability"]
            will_churn = result["customer_will_churn"]

            # Determine Risk Class
            if probability >= 0.7:
                risk_class, risk_label, emoji = "risk-high", "HIGH RISK", "🔴"
            elif probability >= 0.4:
                risk_class, risk_label, emoji = "risk-medium", "MEDIUM RISK", "🟡"
            else:
                risk_class, risk_label, emoji = "risk-low", "LOW RISK", "🟢"

            # Display Results in Columns
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                st.markdown("### Prediction Result")
                st.progress(probability)
                st.markdown(f'<p class="{risk_class}">{emoji} {risk_label}</p>', unsafe_allow_html=True)
                st.caption(f"Probability: **{probability * 100:.1f}%**")
                
            with res_col2:
                st.markdown("### Quick Stats")
                m1, m2, m3 = st.columns(3)
                m1.metric("Tenure", f"{tenure} mo", "Short" if tenure < 12 else "Stable")
                m2.metric("Contract", contract.split("-")[0], "Risky" if "month" in contract.lower() else "Safe")
                m3.metric("Bill", f"${monthly_charges:.0f}")

            # Recommendations
            st.subheader("💡 Personalized Retention Strategy")
            if probability >= 0.7:
                st.markdown("""<div class="rec-box"><strong>🚨 URGENT — Immediate Action Required</strong><br><br>
                • Offer <strong>20-30% discount</strong> on current plan<br>
                • Assign <strong>dedicated account manager</strong><br>
                • Offer <strong>free upgrade</strong> to higher-tier plan for 3 months<br>
                • Schedule <strong>retention call within 24 hours</strong></div>""", unsafe_allow_html=True)
            elif probability >= 0.4:
                st.markdown("""<div class="rec-box"><strong>⚠️ MONITOR — Proactive Outreach Recommended</strong><br><br>
                • Send <strong>personalized check-in email</strong><br>
                • Offer <strong>loyalty discount</strong> (10-15%)<br>
                • Suggest <strong>contract upgrade</strong> from month-to-month<br>
                • Add <strong>free value</strong> (extra data, premium channel trial)</div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div class="rec-box"><strong>✅ STABLE — Maintain Relationship</strong><br><br>
                • Continue <strong>standard engagement</strong><br>
                • Include in <strong>loyalty rewards program</strong><br>
                • Send <strong>satisfaction survey</strong> quarterly</div>""", unsafe_allow_html=True)

            # Risk Factors
            with st.expander("🔍 Detected Risk Factors"):
                factors = []
                if tenure < 12: factors.append("🔴 Very short tenure — customer hasn't committed")
                if contract == "Month-to-month": factors.append("🔴 Month-to-month contract — easy to leave")
                if internet == "Fiber optic": factors.append("🟡 Fiber optic users have higher churn rates")
                if payment == "Electronic check": factors.append("🟡 Electronic check payment — correlated with churn")
                if tech_support == "No": factors.append("🟡 No tech support — less engagement")
                if not factors: factors.append("🟢 No major risk factors detected")
                for f in factors: st.markdown(f)

        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to the prediction API. Please try again in a moment.")