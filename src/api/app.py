import streamlit as st
import requests
import pandas as pd

# Page config
st.set_page_config(
    page_title="Churn Prediction App",
    page_icon="🔄",
    layout="centered"
)

# API URL — uses local Docker when running locally, or Hugging Face in production
API_URL = "http://127.0.0.1:8000"

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        text-align: center;
        color: #6B7B8D;
        margin-bottom: 2rem;
    }
    .risk-high { color: #E74C3C; font-size: 1.5rem; font-weight: 700; }
    .risk-low { color: #27AE60; font-size: 1.5rem; font-weight: 700; }
    .risk-medium { color: #F39C12; font-size: 1.5rem; font-weight: 700; }
    .metric-card {
        background: #F8F9FA;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .recommendation-box {
        background: #EBF5FB;
        border-left: 5px solid #3498DB;
        padding: 1rem;
        border-radius: 5px;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🔄 Customer Churn Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Identify at-risk customers before they leave — reduce churn, increase revenue</div>', unsafe_allow_html=True)

# Business context
with st.expander("💡 Why Churn Prediction Matters"):
    st.markdown("""
    **The Business Problem:**
    - Acquiring a new customer costs **5-7x more** than keeping an existing one
    - Telecom industry average churn rate: **1.9% monthly** (22% annually)
    - A single lost customer = **$1,200 - $2,400/year** in lost revenue
    - For 7,000 customers, even a 1% churn reduction = **$84,000 - $168,000 saved/year**

    **How This Tool Helps:**
    - Flag high-risk customers **before** they cancel
    - Enable proactive retention campaigns (discounts, upgrades, support)
    - Prioritize limited retention budget on highest-risk accounts
    """)

st.divider()

# Input form
st.subheader("📋 Customer Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone = st.selectbox("Phone Service", ["Yes", "No"])

with col2:
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

col3, col4 = st.columns(2)

with col3:
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

with col4:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])

monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=0.05)
total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=monthly_charges * tenure * 0.8, step=0.05)

# Predict button
st.divider()
predict_btn = st.button("🔍 Predict Churn Risk", type="primary", use_container_width=True)

if predict_btn:
    # Build payload
    payload = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    with st.spinner("Analyzing customer data..."):
        try:
            response = requests.post(f"{API_URL}/predict", json=payload)
            result = response.json()

            probability = result["churn_probability"]
            will_churn = result["customer_will_churn"]

            # Risk level
            if probability >= 0.7:
                risk_class = "risk-high"
                risk_label = "HIGH RISK"
                emoji = "🔴"
            elif probability >= 0.4:
                risk_class = "risk-medium"
                risk_label = "MEDIUM RISK"
                emoji = "🟡"
            else:
                risk_class = "risk-low"
                risk_label = "LOW RISK"
                emoji = "🟢"

            # Results section
            st.divider()
            st.subheader("📊 Prediction Result")

            # Probability bar
            st.progress(probability)
            st.markdown(f'<p class="{risk_class}">{emoji} {risk_label} — {probability * 100:.1f}% churn probability</p>', unsafe_allow_html=True)

            # Metric cards
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Tenure", f"{tenure} months", "Short tenure = higher risk" if tenure < 12 else "Stable")
            with m2:
                st.metric("Contract", contract, "High risk" if contract == "Month-to-month" else "Low risk")
            with m3:
                st.metric("Monthly Bill", f"${monthly_charges:.2f}")

            # Recommendations
            st.subheader("💡 Personalized Retention Strategy")

            if probability >= 0.7:
                st.markdown("""
                <div class="recommendation-box">
                <strong>🚨 URGENT — Immediate Action Required</strong><br><br>
                • Offer **20-30% discount** on current plan<br>
                • Assign **dedicated account manager**<br>
                • Offer **free upgrade** to higher-tier plan for 3 months<br>
                • Schedule **retention call within 24 hours**<br>
                • Investigate recent service issues
                </div>
                """, unsafe_allow_html=True)
            elif probability >= 0.4:
                st.markdown("""
                <div class="recommendation-box">
                <strong>⚠️ MONITOR — Proactive Outreach Recommended</strong><br><br>
                • Send **personalized check-in email**<br>
                • Offer **loyalty discount** (10-15%)<br>
                • Suggest **contract upgrade** from month-to-month<br>
                • Add **free value** (extra data, premium channel trial)
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="recommendation-box">
                <strong>✅ STABLE — Maintain Relationship</strong><br><br>
                • Continue **standard engagement**<br>
                • Include in **loyalty rewards program**<br>
                • Monitor for **service quality changes**<br>
                • Send **satisfaction survey** quarterly
                </div>
                """, unsafe_allow_html=True)

            # Key risk factors
            st.subheader("🔍 Key Risk Factors Detected")
            factors = []
            if tenure < 12:
                factors.append("🔴 Very short tenure — customer hasn't committed")
            if contract == "Month-to-month":
                factors.append("🔴 Month-to-month contract — easy to leave")
            if internet == "Fiber optic":
                factors.append("🟡 Fiber optic users have higher churn rates")
            if payment == "Electronic check":
                factors.append("🟡 Electronic check payment — correlated with churn")
            if tech_support == "No":
                factors.append("🟡 No tech support — less engagement")
            if not factors:
                factors.append("🟢 No major risk factors detected")

            for factor in factors:
                st.markdown(factor)

        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API. Make sure Docker container is running:\n\n`docker run -d -p 8000:8000 --name churn-container churn-api:latest`")

# Footer
st.divider()
st.caption("Built with Python • Scikit-learn • FastAPI • Docker • Streamlit | MLOps Pipeline by Assyrian")