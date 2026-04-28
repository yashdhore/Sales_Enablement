import os
import uuid
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.express as px
from PIL import Image

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Sales Lead Nurture Model Dashboard",
    layout="wide"
)

# =========================================================
# CUSTOM STYLING (TABS + UI)
# =========================================================

st.markdown("""
<style>

/* Tabs spacing */
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
}

/* Default tabs */
.stTabs [data-baseweb="tab"] {
    height: 52px;
    padding: 12px 22px;
    border-radius: 10px 10px 0px 0px;
    background: linear-gradient(135deg, #e6f0ff, #f5f7fa);
    color: #1a1a1a;
    font-weight: 600;
    font-size: 16px;
    border: 1px solid #d0d7e2;
}

/* Active tab */
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #00c6ff, #0072ff);
    color: white !important;
    border: none;
}

/* Hover */
.stTabs [data-baseweb="tab"]:hover {
    background: #cfe3ff;
}

/* KPI cards */
.metric-card {
    background: linear-gradient(135deg, #ffffff, #f5f7fa);
    border-radius: 12px;
    padding: 15px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# CONSTANTS
# =========================================================

MODEL_FILE = "xgboost_ptb_pipeline.pkl"
LOGO_FILE = "analytics_ai_logo.png"

REQUIRED_FEATURES = [
    "Age", "Gender", "Annual Income", "Income Bracket", "Marital Status",
    "Employment Status", "Region", "Urban/Rural Flag", "State", "ZIP Code",
    "Plan Preference Type", "Web Form Completion Rate", "Quote Requested",
    "Application Started", "Behavior Score", "Application Submitted",
    "Application Applied"
]

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), MODEL_FILE)
    if not os.path.exists(model_path):
        st.error("Model file not found!")
        st.stop()
    return joblib.load(model_path)

model = load_model()

# =========================================================
# LOGO
# =========================================================

logo_path = os.path.join(os.path.dirname(__file__), LOGO_FILE)
if os.path.exists(logo_path):
    st.image(Image.open(logo_path), width=300)

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def assign_tier(score):
    if score >= 90:
        return "Platinum"
    elif score >= 75:
        return "Gold"
    elif score >= 50:
        return "Silver"
    else:
        return "Bronze"

def next_best_action(tier):
    return {
        "Platinum": "Immediate sales outreach",
        "Gold": "Personalized campaign",
        "Silver": "Automated nurture",
        "Bronze": "Low-touch awareness"
    }[tier]

def sales_priority(tier):
    return {
        "Platinum": "Very High",
        "Gold": "High",
        "Silver": "Medium",
        "Bronze": "Low"
    }[tier]

# =========================================================
# SAMPLE DATA
# =========================================================

def generate_sample_data(n=300):
    np.random.seed(42)
    df = pd.DataFrame({
        "Age": np.random.randint(22, 75, n),
        "Gender": np.random.choice(["Male", "Female"], n),
        "Annual Income": np.random.randint(30000, 150000, n),
        "Income Bracket": np.random.choice(["Low", "Medium", "High"], n),
        "Marital Status": np.random.choice(["Single", "Married"], n),
        "Employment Status": np.random.choice(["Employed", "Self-Employed"], n),
        "Region": np.random.choice(["West", "Midwest", "South"], n),
        "Urban/Rural Flag": np.random.choice(["Urban", "Rural"], n),
        "State": np.random.choice(["CA", "TX", "NY"], n),
        "ZIP Code": np.random.randint(10000, 99999, n).astype(str),
        "Plan Preference Type": np.random.choice(["Basic", "Premium"], n),
        "Web Form Completion Rate": np.random.rand(n),
        "Quote Requested": np.random.choice([0,1], n),
        "Application Started": np.random.choice([0,1], n),
        "Behavior Score": np.random.randint(1,100,n),
        "Application Submitted": np.random.choice([0,1], n),
        "Application Applied": np.random.choice([0,1], n),
        "Policy Purchased": np.random.choice([0,1], n)
    })
    return df

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ Controls")
mode = st.sidebar.radio("Select Mode", ["Demo Data", "Upload File"])

if mode == "Demo Data":
    df = generate_sample_data()
else:
    file = st.sidebar.file_uploader("Upload CSV/Excel", ["csv","xlsx"])
    if file:
        df = pd.read_csv(file) if file.name.endswith("csv") else pd.read_excel(file)
    else:
        st.stop()

# =========================================================
# MAIN APP
# =========================================================

st.title("Sales Lead Nurture Model Dashboard")

tabs = st.tabs([
    "Score Leads",
    "Executive KPIs",
    "Insights",
    "Export"
])

# =========================================================
# TAB 1
# =========================================================

with tabs[0]:
    st.dataframe(df.head())

    if any(col not in df.columns for col in REQUIRED_FEATURES):
        st.error("Missing required columns")
        st.stop()

    scored = df.copy()
    scored["PTB_Score"] = model.predict_proba(df[REQUIRED_FEATURES])[:,1]*100
    scored["Lead_Tier"] = scored["PTB_Score"].apply(assign_tier)
    scored["Next_Action"] = scored["Lead_Tier"].apply(next_best_action)

    st.dataframe(scored)

# =========================================================
# TAB 2
# =========================================================

with tabs[1]:
    total = len(scored)
    avg = scored["PTB_Score"].mean()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Leads", total)
    c2.metric("Avg Score", f"{avg:.2f}")
    c3.metric("Platinum", (scored["Lead_Tier"]=="Platinum").sum())
    c4.metric("Gold", (scored["Lead_Tier"]=="Gold").sum())

# =========================================================
# TAB 3
# =========================================================

with tabs[2]:
    fig = px.histogram(scored, x="State", color="Lead_Tier")
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB 4
# =========================================================

with tabs[3]:
    st.download_button("Download CSV", scored.to_csv(index=False))
