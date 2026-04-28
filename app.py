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
# CONSTANTS
# =========================================================

MODEL_FILE = "xgboost_ptb_pipeline.pkl"
LOGO_FILE = "analytics_ai_logo.png"

REQUIRED_FEATURES = [
    "Age",
    "Gender",
    "Annual Income",
    "Income Bracket",
    "Marital Status",
    "Employment Status",
    "Region",
    "Urban/Rural Flag",
    "State",
    "ZIP Code",
    "Plan Preference Type",
    "Web Form Completion Rate",
    "Quote Requested",
    "Application Started",
    "Behavior Score",
    "Application Submitted",
    "Application Applied"
]

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), MODEL_FILE)
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {MODEL_FILE}")
        st.stop()
    return joblib.load(model_path)

model = load_model()

# =========================================================
# LOGO
# =========================================================

logo_path = os.path.join(os.path.dirname(__file__), LOGO_FILE)

if os.path.exists(logo_path):
    logo = Image.open(logo_path)
    st.image(logo, width=300)

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
    if tier == "Platinum":
        return "Immediate sales outreach by senior sales rep"
    elif tier == "Gold":
        return "High-priority nurture campaign with personalized offer"
    elif tier == "Silver":
        return "Automated nurture sequence and educational content"
    else:
        return "Low-touch awareness campaign"


def sales_priority(tier):
    if tier == "Platinum":
        return "Very High"
    elif tier == "Gold":
        return "High"
    elif tier == "Silver":
        return "Medium"
    else:
        return "Low"


def generate_sample_data(n=500):
    np.random.seed(42)

    states = ["CA", "MN", "TX", "FL", "NY", "IL", "AZ", "GA"]
    genders = ["Male", "Female", "Other"]
    income_brackets = ["<50K", "50K-75K", "75K-100K", "100K+"]
    marital_statuses = ["Single", "Married", "Divorced"]
    employment_statuses = ["Employed", "Self-Employed", "Retired", "Unemployed"]
    regions = ["West", "Midwest", "South", "Northeast"]
    urban_rural = ["Urban", "Suburban", "Rural"]
    plan_preferences = ["Basic", "Standard", "Premium"]

    df = pd.DataFrame({
        "Customer ID": [str(uuid.uuid4())[:8] for _ in range(n)],
        "Age": np.random.randint(22, 75, n),
        "Gender": np.random.choice(genders, n),
        "Annual Income": np.random.randint(30000, 180000, n),
        "Income Bracket": np.random.choice(income_brackets, n),
        "Marital Status": np.random.choice(marital_statuses, n),
        "Employment Status": np.random.choice(employment_statuses, n),
        "Region": np.random.choice(regions, n),
        "Urban/Rural Flag": np.random.choice(urban_rural, n),
        "State": np.random.choice(states, n),
        "ZIP Code": np.random.randint(10000, 99999, n).astype(str),
        "Plan Preference Type": np.random.choice(plan_preferences, n),
        "Web Form Completion Rate": np.round(np.random.uniform(0.1, 1.0, n), 2),
        "Quote Requested": np.random.choice([0, 1], n),
        "Application Started": np.random.choice([0, 1], n),
        "Behavior Score": np.random.randint(1, 100, n),
        "Application Submitted": np.random.choice([0, 1], n),
        "Application Applied": np.random.choice([0, 1], n),
        "Policy Purchased": np.random.choice([0, 1], n, p=[0.75, 0.25]),
        "Purchase Channel": np.random.choice(["Web", "Call Center", "Broker", "Email"], n)
    })

    df["Age Group"] = pd.cut(
        df["Age"],
        bins=[0, 30, 45, 60, 100],
        labels=["18-30", "31-45", "46-60", "60+"]
    ).astype(str)

    return df


def read_uploaded_file(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)
    else:
        st.error("Please upload a CSV or Excel file.")
        st.stop()


def validate_input_columns(df):
    missing = [col for col in REQUIRED_FEATURES if col not in df.columns]
    return missing


def score_leads(df):
    input_df = df[REQUIRED_FEATURES].copy()

    proba = model.predict_proba(input_df)[:, 1]
    df["PTB_Score"] = proba * 100
    df["Lead_Tier"] = df["PTB_Score"].apply(assign_tier)
    df["Sales_Priority"] = df["Lead_Tier"].apply(sales_priority)
    df["Next_Best_Action"] = df["Lead_Tier"].apply(next_best_action)

    return df


def safe_sum(df, col):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0).sum()
    return 0


def bool_count(df, col):
    if col not in df.columns:
        return 0
    return df[col].isin([1, "1", "Yes", "Y", True, "True"]).sum()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Demo Controls")

mode = st.sidebar.radio(
    "Select data mode",
    ["Demo Data", "Upload CSV / Excel"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model")
st.sidebar.success("XGBoost PTB Pipeline Loaded")

# =========================================================
# LOAD DATA
# =========================================================

if mode == "Demo Data":
    df = generate_sample_data()
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload lead data",
        type=["csv", "xlsx"]
    )

    if uploaded_file is None:
        st.info("Upload a CSV or Excel file to begin.")
        st.stop()

    df = read_uploaded_file(uploaded_file)

# =========================================================
# MAIN APP
# =========================================================

st.title("Sales Lead Nurture Model Dashboard")
st.markdown(
    """
    This dashboard demonstrates a healthcare sales lead scoring model that predicts 
    propensity-to-buy, assigns lead tiers, and recommends next-best actions for sales 
    and marketing teams.
    """
)

tabs = st.tabs([
    "🤖 Score Leads",
    "📊 Executive KPIs",
    "📈 Lead Insights",
    "📤 Export"
])

# =========================================================
# TAB 1: SCORE LEADS
# =========================================================

with tabs[0]:
    st.subheader("Input Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

    missing_cols = validate_input_columns(df)

    if missing_cols:
        st.error("Missing required columns:")
        st.write(missing_cols)
        st.stop()

    scored_df = score_leads(df.copy())

    st.subheader("Scored Leads")

    display_df = scored_df.copy()
    display_df["PTB_Score"] = display_df["PTB_Score"].round(2)

    st.dataframe(display_df, use_container_width=True)

    st.markdown("### Lead Tier Explanation")

    st.markdown(
        """
        - **Platinum:** Highest propensity-to-buy; immediate sales outreach  
        - **Gold:** Strong opportunity; personalized nurture campaign  
        - **Silver:** Moderate opportunity; automated nurture  
        - **Bronze:** Low current propensity; low-touch engagement  
        """
    )

# =========================================================
# TAB 2: KPIs
# =========================================================

with tabs[1]:
    scored_df = score_leads(df.copy())

    total_leads = len(scored_df)
    avg_score = scored_df["PTB_Score"].mean()
    platinum_count = (scored_df["Lead_Tier"] == "Platinum").sum()
    gold_count = (scored_df["Lead_Tier"] == "Gold").sum()
    silver_count = (scored_df["Lead_Tier"] == "Silver").sum()
    bronze_count = (scored_df["Lead_Tier"] == "Bronze").sum()

    purchased = safe_sum(scored_df, "Policy Purchased")
    conversion_rate = (purchased / total_leads) * 100 if total_leads > 0 else 0

    quote_requested = bool_count(scored_df, "Quote Requested")
    quote_rate = (quote_requested / total_leads) * 100 if total_leads > 0 else 0

    app_started = bool_count(scored_df, "Application Started")
    app_started_rate = (app_started / total_leads) * 100 if total_leads > 0 else 0

    app_submitted = bool_count(scored_df, "Application Submitted")
    app_submitted_rate = (app_submitted / total_leads) * 100 if total_leads > 0 else 0

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Leads", f"{total_leads:,}")
    c2.metric("Average PTB Score", f"{avg_score:.2f}%")
    c3.metric("Platinum Leads", f"{platinum_count:,}")
    c4.metric("Gold Leads", f"{gold_count:,}")

    c5, c6, c7, c8 = st.columns(4)

    c5.metric("Quote Requested Rate", f"{quote_rate:.2f}%")
    c6.metric("Application Started Rate", f"{app_started_rate:.2f}%")
    c7.metric("Application Submitted Rate", f"{app_submitted_rate:.2f}%")
    c8.metric("Observed Conversion Rate", f"{conversion_rate:.2f}%")

    st.markdown("---")
    st.subheader("Lead Tier Distribution")

    tier_df = (
        scored_df["Lead_Tier"]
        .value_counts()
        .reset_index()
    )
    tier_df.columns = ["Lead Tier", "Count"]

    fig = px.bar(
        tier_df,
        x="Lead Tier",
        y="Count",
        title="Lead Tier Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB 3: CHARTS
# =========================================================

with tabs[2]:
    scored_df = score_leads(df.copy())

    st.subheader("Lead Tier by State")

    if "State" in scored_df.columns:
        states = sorted(scored_df["State"].dropna().unique().tolist())
        selected_states = st.multiselect(
            "Filter by State",
            states,
            default=states[:5] if len(states) > 5 else states
        )

        filtered_df = (
            scored_df[scored_df["State"].isin(selected_states)]
            if selected_states else scored_df
        )

        fig_state = px.histogram(
            filtered_df,
            x="State",
            color="Lead_Tier",
            barmode="group",
            title="Lead Tier by State"
        )

        st.plotly_chart(fig_state, use_container_width=True)

    st.subheader("Lead Tier by Income Bracket")

    if "Income Bracket" in scored_df.columns:
        fig_income = px.histogram(
            scored_df,
            x="Income Bracket",
            color="Lead_Tier",
            barmode="group",
            title="Lead Tier by Income Bracket"
        )

        st.plotly_chart(fig_income, use_container_width=True)

    st.subheader("Lead Tier by Employment Status")

    if "Employment Status" in scored_df.columns:
        fig_emp = px.histogram(
            scored_df,
            x="Employment Status",
            color="Lead_Tier",
            barmode="group",
            title="Lead Tier by Employment Status"
        )

        st.plotly_chart(fig_emp, use_container_width=True)

    st.subheader("PTB Score Distribution")

    fig_score = px.histogram(
        scored_df,
        x="PTB_Score",
        nbins=30,
        title="Distribution of Propensity-to-Buy Scores"
    )

    st.plotly_chart(fig_score, use_container_width=True)

    st.subheader("Next Best Action Summary")

    nba_df = (
        scored_df["Next_Best_Action"]
        .value_counts()
        .reset_index()
    )
    nba_df.columns = ["Next Best Action", "Count"]

    fig_nba = px.bar(
        nba_df,
        x="Next Best Action",
        y="Count",
        title="Recommended Sales Actions"
    )

    st.plotly_chart(fig_nba, use_container_width=True)

# =========================================================
# TAB 4: EXPORT
# =========================================================

with tabs[3]:
    scored_df = score_leads(df.copy())
    scored_df["PTB_Score"] = scored_df["PTB_Score"].round(2)

    st.subheader("Download Scored Leads")

    csv = scored_df.to_csv(index=False)

    st.download_button(
        label="Download Scored Leads CSV",
        data=csv,
        file_name="scored_leads.csv",
        mime="text/csv"
    )

    st.markdown("### Recommended Demo Talk Track")

    st.markdown(
        """
        This prototype demonstrates how an AI/ML model can help sales and marketing teams 
        prioritize outreach based on propensity-to-buy. 
        
        Instead of treating every lead equally, the model segments leads into tiers, 
        recommends the next-best action, and creates a decision-ready dashboard for 
        sales leadership.
        """
    )