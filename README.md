# Sales Lead Nurture Model

## AI-Powered Lead Scoring Dashboard

This project demonstrates how AI/ML can be applied to healthcare and insurance business problems such as lead prioritization, customer engagement, and sales optimization.

The approach is transferable to enterprise use cases including customer journey analytics, conversion optimization, and decision support.

---

## Business Problem

Sales and marketing teams often treat all leads equally, which can lead to:

- Inefficient outreach
- Missed high-value opportunities
- Low conversion rates
- Poor resource allocation

This solution uses machine learning to score leads, segment them into tiers, and recommend next-best actions.

---

## Solution Overview

The application provides an end-to-end lead scoring and decision-support workflow.

### 1. Propensity-to-Buy Scoring

The model predicts each lead’s likelihood to convert using an XGBoost classification pipeline.

### 2. Lead Tiering

| Tier | PTB Score Range | Recommended Strategy |
|---|---:|---|
| Platinum | 90%+ | Immediate sales outreach |
| Gold | 75%–89% | High-priority nurture |
| Silver | 50%–74% | Automated nurture |
| Bronze | < 50% | Low-touch awareness campaign |

### 3. Next-Best Action

Each scored lead receives:

- PTB score
- Lead tier
- Sales priority
- Recommended next-best action

---

## Streamlit Dashboard Features

The dashboard includes:

- Lead scoring from demo data or uploaded data
- Executive KPI summary
- Lead tier distribution
- Lead insights by state, income bracket, and employment status
- PTB score distribution
- Next-best-action summary
- Exportable scored lead file

---

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Plotly
- Pillow
- Joblib

---

## Project Structure

```text
app.py
requirements.txt
README.md
xgboost_ptb_pipeline.pkl
analytics_ai_logo.png
