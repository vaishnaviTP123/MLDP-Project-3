import streamlit as st
import pandas as pd
import joblib
from datetime import date

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Walmart Weekly Sales Predictor",
    page_icon="🛒",
    layout="centered",
)

# ============================================================
# Clean UI CSS (simple, spacious, form card)
# ============================================================
st.markdown(
    """
    <style>
      /* Background */
      [data-testid="stAppViewContainer"]{
        background: linear-gradient(180deg, #070A16 0%, #050611 100%);
      }

      /* Keep content centered + reduce clutter */
      .block-container{
        max-width: 820px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
      }

      /* Titles */
      .title{
        font-size: 2.0rem;
        font-weight: 900;
        letter-spacing: -0.6px;
        margin-bottom: 0.25rem;
      }
      .subtitle{
        color: rgba(255,255,255,0.70);
        font-size: 1.0rem;
        margin-bottom: 1.6rem;
      }

      /* Card */
      .card{
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.04);
        border-radius: 18px;
        padding: 1.35rem 1.35rem;
        box-shadow: 0 14px 34px rgba(0,0,0,0.35);
      }

      /* Spacing between sections */
      .section{
        margin-top: 1.1rem;
      }
      .section-label{
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.2px;
      }
      .hint{
        color: rgba(255,255,255,0.65);
        font-size: 0.92rem;
        margin-top: 0.2rem;
      }

      /* Inputs */
      [data-baseweb="select"] > div,
      .stNumberInput input,
      .stDateInput input{
        border-radius: 12px !important;
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.14) !important;
      }

      /* Primary button */
      .stButton>button{
        width: 100%;
        border-radius: 14px !important;
        border: 1px solid rgba(255,255,255,0.14) !important;
        background: linear-gradient(180deg, rgba(34,197,94,0.22), rgba(34,197,94,0.10)) !important;
        color: rgba(255,255,255,0.94) !important;
        padding: 0.8rem 1rem !important;
        font-weight: 800 !important;
        letter-spacing: 0.2px;
      }
      .stButton>button:hover{
        border: 1px solid rgba(34,197,94,0.40) !important;
        background: linear-gradient(180deg, rgba(34,197,94,0.28), rgba(34,197,94,0.12)) !important;
      }

      /* Result box */
      .result{
        margin-top: 1.1rem;
        border: 1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.05);
        border-radius: 16px;
        padding: 1.05rem 1.15rem;
      }
      .result-label{
        color: rgba(255,255,255,0.70);
        font-size: 0.95rem;
        margin-bottom: 0.25rem;
      }
      .result-value{
        font-size: 2.0rem;
        font-weight: 900;
        letter-spacing: -0.6px;
      }

      footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Load model + feature columns
# ============================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    cols = joblib.load("columns.pkl")
    return model, cols

model, feature_cols = load_artifacts()

# ============================================================
# Helpers
# ============================================================
def money(x: float) -> str:
    return "${:,.2f}".format(float(x))

def validate_inputs(temp, fuel, cpi, unemp):
    issues = []
    if fuel <= 0:
        issues.append("Fuel Price must be > 0.")
    if cpi <= 0:
        issues.append("CPI must be > 0.")
    if not (-10 <= temp <= 120):
        issues.append("Temperature looks unrealistic. Keep within -10°F to 120°F.")
    if not (0 <= unemp <= 25):
        issues.append("Unemployment rate looks unrealistic. Keep within 0% to 25%.")
    return issues

def build_features(store, holiday_flag, temp, fuel, cpi, unemp, week_date, feature_cols):
    dt = pd.to_datetime(week_date)
    base = {
        "Store": int(store),
        "Holiday_Flag": int(holiday_flag),
        "Temperature": float(temp),
        "Fuel_Price": float(fuel),
        "CPI": float(cpi),
        "Unemployment": float(unemp),
        "Year": int(dt.year),
        "Month": int(dt.month),
        "WeekOfYear": int(dt.isocalendar().week),
        "DayOfWeek": int(dt.dayofweek),
    }
    X = pd.DataFrame([base]).reindex(columns=feature_cols, fill_value=0)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    return X

def predict_one(store, holiday_flag, temp, fuel, cpi, unemp, week_date):
    X = build_features(store, holiday_flag, temp, fuel, cpi, unemp, week_date, feature_cols)
    pred = float(model.predict(X)[0])
    return pred

# ============================================================
# Header
# ============================================================
st.markdown('<div class="title">Walmart Weekly Sales Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter the week details and click Predict to get the forecasted weekly sales.</div>', unsafe_allow_html=True)

# ============================================================
# Main clean form (everything together)
# ============================================================
st.markdown('<div class="card">', unsafe_allow_html=True)

with st.form("predict_form", clear_on_submit=False):

    st.markdown('<div class="section-label">Store & Week</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        store = st.selectbox("Store", list(range(1, 46)), index=0)
    with c2:
        week_date = st.date_input("Week Date", value=date(2012, 2, 10))

    c3, c4 = st.columns(2, gap="large")
    with c3:
        week_type = st.selectbox("Week Type", ["Non-holiday Week", "Holiday Week"])
    with c4:
        holiday_flag = 1 if week_type == "Holiday Week" else 0

    st.markdown('<div class="section section-label">Economic Indicators</div>', unsafe_allow_html=True)
    c5, c6 = st.columns(2, gap="large")
    with c5:
        temp = st.number_input("Temperature (°F)", value=60.0, step=0.1)
        fuel = st.number_input("Fuel Price", value=3.50, step=0.01, format="%.2f")
    with c6:
        cpi = st.number_input("CPI", value=180.0, step=0.1)
        unemp = st.number_input("Unemployment (%)", value=7.50, step=0.01, format="%.2f")

    st.markdown('<div class="section"></div>', unsafe_allow_html=True)
    submit = st.form_submit_button("Predict Weekly Sales")

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# Output
# ============================================================
if submit:
    issues = validate_inputs(temp, fuel, cpi, unemp)
    if issues:
        st.error("Please fix these inputs before predicting:")
        for msg in issues:
            st.write(f"• {msg}")
    else:
        try:
            pred = predict_one(store, holiday_flag, temp, fuel, cpi, unemp, week_date)

            st.markdown(
                f"""
                <div class="result">
                  <div class="result-label">Predicted Weekly Sales</div>
                  <div class="result-value">{money(pred)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception as e:
            st.error("Prediction failed. Check that model.pkl and columns.pkl are in the same folder as app.py.")
            st.exception(e)
