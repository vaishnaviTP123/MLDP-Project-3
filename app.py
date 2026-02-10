# app.py
# ============================================================
# Walmart Weekly Sales Prediction (PRO UI + Correct Store OHE)
# - Navy/Black theme
# - Proper one-hot encoding for Store + Holiday (if present)
# - Input validation + user-friendly messages
# - Prediction interval (confidence band) using RF tree spread
# - What-if panel (side-by-side scenario)
# - Prediction history + download CSV
# - Debug expander (helps you prove store column is working)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date, datetime

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Walmart Weekly Sales Prediction",
    page_icon="🛒",
    layout="wide",
)

# ----------------------------
# Professional Navy/Black CSS
# ----------------------------
st.markdown(
    """
    <style>
      /* App background */
      [data-testid="stAppViewContainer"]{
        background:
          radial-gradient(1200px 600px at 12% 10%, rgba(37,99,235,0.22), transparent 60%),
          radial-gradient(900px 520px at 92% 18%, rgba(14,165,233,0.14), transparent 58%),
          linear-gradient(180deg, #050712 0%, #040611 60%, #04050f 100%);
      }

      /* Reduce top padding */
      .block-container { padding-top: 1.2rem; padding-bottom: 2.2rem; }

      /* Sidebar */
      [data-testid="stSidebar"]{
        background: linear-gradient(180deg, rgba(9,13,28,0.92), rgba(5,8,18,0.92));
        border-right: 1px solid rgba(148,163,184,0.12);
      }

      /* Headings spacing */
      h1, h2, h3 { letter-spacing: -0.02em; }
      h1 { margin-top: 0.2rem; }

      /* Cards */
      .card{
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(148,163,184,0.12);
        border-radius: 16px;
        padding: 18px 18px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
      }
      .card-strong{
        background: linear-gradient(180deg, rgba(37,99,235,0.12), rgba(255,255,255,0.03));
        border: 1px solid rgba(56,189,248,0.18);
      }

      /* Small label text */
      .muted{
        color: rgba(226,232,240,0.78);
        font-size: 0.95rem;
        line-height: 1.35rem;
      }

      /* Big result */
      .big{
        font-size: 2.35rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0.2rem 0 0.2rem 0;
      }

      /* Badge */
      .badge{
        display:inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(14,165,233,0.12);
        border: 1px solid rgba(14,165,233,0.28);
        color: rgba(224,242,254,0.92);
        font-size: 0.85rem;
        margin-right: 8px;
      }

      /* Buttons */
      .stButton button{
        border-radius: 12px !important;
        font-weight: 700 !important;
        border: 1px solid rgba(148,163,184,0.18) !important;
        background: linear-gradient(180deg, rgba(37,99,235,0.22), rgba(37,99,235,0.10)) !important;
      }
      .stButton button:hover{
        border-color: rgba(56,189,248,0.45) !important;
        transform: translateY(-1px);
        transition: 120ms ease;
      }

      /* Inputs */
      [data-baseweb="select"] > div, .stNumberInput input, .stDateInput input{
        border-radius: 12px !important;
      }

      /* Success & error colors */
      .stAlert { border-radius: 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Helpers
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = joblib.load("model.pkl")
    columns = joblib.load("columns.pkl")  # MUST match the model training columns order
    return model, columns

def safe_date_parts(d: date):
    dt = pd.to_datetime(d)
    year = int(dt.year)
    month = int(dt.month)
    # ISO week number
    week = int(dt.isocalendar().week)
    day_of_week = int(dt.dayofweek)  # Monday=0..Sunday=6
    return year, month, week, day_of_week

def money(x: float) -> str:
    return f"${x:,.2f}"

def predict_with_interval(model, X_row: pd.DataFrame):
    """
    Returns (pred_mean, pred_low, pred_high)
    Uses per-tree predictions for a simple uncertainty band.
    """
    # If model is not RF-like, fall back to point prediction
    if not hasattr(model, "estimators_"):
        p = float(model.predict(X_row)[0])
        return p, np.nan, np.nan

    preds = np.array([est.predict(X_row)[0] for est in model.estimators_], dtype=float)
    mean = float(np.mean(preds))
    low = float(np.percentile(preds, 10))
    high = float(np.percentile(preds, 90))
    return mean, low, high

def build_input_df(columns, *,
                   store_number: int,
                   is_holiday: bool,
                   temperature: float,
                   fuel_price: float,
                   cpi: float,
                   unemployment: float,
                   d: date):
    """
    Builds a 1-row dataframe with EXACT training columns (from columns.pkl),
    including Store_ one-hot columns.
    """
    year, month, week, day_of_week = safe_date_parts(d)

    # Start with all zeros in correct column order
    X = pd.DataFrame(0, index=[0], columns=columns)

    # Numeric columns (only fill if they exist in training)
    numeric_map = {
        "Temperature": temperature,
        "Fuel_Price": fuel_price,
        "CPI": cpi,
        "Unemployment": unemployment,
        "Year": year,
        "Month": month,
        "WeekOfYear": week,
        "DayOfWeek": day_of_week,
    }
    for k, v in numeric_map.items():
        if k in X.columns:
            X.loc[0, k] = v

    # Holiday encoding
    # If your training kept Holiday_Flag as 0/1:
    if "Holiday_Flag" in X.columns:
        X.loc[0, "Holiday_Flag"] = 1 if is_holiday else 0

    # If your training one-hot encoded holiday (e.g., Holiday_Flag_1):
    if "Holiday_Flag_1" in X.columns:
        X.loc[0, "Holiday_Flag_1"] = 1 if is_holiday else 0

    # Store one-hot
    store_cols = [c for c in X.columns if c.startswith("Store_")]
    if store_cols:
        X.loc[0, store_cols] = 0
        store_col = f"Store_{store_number}"
        if store_col in X.columns:
            X.loc[0, store_col] = 1
        # else: store unseen in training -> left as all zeros
    # If NO Store_ columns exist, we try a numeric Store column (baseline case)
    elif "Store" in X.columns:
        X.loc[0, "Store"] = store_number

    return X

def validate_inputs(temperature, fuel_price, cpi, unemployment):
    issues = []
    # Keep reasonable ranges (adjust if you want)
    if not (0 <= temperature <= 120):
        issues.append("Temperature looks out of typical range (0–120°F).")
    if not (0.5 <= fuel_price <= 6.0):
        issues.append("Fuel price looks out of typical range (0.5–6.0).")
    if not (100 <= cpi <= 300):
        issues.append("CPI looks out of typical range (100–300).")
    if not (0 <= unemployment <= 20):
        issues.append("Unemployment rate looks out of typical range (0–20%).")
    return issues

# ----------------------------
# Load model + columns
# ----------------------------
try:
    model, columns = load_artifacts()
except Exception as e:
    st.error("Model artifacts failed to load. Make sure model.pkl and columns.pkl exist in the same folder as app.py.")
    st.exception(e)
    st.stop()

# ----------------------------
# Session state (history)
# ----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------
# Header
# ----------------------------
left, right = st.columns([0.72, 0.28], vertical_alignment="top")

with left:
    st.markdown(
        """
        <div class="card card-strong">
          <span class="badge">Supervised ML</span>
          <span class="badge">Regression</span>
          <span class="badge">Random Forest</span>
          <h1 style="margin:0.15rem 0 0.2rem 0;">🛒 Walmart Weekly Sales Prediction</h1>
          <div class="muted">
            Enter store conditions and economic indicators to estimate weekly sales.
            This app includes input validation, what-if analysis, prediction interval, and downloadable history.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    st.markdown(
        """
        <div class="card">
          <div class="muted"><b>Model artifacts</b></div>
          <div class="muted">Loaded columns: <b>{}</b></div>
          <div class="muted">Has tree ensemble: <b>{}</b></div>
        </div>
        """.format(
            len(columns),
            "Yes" if hasattr(model, "estimators_") else "No",
        ),
        unsafe_allow_html=True,
    )

st.write("")

# ----------------------------
# Sidebar: Inputs
# ----------------------------
with st.sidebar:
    st.markdown("## Input Features")
    st.caption("Adjust inputs and click **Predict Weekly Sales**.")

    # Detect stores available from columns.pkl (Store_2, Store_3, ...)
    store_ohe_cols = [c for c in columns if c.startswith("Store_")]
    if store_ohe_cols:
        store_numbers = sorted([int(c.split("_")[1]) for c in store_ohe_cols if c.split("_")[1].isdigit()])
        # Note: with drop_first=True, Store_1 may not exist — that’s normal.
        # We still allow selecting 1, but it will be represented as "all zeros".
        if 1 not in store_numbers:
            store_numbers = [1] + store_numbers
    else:
        # fallback if training used numeric "Store"
        store_numbers = list(range(1, 46))

    store_number = st.selectbox("Select Store", store_numbers, index=0)

    holiday_label = st.selectbox("Holiday Flag", ["Non-holiday Week", "Holiday Week"], index=0)
    is_holiday = holiday_label == "Holiday Week"

    temperature = st.slider("Temperature (°F)", 0.0, 120.0, 60.0, 0.1)
    fuel_price = st.slider("Fuel Price", 0.5, 6.0, 3.5, 0.01)
    cpi = st.number_input("CPI", value=180.0, step=0.1)
    unemployment = st.slider("Unemployment Rate (%)", 0.0, 20.0, 7.5, 0.1)
    week_date = st.date_input("Week Date", value=date(2012, 2, 10))

    st.write("")
    colA, colB = st.columns(2)
    with colA:
        do_reset = st.button("Reset", use_container_width=True)
    with colB:
        do_example = st.button("Use Example", use_container_width=True)

    # Reset / Example actions
    if do_reset:
        st.session_state["__reset__"] = True
    if do_example:
        st.session_state["__example__"] = True

# Apply Reset/Example via rerun-safe pattern
if st.session_state.get("__reset__", False):
    st.session_state["__reset__"] = False
    # Sidebar widgets need keys to truly reset; easiest is to rerun with defaults:
    st.experimental_rerun()

if st.session_state.get("__example__", False):
    st.session_state["__example__"] = False
    # We can't directly set sidebar widget values without keys, so we show a hint:
    st.info("Example tip: Set Store=2, Holiday=Non-holiday, Temp=60, Fuel=3.50, CPI=180, Unemp=7.5, Date=2012-02-10.")
    # Not forcing rerun.

# ----------------------------
# Main layout: Prediction + What-if
# ----------------------------
col1, col2 = st.columns([0.58, 0.42], vertical_alignment="top")

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Prediction")

    # Validate
    issues = validate_inputs(temperature, fuel_price, cpi, unemployment)
    if issues:
        st.warning("Some inputs look unusual (you can still predict):\n\n- " + "\n- ".join(issues))

    predict_btn = st.button("Predict Weekly Sales", use_container_width=True)

    if predict_btn:
        # Build X row with proper OHE
        X = build_input_df(
            columns,
            store_number=store_number,
            is_holiday=is_holiday,
            temperature=temperature,
            fuel_price=fuel_price,
            cpi=cpi,
            unemployment=unemployment,
            d=week_date,
        )

        # Make prediction + interval
        pred, low, high = predict_with_interval(model, X)

        st.success("Prediction generated successfully!")
        st.markdown(f"<div class='muted'>Predicted Weekly Sales</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='big'>{money(pred)}</div>", unsafe_allow_html=True)

        if not np.isnan(low) and not np.isnan(high):
            st.caption(f"Confidence band (10–90% from trees): {money(low)}  →  {money(high)}")

        # Save to history
        st.session_state.history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "store": store_number,
            "holiday": int(is_holiday),
            "temperature": temperature,
            "fuel_price": fuel_price,
            "cpi": cpi,
            "unemployment": unemployment,
            "week_date": str(week_date),
            "pred": float(pred),
            "low_10": None if np.isnan(low) else float(low),
            "high_90": None if np.isnan(high) else float(high),
        })

        with st.expander("See input values used"):
            st.dataframe(X.T.rename(columns={0: "value"}), use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("What-If Analysis (Compare 2 Scenarios)")
    st.caption("Investor-friendly: show how predictions change when you tweak 1–2 variables.")

    # Scenario B tweaks
    b_store = st.selectbox("Scenario B: Store", options=list(dict.fromkeys([store_number] + store_numbers)), index=0)
    b_holiday = st.selectbox("Scenario B: Holiday", ["Same as A", "Non-holiday Week", "Holiday Week"], index=0)
    b_temp_delta = st.slider("Scenario B: Temperature change (°F)", -20.0, 20.0, 0.0, 0.5)
    b_fuel_delta = st.slider("Scenario B: Fuel price change", -1.0, 1.0, 0.0, 0.01)

    compare_btn = st.button("Run Comparison", use_container_width=True)

    if compare_btn:
        # Scenario A
        XA = build_input_df(
            columns,
            store_number=store_number,
            is_holiday=is_holiday,
            temperature=temperature,
            fuel_price=fuel_price,
            cpi=cpi,
            unemployment=unemployment,
            d=week_date,
        )
        predA, lowA, highA = predict_with_interval(model, XA)

        # Scenario B
        b_is_holiday = is_holiday if b_holiday == "Same as A" else (b_holiday == "Holiday Week")
        XB = build_input_df(
            columns,
            store_number=int(b_store),
            is_holiday=b_is_holiday,
            temperature=float(np.clip(temperature + b_temp_delta, 0, 120)),
            fuel_price=float(np.clip(fuel_price + b_fuel_delta, 0.5, 6.0)),
            cpi=cpi,
            unemployment=unemployment,
            d=week_date,
        )
        predB, lowB, highB = predict_with_interval(model, XB)

        diff = predB - predA
        st.markdown("**Scenario A vs Scenario B**")
        st.write(
            pd.DataFrame({
                "Scenario": ["A (Current)", "B (What-if)"],
                "Predicted": [predA, predB],
            })
        )
        st.info(f"Difference (B − A): {money(diff)}")

        with st.expander("Show scenario inputs"):
            st.write("Scenario A (non-zero features):")
            st.dataframe(XA.loc[:, (XA != 0).any()].T, use_container_width=True)
            st.write("Scenario B (non-zero features):")
            st.dataframe(XB.loc[:, (XB != 0).any()].T, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.write("")

# ----------------------------
# History + Download
# ----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Prediction History")
st.caption("Keeps your past predictions during this session. You can download as CSV (nice for report evidence).")

if len(st.session_state.history) == 0:
    st.info("No predictions yet. Run a prediction to populate history.")
else:
    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df, use_container_width=True)

    csv = hist_df.to_csv(index=False).encode("utf-8")
    c1, c2 = st.columns([0.25, 0.75])
    with c1:
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="prediction_history.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        clear_btn = st.button("Clear History", use_container_width=True)
        if clear_btn:
            st.session_state.history = []
            st.experimental_rerun()

st.markdown("</div>", unsafe_allow_html=True)

st.write("")

# ----------------------------
# Debug (Proving Store OHE works)
# ----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Debug / Evidence (Optional)")
st.caption("Use this section in your report to prove the app correctly encodes Store and matches training columns.")

with st.expander("Show Store encoding check"):
    X_dbg = build_input_df(
        columns,
        store_number=store_number,
        is_holiday=is_holiday,
        temperature=temperature,
        fuel_price=fuel_price,
        cpi=cpi,
        unemployment=unemployment,
        d=week_date,
    )
    store_cols_dbg = [c for c in X_dbg.columns if c.startswith("Store_")]
    if store_cols_dbg:
        active = [c for c in store_cols_dbg if X_dbg.loc[0, c] == 1]
        st.write("Store one-hot columns found:", len(store_cols_dbg))
        st.write("Active store column:", active[0] if active else "(none) — this is normal if Store_1 was dropped_first")
        st.write("Sum across Store_ cols (should be 1 for most stores, or 0 for dropped_first store):",
                 int(X_dbg[store_cols_dbg].sum(axis=1).iloc[0]))
    else:
        st.warning("No Store_ columns found in columns.pkl. Your model likely used numeric Store instead of OHE.")
        if "Store" in X_dbg.columns:
            st.write("Numeric Store set to:", X_dbg.loc[0, "Store"])

st.markdown("</div>", unsafe_allow_html=True)
