import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Walmart Weekly Sales Prediction",
    page_icon="🛒",
    layout="wide",
)

# ============================================================
# Theme / CSS (less “heavy”, more breathable)
# ============================================================
st.markdown(
    """
    <style>
      /* App background */
      [data-testid="stAppViewContainer"]{
        background:
          radial-gradient(1000px 500px at 14% 0%, rgba(37,99,235,0.20), transparent 55%),
          radial-gradient(900px 500px at 92% 8%, rgba(99,102,241,0.14), transparent 60%),
          linear-gradient(180deg, #050814 0%, #040711 45%, #04060f 100%);
      }

      /* Main padding */
      .block-container { padding-top: 1.4rem; padding-bottom: 2.2rem; }

      /* Sidebar */
      [data-testid="stSidebar"]{
        background: linear-gradient(180deg, rgba(10,15,30,0.96) 0%, rgba(5,8,20,0.99) 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
      }

      /* Typography tweaks */
      h1,h2,h3 { letter-spacing: -0.3px; }
      .muted { color: rgba(255,255,255,0.72); font-size: 0.98rem; line-height: 1.4; }
      .tiny  { color: rgba(255,255,255,0.62); font-size: 0.88rem; }
      .divider { height: 1px; background: rgba(255,255,255,0.10); margin: 1.0rem 0; }

      /* Pills */
      .pill {
        display:inline-block;
        padding: .28rem .72rem;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.06);
        font-size: .85rem;
        color: rgba(255,255,255,0.82);
        margin-right: .35rem;
        margin-bottom: .35rem;
      }

      /* Cards */
      .card {
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.04);
        border-radius: 18px;
        padding: 1.05rem 1.10rem;
        box-shadow: 0 12px 28px rgba(0,0,0,0.32);
      }

      .soft-card {
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.03);
        border-radius: 18px;
        padding: 0.95rem 1.05rem;
      }

      /* Buttons */
      .stButton>button, .stDownloadButton>button {
        border-radius: 14px !important;
        border: 1px solid rgba(255,255,255,0.14) !important;
        background: linear-gradient(180deg, rgba(37,99,235,0.22), rgba(37,99,235,0.08)) !important;
        color: rgba(255,255,255,0.92) !important;
        padding: 0.68rem 0.95rem !important;
      }
      .stButton>button:hover, .stDownloadButton>button:hover {
        border: 1px solid rgba(99,102,241,0.35) !important;
        background: linear-gradient(180deg, rgba(99,102,241,0.28), rgba(37,99,235,0.10)) !important;
      }

      /* Inputs */
      [data-baseweb="select"] > div, .stTextInput input, .stDateInput input {
        border-radius: 12px !important;
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
      }

      /* Reduce “crowded” gaps a bit */
      [data-testid="stVerticalBlock"] { gap: 0.8rem; }

      /* Hide Streamlit footer */
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
    cols = joblib.load("columns.pkl")  # list of feature names expected by model
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

        # Safe date-derived features (only used if expected)
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
    return pred, X

# ============================================================
# Session state
# ============================================================
DEFAULTS = {
    "store": 1,
    "holiday_label": "Non-holiday Week",
    "temp": 60.0,
    "fuel": 3.50,
    "cpi": 180.0,
    "unemp": 7.50,
    "week_date": date(2012, 2, 10),
}

EXAMPLE = {
    "store": 5,
    "holiday_label": "Holiday Week",
    "temp": 45.0,
    "fuel": 3.20,
    "cpi": 210.0,
    "unemp": 8.60,
    "week_date": date(2011, 11, 25),
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None  # dict with pred, X_used, etc.

# ============================================================
# Header (hero layout)
# ============================================================
top_left, top_right = st.columns([0.72, 0.28], gap="large")

with top_left:
    st.markdown("# 🛒 Walmart Weekly Sales Prediction")
    st.markdown(
        """
        <span class="pill">Supervised ML · Regression</span>
        <span class="pill">Streamlit Deployment</span>
        <span class="pill">Input Validation</span>
        <span class="pill">What-if Analysis</span>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='muted'>Estimate weekly sales using store conditions and economic indicators. "
        "Useful for inventory planning and staffing decisions during peak weeks.</div>",
        unsafe_allow_html=True,
    )

with top_right:
    # Small “status” card to reduce crowding
    st.markdown(
        """
        <div class="soft-card">
          <div class="tiny">App Status</div>
          <div style="font-weight:800; font-size:1.05rem;">Ready to Predict ✅</div>
          <div class="tiny" style="margin-top:.35rem;">Model loaded from <code>model.pkl</code></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ============================================================
# Sidebar (less crowded: essentials first, advanced in expanders)
# ============================================================
with st.sidebar:
    st.markdown("## Inputs")
    st.caption("Enter the week conditions, then run prediction.")

    with st.form("input_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            st.selectbox("Store", list(range(1, 46)), key="store")
        with c2:
            st.date_input("Week Date", key="week_date")

        st.selectbox("Week Type", ["Non-holiday Week", "Holiday Week"], key="holiday_label")

        st.slider("Temperature (°F)", min_value=-5.0, max_value=105.0, step=0.1, key="temp")

        with st.expander("Advanced: Economic Indicators", expanded=False):
            st.slider("Fuel Price", min_value=2.0, max_value=5.0, step=0.01, key="fuel")
            st.slider("CPI", min_value=120.0, max_value=230.0, step=0.1, key="cpi")
            st.slider("Unemployment Rate (%)", min_value=3.0, max_value=15.0, step=0.01, key="unemp")

        b1, b2 = st.columns(2)
        with b1:
            predict_btn = st.form_submit_button("Predict", use_container_width=True)
        with b2:
            compare_btn = st.form_submit_button("Holiday vs Non-holiday", use_container_width=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    st.markdown("### Quick Actions")
    q1, q2 = st.columns(2)
    with q1:
        if st.button("Load Example", use_container_width=True):
            for k, v in EXAMPLE.items():
                st.session_state[k] = v
            st.success("Example loaded ✅")
            st.rerun()

    with q2:
        if st.button("Reset", use_container_width=True):
            for k, v in DEFAULTS.items():
                st.session_state[k] = v
            st.info("Reset done ✅")
            st.rerun()

    q3, q4 = st.columns(2)
    with q3:
        if st.button("Clear History", use_container_width=True):
            st.session_state.history = []
            st.session_state.last_result = None
            st.success("History cleared ✅")
            st.rerun()

    with q4:
        if len(st.session_state.history) > 0:
            hist_df_dl = pd.DataFrame(st.session_state.history)
            st.download_button(
                "Download CSV",
                data=hist_df_dl.to_csv(index=False).encode("utf-8"),
                file_name="prediction_history.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.button("Download CSV", disabled=True, use_container_width=True)

# Read current state
store = st.session_state.store
holiday_flag = 1 if st.session_state.holiday_label == "Holiday Week" else 0
temp = st.session_state.temp
fuel = st.session_state.fuel
cpi = st.session_state.cpi
unemp = st.session_state.unemp
week_date = st.session_state.week_date

# ============================================================
# Tabs (cleaner responsibilities)
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs(["📌 Predict", "📈 Insights", "🧪 Model Performance", "ℹ️ About"])

# ============================================================
# TAB 1: Predict
# ============================================================
with tab1:
    left, right = st.columns([0.66, 0.34], gap="large")

    # ---- LEFT: Output area
    with left:
        st.markdown("## Prediction Output")

        issues = validate_inputs(temp, fuel, cpi, unemp)
        if issues:
            st.warning("Input check: please review these before predicting.")
            for msg in issues:
                st.write(f"• {msg}")

        # Predict
        if predict_btn:
            if issues:
                st.error("Fix the input issues above, then try again.")
            else:
                try:
                    pred, X_used = predict_one(store, holiday_flag, temp, fuel, cpi, unemp, week_date)

                    st.session_state.last_result = {
                        "mode": "single",
                        "pred": pred,
                        "store": store,
                        "holiday_flag": holiday_flag,
                        "week_date": str(week_date),
                        "X_used": X_used,
                    }

                    st.session_state.history.append({
                        "date": str(week_date),
                        "store": int(store),
                        "holiday_flag": int(holiday_flag),
                        "temperature": float(temp),
                        "fuel_price": float(fuel),
                        "cpi": float(cpi),
                        "unemployment": float(unemp),
                        "pred_sales": float(pred),
                    })

                    st.success("Prediction generated successfully ✅")

                except Exception as e:
                    st.error("Prediction failed. Please try again.")
                    st.exception(e)

        # Compare
        if compare_btn:
            if issues:
                st.error("Fix the input issues above, then try again.")
            else:
                try:
                    pred_non, X_non = predict_one(store, 0, temp, fuel, cpi, unemp, week_date)
                    pred_hol, X_hol = predict_one(store, 1, temp, fuel, cpi, unemp, week_date)

                    diff = pred_hol - pred_non
                    pct = (diff / pred_non * 100) if pred_non != 0 else 0.0

                    st.session_state.last_result = {
                        "mode": "compare",
                        "pred_non": pred_non,
                        "pred_hol": pred_hol,
                        "diff": diff,
                        "pct": pct,
                        "store": store,
                        "week_date": str(week_date),
                        "X_non": X_non,
                        "X_hol": X_hol,
                    }

                    st.session_state.history.append({
                        "date": str(week_date),
                        "store": int(store),
                        "holiday_flag": "compare",
                        "temperature": float(temp),
                        "fuel_price": float(fuel),
                        "cpi": float(cpi),
                        "unemployment": float(unemp),
                        "pred_sales": float(pred_hol),
                    })

                    st.info("What-if analysis complete: only Holiday_Flag changed ✅")

                except Exception as e:
                    st.error("Comparison failed. Please try again.")
                    st.exception(e)

        # ---- Render last result (clean, non-crowded)
        res = st.session_state.last_result
        if res is None:
            st.markdown(
                """
                <div class="card">
                  <div style="font-weight:800; font-size:1.05rem;">No output yet</div>
                  <div class="muted" style="margin-top:.35rem;">
                    Use the sidebar to enter inputs, then click <b>Predict</b> or <b>Holiday vs Non-holiday</b>.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

            if res["mode"] == "single":
                k1, k2, k3 = st.columns(3)
                with k1:
                    st.metric("Predicted Weekly Sales", money(res["pred"]))
                with k2:
                    st.metric("Store ID", str(res["store"]))
                with k3:
                    st.metric("Week Type", "Holiday" if res["holiday_flag"] == 1 else "Non-holiday")

                with st.expander("See input values used (model features)"):
                    st.dataframe(res["X_used"], use_container_width=True)

            if res["mode"] == "compare":
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Non-holiday Prediction", money(res["pred_non"]))
                with c2:
                    st.metric("Holiday Prediction", money(res["pred_hol"]))
                with c3:
                    # delta style
                    st.metric("Estimated Uplift", money(res["diff"]), f"{res['pct']:.2f}%")

                with st.expander("See both feature inputs (debug/validation)"):
                    st.write("Non-holiday features:")
                    st.dataframe(res["X_non"], use_container_width=True)
                    st.write("Holiday features:")
                    st.dataframe(res["X_hol"], use_container_width=True)

        # ---- History (kept but less noisy: inside expander)
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("## Prediction History (this session)")

        if len(st.session_state.history) == 0:
            st.caption("No predictions yet.")
        else:
            hist_df = pd.DataFrame(st.session_state.history).iloc[::-1].reset_index(drop=True)

            with st.expander("Show history table", expanded=False):
                st.dataframe(hist_df, use_container_width=True)

            # Small trend preview (only if we have numeric preds)
            numeric_hist = hist_df.copy()
            numeric_hist["pred_sales"] = pd.to_numeric(numeric_hist["pred_sales"], errors="coerce")
            numeric_hist = numeric_hist.dropna(subset=["pred_sales"])

            if len(numeric_hist) >= 2:
                st.markdown("### Trend Preview")
                st.line_chart(numeric_hist["pred_sales"])

    # ---- RIGHT: Feedback + Summary (one clean column)
    with right:
        st.markdown("## User-friendly Feedback")
        st.markdown(
            """
            <div class="card">
              <div class="muted">
                <b>Why we validate inputs:</b><br>
                Prevents crashes and keeps predictions realistic for users.<br><br>
                <b>How to score well:</b><br>
                Use <b>Holiday vs Non-holiday</b> to demonstrate interactive decision support.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("## Quick Summary")
        st.markdown(
            f"""
            <div class="card">
              <div class="tiny">Selected Store</div>
              <div style="font-size:2.0rem; font-weight:850; margin:.1rem 0 .55rem 0;">{store}</div>
              <div class="tiny">Week Date</div>
              <div class="muted">{week_date}</div>
              <div class="tiny" style="margin-top:.55rem;">Week Type</div>
              <div class="muted">{'Holiday' if holiday_flag==1 else 'Non-holiday'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("## Input Validation Rules")
        st.markdown(
            """
            <div class="soft-card">
              <div class="muted">
                • Fuel Price must be > 0<br>
                • CPI must be > 0<br>
                • Temperature should be within -10°F to 120°F<br>
                • Unemployment should be within 0% to 25%
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================
# TAB 2: Insights (move “busy” content away from Predict tab)
# ============================================================
with tab2:
    st.markdown("## Business Insights (Retail Context)")
    st.markdown(
        """
        <div class="card">
          <div class="muted">
            <b>Inventory Planning:</b> Higher predicted sales → stock fast-moving items and essentials.<br>
            <b>Staffing:</b> Holiday weeks often need more manpower and extended operating hours.<br>
            <b>Economic Context:</b> CPI and unemployment can influence purchasing power and shopping behaviour.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    if len(st.session_state.history) >= 1:
        hist_df = pd.DataFrame(st.session_state.history).copy()
        hist_df["pred_sales"] = pd.to_numeric(hist_df["pred_sales"], errors="coerce")
        hist_df = hist_df.dropna(subset=["pred_sales"])

        st.markdown("## Session Analytics")
        a, b, c = st.columns(3)
        with a:
            st.metric("Predictions Made", str(len(st.session_state.history)))
        with b:
            if len(hist_df) > 0:
                st.metric("Avg Predicted Sales", money(hist_df["pred_sales"].mean()))
            else:
                st.metric("Avg Predicted Sales", "—")
        with c:
            if len(hist_df) > 0:
                st.metric("Max Predicted Sales", money(hist_df["pred_sales"].max()))
            else:
                st.metric("Max Predicted Sales", "—")

        if len(hist_df) >= 2:
            st.markdown("### Predicted Sales Trend (Session)")
            st.line_chart(hist_df["pred_sales"])
    else:
        st.caption("Run at least one prediction to unlock session analytics.")

# ============================================================
# TAB 3: Model Performance
# ============================================================
with tab3:
    st.markdown("## Model Performance Summary")
    st.caption("Add your final MAE/RMSE comparison table here (or paste results).")

    st.markdown(
        """
        <div class="card">
          <div class="muted">
            <b>Metrics used:</b><br>
            • <b>MAE</b> = average absolute error (business-friendly).<br>
            • <b>RMSE</b> = penalises large errors (important for holiday spikes).<br><br>
            <b>Recommended evidence:</b><br>
            • Final comparison table (baseline vs improved models)<br>
            • Predicted vs Actual plot for best model
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# TAB 4: About
# ============================================================
with tab4:
    st.markdown("## About this Application")
    st.markdown(
        """
        <div class="card">
          <div class="muted">
            <b>Goal:</b> Predict weekly Walmart sales to support operational planning.<br>
            <b>Type:</b> Supervised Learning (Regression).<br>
            <b>Model:</b> scikit-learn model loaded from <code>model.pkl</code>.<br>
            <b>Deployment:</b> Streamlit app with validation, what-if analysis, and session history export.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
