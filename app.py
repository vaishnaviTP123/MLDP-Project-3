import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Walmart Sales Dashboard",
    page_icon="📈",
    layout="wide",
)

# ============================================================
# Dashboard CSS (navbar + sidebar menu + clean cards)
# ============================================================
st.markdown(
    """
    <style>
      /* ====== Background ====== */
      [data-testid="stAppViewContainer"]{
        background:
          radial-gradient(1000px 520px at 12% 0%, rgba(37,99,235,0.18), transparent 58%),
          radial-gradient(900px 520px at 92% 10%, rgba(99,102,241,0.12), transparent 60%),
          linear-gradient(180deg, #060916 0%, #050812 55%, #050710 100%);
      }

      /* Reduce clutter spacing */
      .block-container { padding-top: 0.85rem; padding-bottom: 2rem; max-width: 1400px; }
      [data-testid="stVerticalBlock"] { gap: 0.85rem; }

      /* ====== Sidebar ====== */
      [data-testid="stSidebar"]{
        background: linear-gradient(180deg, rgba(13,18,36,0.98) 0%, rgba(8,12,26,0.99) 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
      }

      /* Sidebar title */
      .sb-title{
        font-weight: 850;
        font-size: 1.15rem;
        letter-spacing: -0.2px;
        margin: 0.2rem 0 0.1rem 0;
      }
      .sb-sub{ color: rgba(255,255,255,0.62); font-size: 0.88rem; margin-bottom: 0.9rem; }

      /* ====== NAVBAR (fake top bar) ====== */
      .topbar{
        border: 1px solid rgba(255,255,255,0.10);
        background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
        border-radius: 18px;
        padding: 0.9rem 1.1rem;
        box-shadow: 0 12px 28px rgba(0,0,0,0.30);
      }
      .brand{
        font-weight: 900;
        font-size: 1.05rem;
        letter-spacing: -0.2px;
      }
      .brand span{ color: rgba(34,197,94,0.95); }
      .navhint{ color: rgba(255,255,255,0.62); font-size: 0.9rem; }

      /* ====== Cards ====== */
      .card{
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.04);
        border-radius: 18px;
        padding: 1.05rem 1.1rem;
        box-shadow: 0 12px 28px rgba(0,0,0,0.30);
      }
      .card-soft{
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.03);
        border-radius: 18px;
        padding: 0.95rem 1.05rem;
      }
      .kpi-title{ color: rgba(255,255,255,0.65); font-size: 0.86rem; }
      .kpi-value{ font-size: 1.65rem; font-weight: 900; margin-top: 0.15rem; }
      .kpi-sub{ color: rgba(255,255,255,0.62); font-size: 0.88rem; margin-top: 0.15rem; }

      .section-title{ font-weight: 900; letter-spacing: -0.3px; margin-bottom: 0.2rem; }
      .muted{ color: rgba(255,255,255,0.70); }

      /* ====== Inputs style ====== */
      [data-baseweb="select"] > div, .stNumberInput input, .stDateInput input {
        border-radius: 12px !important;
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
      }

      /* ====== Make sidebar radio look like a menu (not ugly buttons) ====== */
      div[role="radiogroup"] label{
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 0.65rem 0.75rem;
        border-radius: 14px;
        margin-bottom: 0.45rem;
        transition: all 0.15s ease;
      }
      div[role="radiogroup"] label:hover{
        background: rgba(99,102,241,0.10);
        border: 1px solid rgba(99,102,241,0.28);
        transform: translateY(-1px);
      }

      /* Primary buttons */
      .stButton>button, .stDownloadButton>button {
        border-radius: 14px !important;
        border: 1px solid rgba(255,255,255,0.14) !important;
        background: linear-gradient(180deg, rgba(34,197,94,0.18), rgba(34,197,94,0.08)) !important;
        color: rgba(255,255,255,0.92) !important;
        padding: 0.68rem 0.95rem !important;
      }
      .stButton>button:hover, .stDownloadButton>button:hover {
        border: 1px solid rgba(34,197,94,0.35) !important;
        background: linear-gradient(180deg, rgba(34,197,94,0.24), rgba(34,197,94,0.09)) !important;
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
    return pred, X

# ============================================================
# Session defaults
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
    st.session_state.last_result = None

# ============================================================
# Sidebar layout (menu + compact inputs)
# ============================================================
with st.sidebar:
    st.markdown('<div class="sb-title">📊 Walmart Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-sub">Prediction + analytics view</div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠 Overview", "🧮 Predict", "📈 Insights", "🧪 Model", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Only show inputs on Predict page (so sidebar is not crowded always)
    if page == "🧮 Predict":
        st.markdown("### Inputs")

        st.selectbox("Store", list(range(1, 46)), key="store")
        st.date_input("Week Date", key="week_date")
        st.selectbox("Week Type", ["Non-holiday Week", "Holiday Week"], key="holiday_label")
        st.slider("Temperature (°F)", -5.0, 105.0, step=0.1, key="temp")

        with st.expander("Advanced Inputs", expanded=False):
            st.slider("Fuel Price", 2.0, 5.0, step=0.01, key="fuel")
            st.slider("CPI", 120.0, 230.0, step=0.1, key="cpi")
            st.slider("Unemployment (%)", 3.0, 15.0, step=0.01, key="unemp")

        st.markdown("### Actions")
        c1, c2 = st.columns(2)
        with c1:
            predict_btn = st.button("Predict", use_container_width=True)
        with c2:
            compare_btn = st.button("Compare", use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            if st.button("Example", use_container_width=True):
                for k, v in EXAMPLE.items():
                    st.session_state[k] = v
                st.success("Example loaded ✅")
                st.rerun()

        with c4:
            if st.button("Reset", use_container_width=True):
                for k, v in DEFAULTS.items():
                    st.session_state[k] = v
                st.info("Reset done ✅")
                st.rerun()

        st.markdown("---")
        if len(st.session_state.history) > 0:
            hist_df_dl = pd.DataFrame(st.session_state.history)
            st.download_button(
                "Download history (CSV)",
                data=hist_df_dl.to_csv(index=False).encode("utf-8"),
                file_name="prediction_history.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.button("Download history (CSV)", disabled=True, use_container_width=True)

    else:
        predict_btn = False
        compare_btn = False

# Read inputs
store = st.session_state.store
holiday_flag = 1 if st.session_state.holiday_label == "Holiday Week" else 0
temp = st.session_state.temp
fuel = st.session_state.fuel
cpi = st.session_state.cpi
unemp = st.session_state.unemp
week_date = st.session_state.week_date

# ============================================================
# Topbar (fake navbar)
# ============================================================
top = st.container()
with top:
    left, mid, right = st.columns([0.42, 0.38, 0.20], gap="large")
    with left:
        st.markdown(
            """
            <div class="topbar">
              <div class="brand"><span>▮▮▮</span> MetriX <span style="color:rgba(255,255,255,0.55); font-weight:700;">| Analytics</span></div>
              <div class="navhint">Real-time sales dashboard · Forecast & predictions</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with mid:
        st.markdown(
            """
            <div class="topbar">
              <div class="kpi-title">Current View</div>
              <div class="kpi-value">Walmart Weekly Sales</div>
              <div class="kpi-sub">Use the sidebar menu to switch pages</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="topbar">
              <div class="kpi-title">Status</div>
              <div class="kpi-value" style="font-size:1.2rem;">✅ Online</div>
              <div class="kpi-sub">Model loaded</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================
# Page: Overview
# ============================================================
if page == "🏠 Overview":
    st.markdown("## REAL-TIME SALES DASHBOARD")
    st.markdown('<div class="muted">Tracks predicted weekly revenue metrics and quick summaries.</div>', unsafe_allow_html=True)

    # KPIs based on last result / history
    hist = pd.DataFrame(st.session_state.history) if len(st.session_state.history) else pd.DataFrame()
    if len(hist):
        hist["pred_sales"] = pd.to_numeric(hist["pred_sales"], errors="coerce")
        hist = hist.dropna(subset=["pred_sales"])
    last_pred = st.session_state.last_result

    k1, k2, k3, k4 = st.columns(4, gap="large")

    with k1:
        val = money(last_pred["pred"]) if (last_pred and last_pred.get("mode") == "single") else "—"
        st.markdown(f"""<div class="card"><div class="kpi-title">Latest Forecast</div><div class="kpi-value">{val}</div><div class="kpi-sub">Most recent prediction</div></div>""", unsafe_allow_html=True)

    with k2:
        val = str(store)
        st.markdown(f"""<div class="card"><div class="kpi-title">Selected Store</div><div class="kpi-value">{val}</div><div class="kpi-sub">Current input state</div></div>""", unsafe_allow_html=True)

    with k3:
        val = "Holiday" if holiday_flag == 1 else "Non-holiday"
        st.markdown(f"""<div class="card"><div class="kpi-title">Week Type</div><div class="kpi-value">{val}</div><div class="kpi-sub">Holiday flag view</div></div>""", unsafe_allow_html=True)

    with k4:
        val = str(len(st.session_state.history))
        st.markdown(f"""<div class="card"><div class="kpi-title">Predictions</div><div class="kpi-value">{val}</div><div class="kpi-sub">This session</div></div>""", unsafe_allow_html=True)

    st.markdown("### Forecast Trend")
    if len(hist) >= 2:
        st.line_chart(hist["pred_sales"])
    else:
        st.markdown('<div class="card-soft"><div class="muted">Make at least 2 predictions to see a trend chart here.</div></div>', unsafe_allow_html=True)

    st.markdown("### Recent Predictions")
    if len(hist):
        st.dataframe(hist.tail(8).iloc[::-1], use_container_width=True)
    else:
        st.markdown('<div class="card-soft"><div class="muted">No predictions yet. Go to <b>Predict</b> from the sidebar.</div></div>', unsafe_allow_html=True)

# ============================================================
# Page: Predict
# ============================================================
elif page == "🧮 Predict":
    st.markdown("## Sales Forecasting")
    st.markdown('<div class="muted">Predict weekly sales using store + economic indicators. Includes validation and what-if comparison.</div>', unsafe_allow_html=True)

    issues = validate_inputs(temp, fuel, cpi, unemp)
    if issues:
        st.warning("Input validation found issues:")
        for msg in issues:
            st.write(f"• {msg}")

    # Run prediction actions
    if predict_btn:
        if issues:
            st.error("Fix the issues above, then predict again.")
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

    if compare_btn:
        if issues:
            st.error("Fix the issues above, then compare again.")
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

                st.info("What-if analysis done ✅ (only Holiday_Flag changed)")
            except Exception as e:
                st.error("Comparison failed. Please try again.")
                st.exception(e)

    # Main area layout
    left, right = st.columns([0.68, 0.32], gap="large")

    with left:
        st.markdown("### Output")
        res = st.session_state.last_result

        if res is None:
            st.markdown(
                """
                <div class="card-soft">
                  <div class="muted">
                    No output yet. Use sidebar inputs and click <b>Predict</b> or <b>Compare</b>.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            if res["mode"] == "single":
                a, b, c = st.columns(3, gap="large")
                with a:
                    st.markdown(f"""<div class="card"><div class="kpi-title">Predicted Weekly Sales</div><div class="kpi-value">{money(res["pred"])}</div><div class="kpi-sub">Estimated revenue</div></div>""", unsafe_allow_html=True)
                with b:
                    st.markdown(f"""<div class="card"><div class="kpi-title">Store</div><div class="kpi-value">{res["store"]}</div><div class="kpi-sub">Store-specific demand</div></div>""", unsafe_allow_html=True)
                with c:
                    wt = "Holiday" if res["holiday_flag"] == 1 else "Non-holiday"
                    st.markdown(f"""<div class="card"><div class="kpi-title">Week Type</div><div class="kpi-value">{wt}</div><div class="kpi-sub">Seasonal uplift</div></div>""", unsafe_allow_html=True)

                with st.expander("See model features used"):
                    st.dataframe(res["X_used"], use_container_width=True)

            if res["mode"] == "compare":
                a, b, c = st.columns(3, gap="large")
                with a:
                    st.markdown(f"""<div class="card"><div class="kpi-title">Non-holiday</div><div class="kpi-value">{money(res["pred_non"])}</div><div class="kpi-sub">Holiday_Flag = 0</div></div>""", unsafe_allow_html=True)
                with b:
                    st.markdown(f"""<div class="card"><div class="kpi-title">Holiday</div><div class="kpi-value">{money(res["pred_hol"])}</div><div class="kpi-sub">Holiday_Flag = 1</div></div>""", unsafe_allow_html=True)
                with c:
                    st.markdown(f"""<div class="card"><div class="kpi-title">Uplift</div><div class="kpi-value">{money(res["diff"])}</div><div class="kpi-sub">{res["pct"]:.2f}% change</div></div>""", unsafe_allow_html=True)

                with st.expander("See both feature inputs (debug)"):
                    st.write("Non-holiday features:")
                    st.dataframe(res["X_non"], use_container_width=True)
                    st.write("Holiday features:")
                    st.dataframe(res["X_hol"], use_container_width=True)

        st.markdown("### Session History")
        if len(st.session_state.history) == 0:
            st.caption("No predictions yet.")
        else:
            hist_df = pd.DataFrame(st.session_state.history).iloc[::-1].reset_index(drop=True)
            with st.expander("Show history table", expanded=False):
                st.dataframe(hist_df, use_container_width=True)

            numeric_hist = hist_df.copy()
            numeric_hist["pred_sales"] = pd.to_numeric(numeric_hist["pred_sales"], errors="coerce")
            numeric_hist = numeric_hist.dropna(subset=["pred_sales"])
            if len(numeric_hist) >= 2:
                st.markdown("### Trend (Predicted Sales)")
                st.line_chart(numeric_hist["pred_sales"])

    with right:
        st.markdown("### Guidance")
        st.markdown(
            """
            <div class="card">
              <div class="section-title">User-friendly Feedback</div>
              <div class="muted">
                <b>Why validate inputs?</b><br>
                Prevents crashes and keeps predictions realistic.<br><br>
                <b>How to show “interactive”:</b><br>
                Use <b>Compare</b> to demonstrate decision support for holiday planning.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="card" style="margin-top:0.9rem;">
              <div class="section-title">Quick Summary</div>
              <div class="kpi-title">Store</div>
              <div class="kpi-value" style="font-size:1.3rem;">{store}</div>
              <div class="kpi-title" style="margin-top:.5rem;">Week Date</div>
              <div class="muted">{week_date}</div>
              <div class="kpi-title" style="margin-top:.5rem;">Week Type</div>
              <div class="muted">{'Holiday' if holiday_flag==1 else 'Non-holiday'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================
# Page: Insights
# ============================================================
elif page == "📈 Insights":
    st.markdown("## Insights")
    st.markdown('<div class="muted">Business interpretation of features and predicted outputs.</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="card">
          <div class="section-title">Retail Insights</div>
          <div class="muted">
            <b>Inventory Planning:</b> Higher predicted sales → stock fast-moving essentials.<br>
            <b>Staffing:</b> Holiday weeks often require more manpower and extended operating hours.<br>
            <b>Economic Context:</b> CPI and unemployment influence spending power and shopping behaviour.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    hist = pd.DataFrame(st.session_state.history) if len(st.session_state.history) else pd.DataFrame()
    if len(hist):
        hist["pred_sales"] = pd.to_numeric(hist["pred_sales"], errors="coerce")
        hist = hist.dropna(subset=["pred_sales"])

    st.markdown("### Session Analytics")
    a, b, c = st.columns(3, gap="large")
    with a:
        st.markdown(f"""<div class="card-soft"><div class="kpi-title">Predictions Made</div><div class="kpi-value" style="font-size:1.25rem;">{len(st.session_state.history)}</div></div>""", unsafe_allow_html=True)
    with b:
        st.markdown(f"""<div class="card-soft"><div class="kpi-title">Avg Forecast</div><div class="kpi-value" style="font-size:1.25rem;">{money(hist["pred_sales"].mean()) if len(hist) else "—"}</div></div>""", unsafe_allow_html=True)
    with c:
        st.markdown(f"""<div class="card-soft"><div class="kpi-title">Max Forecast</div><div class="kpi-value" style="font-size:1.25rem;">{money(hist["pred_sales"].max()) if len(hist) else "—"}</div></div>""", unsafe_allow_html=True)

    if len(hist) >= 2:
        st.markdown("### Predicted Sales Trend")
        st.line_chart(hist["pred_sales"])
    else:
        st.caption("Make at least 2 predictions to see trend analytics.")

# ============================================================
# Page: Model
# ============================================================
elif page == "🧪 Model":
    st.markdown("## Model Performance")
    st.markdown('<div class="muted">Add your MAE/RMSE results and evidence here for the report.</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="card">
          <div class="section-title">What to include (for A grade)</div>
          <div class="muted">
            • MAE and RMSE comparison table (baseline vs best model)<br>
            • Predicted vs Actual plot for best model<br>
            • Brief explanation why RMSE matters (penalises spikes during holidays)
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# Page: About
# ============================================================
else:
    st.markdown("## About")
    st.markdown(
        """
        <div class="card">
          <div class="section-title">Application Summary</div>
          <div class="muted">
            <b>Goal:</b> Predict weekly Walmart sales to support operational planning.<br>
            <b>Model:</b> scikit-learn model loaded from <code>model.pkl</code>.<br>
            <b>Deployment:</b> Streamlit dashboard UI with validation, what-if analysis and exportable history.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
