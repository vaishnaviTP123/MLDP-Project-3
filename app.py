import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Walmart Weekly Sales Prediction",
    page_icon="🛒",
    layout="wide",
)

# ----------------------------
# Navy + Black Theme (more colour)
# ----------------------------
st.markdown(
    """
    <style>
      /* Page background */
      [data-testid="stAppViewContainer"]{
        background: radial-gradient(1200px 600px at 20% 0%, rgba(37,99,235,0.18), transparent 55%),
                    radial-gradient(900px 500px at 90% 10%, rgba(99,102,241,0.14), transparent 60%),
                    linear-gradient(180deg, #050814 0%, #040711 45%, #04060f 100%);
      }

      /* Sidebar background */
      [data-testid="stSidebar"]{
        background: linear-gradient(180deg, rgba(10,15,30,0.95) 0%, rgba(5,8,20,0.98) 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
      }

      /* Make blocks feel "filled" */
      .block-container { padding-top: 1.8rem; padding-bottom: 2.2rem; }

      /* Typography */
      h1,h2,h3 { letter-spacing: -0.3px; }

      /* Cards */
      .card {
        border: 1px solid rgba(255,255,255,0.09);
        background: rgba(255,255,255,0.04);
        border-radius: 18px;
        padding: 1.0rem 1.1rem;
        box-shadow: 0 10px 28px rgba(0,0,0,0.35);
      }

      .kpi {
        border: 1px solid rgba(255,255,255,0.10);
        background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
        border-radius: 18px;
        padding: 1.0rem 1.1rem;
      }

      .pill {
        display:inline-block;
        padding: .25rem .65rem;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.06);
        font-size: .85rem;
        color: rgba(255,255,255,0.80);
        margin-right: .35rem;
      }

      .muted { color: rgba(255,255,255,0.70); font-size: 0.96rem; }
      .tiny { color: rgba(255,255,255,0.60); font-size: 0.88rem; }

      .big-number { font-size: 2.25rem; font-weight: 850; margin: .2rem 0 .15rem 0; }
      .divider { height: 1px; background: rgba(255,255,255,0.10); margin: 0.95rem 0; }

      /* Buttons: more premium */
      .stButton>button {
        border-radius: 14px !important;
        border: 1px solid rgba(255,255,255,0.14) !important;
        background: linear-gradient(180deg, rgba(37,99,235,0.22), rgba(37,99,235,0.08)) !important;
        color: rgba(255,255,255,0.92) !important;
        padding: 0.70rem 0.95rem !important;
      }
      .stButton>button:hover {
        border: 1px solid rgba(99,102,241,0.35) !important;
        background: linear-gradient(180deg, rgba(99,102,241,0.28), rgba(37,99,235,0.10)) !important;
      }

      /* Input widgets */
      [data-baseweb="select"] > div, .stTextInput input, .stDateInput input {
        border-radius: 12px !important;
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Load model + columns
# ----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    cols = joblib.load("columns.pkl")
    return model, cols

def money(x: float) -> str:
    return "${:,.2f}".format(float(x))

def build_features(store, holiday_flag, temp, fuel, cpi, unemp, week_date, feature_cols):
    dt = pd.to_datetime(week_date)
    base = {
        "Store": int(store),
        "Holiday_Flag": int(holiday_flag),
        "Temperature": float(temp),
        "Fuel_Price": float(fuel),
        "CPI": float(cpi),
        "Unemployment": float(unemp),
        # Date features only matter if your model expects them
        "Year": int(dt.year),
        "Month": int(dt.month),
        "WeekOfYear": int(dt.isocalendar().week),
        "DayOfWeek": int(dt.dayofweek),
    }
    X = pd.DataFrame([base]).reindex(columns=feature_cols, fill_value=0)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    return X

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

# ----------------------------
# Session state
# ----------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "preset_loaded" not in st.session_state:
    st.session_state.preset_loaded = False

model, feature_cols = load_artifacts()

# ----------------------------
# Header
# ----------------------------
st.markdown("## 🛒 Walmart Weekly Sales Prediction")
st.markdown(
    '<span class="pill">Navy/Black Theme</span>'
    '<span class="pill">Supervised ML · Regression</span>'
    '<span class="pill">Streamlit Deployment</span>'
    '<span class="pill">Input Validation</span>',
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='muted'>Estimate weekly sales using store conditions and economic indicators. "
    "This supports decisions like inventory planning and staffing for peak weeks.</div>",
    unsafe_allow_html=True,
)
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ----------------------------
# Sidebar (form + more working buttons)
# ----------------------------
with st.sidebar:
    st.markdown("### Input Features")
    st.caption("Enter conditions for the week to estimate sales.")

    # Use a form so inputs + predict feel like a proper “app”
    with st.form("input_form", clear_on_submit=False):
        store = st.selectbox("Select Store", list(range(1, 46)), index=0)

        holiday_label = st.selectbox("Holiday Flag", ["Non-holiday Week", "Holiday Week"], index=0)
        holiday_flag = 1 if holiday_label == "Holiday Week" else 0

        temp = st.slider("Temperature (°F)", min_value=-5.0, max_value=105.0, value=60.0, step=0.1)
        fuel = st.slider("Fuel Price", min_value=2.0, max_value=5.0, value=3.50, step=0.01)
        cpi = st.slider("CPI", min_value=120.0, max_value=230.0, value=180.0, step=0.1)
        unemp = st.slider("Unemployment Rate (%)", min_value=3.0, max_value=15.0, value=7.50, step=0.01)

        week_date = st.date_input("Week Date", value=date(2012, 2, 10))

        col1, col2 = st.columns(2)
        with col1:
            predict_btn = st.form_submit_button("Predict", use_container_width=True)
        with col2:
            compare_btn = st.form_submit_button("Holiday vs Non-holiday", use_container_width=True)

    st.markdown("---")
    st.markdown("### Quick Actions")

    qa1, qa2 = st.columns(2)
    with qa1:
        if st.button("Load Example", use_container_width=True):
            # store example in session (we just mark it; user sees changed values by re-run)
            st.session_state.preset_loaded = True
            st.success("Example preset loaded. Scroll up to Predict.")
    with qa2:
        if st.button("Reset Inputs", use_container_width=True):
            # Streamlit can't directly overwrite widget values easily without keys,
            # but we can force a rerun and rely on default values.
            st.session_state.preset_loaded = False
            st.info("Inputs reset to defaults (app refreshed).")
            st.rerun()

    qa3, qa4 = st.columns(2)
    with qa3:
        if st.button("Clear History", use_container_width=True):
            st.session_state.history = []
            st.success("Prediction history cleared.")
    with qa4:
        # download history (works if exists)
        if len(st.session_state.history) > 0:
            hist_df = pd.DataFrame(st.session_state.history)
            st.download_button(
                "Download History",
                data=hist_df.to_csv(index=False).encode("utf-8"),
                file_name="prediction_history.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.button("Download History", disabled=True, use_container_width=True)

# If user pressed "Load Example", override displayed defaults by defining a preset here
# (This is a best-effort approach; for full control you'd add keys to every widget.)
if st.session_state.preset_loaded:
    store = 5
    holiday_flag = 1
    holiday_label = "Holiday Week"
    temp = 45.0
    fuel = 3.20
    cpi = 210.0
    unemp = 8.60
    week_date = date(2011, 11, 25)

# ----------------------------
# Tabs to make app feel "full"
# ----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📌 Predict", "📊 Insights", "🧪 Model Performance", "ℹ️ About"])

# ============================
# TAB 1: Predict
# ============================
with tab1:
    left, right = st.columns([0.62, 0.38], gap="large")

    with left:
        st.markdown("### Prediction Output")

        issues = validate_inputs(temp, fuel, cpi, unemp)
        if issues:
            st.warning("Please review your inputs:")
            for msg in issues:
                st.write(f"• {msg}")

        def do_predict(holiday_value):
            X = build_features(store, holiday_value, temp, fuel, cpi, unemp, week_date, feature_cols)
            pred = float(model.predict(X)[0])
            return pred, X

        if predict_btn:
            try:
                pred, X_used = do_predict(holiday_flag)

                st.success("Prediction generated successfully ✅")

                k1, k2, k3 = st.columns(3)
                with k1:
                    st.markdown(
                        f"<div class='kpi'><div class='tiny'>Predicted Weekly Sales</div>"
                        f"<div class='big-number'>{money(pred)}</div>"
                        f"<div class='muted'>Estimated revenue for the selected week</div></div>",
                        unsafe_allow_html=True,
                    )
                with k2:
                    st.markdown(
                        f"<div class='kpi'><div class='tiny'>Store ID</div>"
                        f"<div class='big-number'>{store}</div>"
                        f"<div class='muted'>Store-specific demand patterns</div></div>",
                        unsafe_allow_html=True,
                    )
                with k3:
                    st.markdown(
                        f"<div class='kpi'><div class='tiny'>Week Type</div>"
                        f"<div class='big-number'>{'Holiday' if holiday_flag==1 else 'Non-holiday'}</div>"
                        f"<div class='muted'>Seasonal uplift indicator</div></div>",
                        unsafe_allow_html=True,
                    )

                # Log history
                st.session_state.history.append({
                    "date": str(week_date),
                    "store": int(store),
                    "holiday_flag": int(holiday_flag),
                    "temperature": float(temp),
                    "fuel_price": float(fuel),
                    "cpi": float(cpi),
                    "unemployment": float(unemp),
                    "pred_sales": pred
                })

                with st.expander("See input values used (model features)"):
                    st.dataframe(X_used, use_container_width=True)

            except Exception as e:
                st.error("Something went wrong while predicting. Please try again.")
                st.exception(e)

        if compare_btn:
            try:
                pred_non, X_non = do_predict(0)
                pred_hol, X_hol = do_predict(1)

                st.info("What-if analysis: Holiday vs Non-holiday (same inputs, only Holiday_Flag changed)")

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(
                        f"<div class='kpi'><div class='tiny'>Non-holiday Prediction</div>"
                        f"<div class='big-number'>{money(pred_non)}</div>"
                        f"<div class='muted'>Holiday_Flag = 0</div></div>",
                        unsafe_allow_html=True,
                    )
                with c2:
                    st.markdown(
                        f"<div class='kpi'><div class='tiny'>Holiday Prediction</div>"
                        f"<div class='big-number'>{money(pred_hol)}</div>"
                        f"<div class='muted'>Holiday_Flag = 1</div></div>",
                        unsafe_allow_html=True,
                    )

                diff = pred_hol - pred_non
                st.markdown(
                    f"<div class='card'><b>Estimated uplift:</b> {money(diff)} "
                    f"({(diff / pred_non * 100 if pred_non != 0 else 0):.2f}% change)</div>",
                    unsafe_allow_html=True,
                )

                st.session_state.history.append({
                    "date": str(week_date),
                    "store": int(store),
                    "holiday_flag": "compare",
                    "temperature": float(temp),
                    "fuel_price": float(fuel),
                    "cpi": float(cpi),
                    "unemployment": float(unemp),
                    "pred_sales": float(pred_hol)
                })

                with st.expander("See both feature inputs (debug/validation)"):
                    st.write("Non-holiday features:")
                    st.dataframe(X_non, use_container_width=True)
                    st.write("Holiday features:")
                    st.dataframe(X_hol, use_container_width=True)

            except Exception as e:
                st.error("Comparison failed. Please try again.")
                st.exception(e)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("### Prediction History (this session)")

        if len(st.session_state.history) == 0:
            st.caption("No predictions yet. Use Predict to generate outputs.")
        else:
            hist_df = pd.DataFrame(st.session_state.history).iloc[::-1].reset_index(drop=True)
            st.dataframe(hist_df, use_container_width=True)

            st.markdown("#### Trend (Predicted Sales)")
            st.line_chart(hist_df["pred_sales"])

    with right:
        st.markdown("### User-friendly Feedback")
        st.markdown(
            """
            <div class="card">
              <div class="muted">
                <b>Why we validate inputs:</b><br>
                Prevents crashes and ensures predictions remain realistic for users.<br><br>
                <b>Tip:</b> Use Holiday vs Non-holiday to show investor-grade interactivity.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("### Quick Summary")
        st.markdown(
            f"""
            <div class="card">
              <div class="tiny">Selected Store</div>
              <div class="big-number">{store}</div>
              <div class="tiny">Week Date</div>
              <div class="muted">{week_date}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================
# TAB 2: Insights
# ============================
with tab2:
    st.markdown("### Business Insights (Retail Context)")
    st.markdown(
        """
        <div class="card">
          <div class="muted">
            <b>Inventory Planning:</b> Higher predicted sales → stock fast-moving SKUs and essentials.<br>
            <b>Staffing:</b> Holiday weeks usually require more manpower and longer operating hours.<br>
            <b>Pricing/Promotions:</b> When CPI rises or unemployment increases, shoppers may become more price-sensitive.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("### Why these features exist (your earlier question)")
    st.info(
        "The dataset includes economic indicators (Fuel Price, CPI, Unemployment) because they affect consumer spending power "
        "and shopping behaviour, which influences Walmart’s weekly sales."
    )

# ============================
# TAB 3: Model Performance
# ============================
with tab3:
    st.markdown("### Model Performance Summary")
    st.caption("Use your notebook results (MAE/RMSE) to justify model selection.")

    st.markdown(
        """
        <div class="card">
          <div class="muted">
            <b>Suggested evaluation metrics:</b><br>
            • MAE (average absolute error in dollars) – easy to explain to business users.<br>
            • RMSE (penalizes big errors more) – helps avoid very wrong predictions.<br><br>
            Put your final comparison table screenshot here in your report.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================
# TAB 4: About
# ============================
with tab4: 
    st.markdown("### About this Application")
    st.markdown(
        """
        <div class="card">
          <div class="muted">
            <b>Problem:</b> Predict weekly Walmart sales to support operational planning.<br>
            <b>Type:</b> Supervised learning (Regression).<br>
            <b>Model:</b> scikit-learn RandomForestRegressor.<br>
            <b>Deployment:</b> Streamlit web application with input validation and interactive what-if analysis.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
