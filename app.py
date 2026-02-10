import streamlit as st
import pandas as pd
import joblib
from datetime import date
from pathlib import Path

# ============================================================
# Page config (IMPORTANT: expanded so you don't "lose" it)
# ============================================================
st.set_page_config(
    page_title="Walmart Weekly Sales Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CSS: White + Blue/Purple theme, KEEP header (for sidebar toggle)
# ============================================================
st.markdown(
    """
    <style>
      /* Hide menu + footer only (DO NOT hide header, or sidebar toggle disappears) */
      #MainMenu { visibility: hidden; }
      footer { visibility: hidden; }

      /* Make the top header/toolbar clean (no ugly dark bar) */
      [data-testid="stHeader"]{
        background: rgba(255,255,255,0.85) !important;
        border-bottom: 1px solid rgba(15,23,42,0.08) !important;
        backdrop-filter: blur(10px);
      }
      [data-testid="stToolbar"]{
        background: transparent !important;
      }

      /* Page background */
      [data-testid="stAppViewContainer"]{
        background: linear-gradient(180deg, #f6f7ff 0%, #ffffff 60%, #f7f7ff 100%);
      }

      /* Force text dark */
      html, body, [class*="css"] { color: #0f172a !important; }
      h1,h2,h3,h4,h5,h6,p,span,div,label { color:#0f172a !important; }

      /* Container spacing */
      .block-container{
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
      }

      /* Sidebar */
      [data-testid="stSidebar"]{
        background: linear-gradient(180deg, #ffffff 0%, #f6f7ff 100%);
        border-right: 1px solid rgba(15,23,42,0.10);
      }

      .sb-title{ font-size: 1.1rem; font-weight: 900; margin: 0.2rem 0 0.1rem 0; }
      .sb-sub{ font-size: 0.9rem; color: rgba(15,23,42,0.65) !important; margin-bottom: 0.8rem; }

      /* Cards */
      .card{
        background: #ffffff;
        border: 1px solid rgba(15, 23, 42, 0.10);
        border-radius: 18px;
        padding: 1.05rem 1.1rem;
        box-shadow: 0 10px 22px rgba(15, 23, 42, 0.06);
      }
      .card-soft{
        background: #ffffff;
        border: 1px dashed rgba(99,102,241, 0.45);
        border-radius: 18px;
        padding: 1rem 1.1rem;
      }

      .kpi-title{
        font-size: 0.85rem;
        color: rgba(15,23,42,0.65) !important;
        font-weight: 800;
      }
      .kpi-value{
        font-size: 1.7rem;
        font-weight: 950;
        margin-top: 0.1rem;
        background: linear-gradient(90deg, #2563eb 0%, #7c3aed 55%, #db2777 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
      }
      .kpi-sub{
        font-size: 0.9rem;
        color: rgba(15,23,42,0.65) !important;
        margin-top: 0.25rem;
      }
      .muted{ color: rgba(15,23,42,0.65) !important; }

      /* Inputs */
      [data-baseweb="select"] > div,
      .stNumberInput input,
      .stDateInput input,
      .stTextInput input {
        border-radius: 12px !important;
        background: #ffffff !important;
        border: 1px solid rgba(15,23,42,0.14) !important;
        color: #0f172a !important;
      }
      label{ color: rgba(15,23,42,0.85) !important; font-weight: 800 !important; }

      /* Sidebar radio menu */
      div[role="radiogroup"] label{
        background: rgba(99,102,241,0.06);
        border: 1px solid rgba(99,102,241,0.18);
        padding: 0.65rem 0.75rem;
        border-radius: 14px;
        margin-bottom: 0.45rem;
        transition: all 0.15s ease;
      }
      div[role="radiogroup"] label:hover{
        background: rgba(99,102,241,0.12);
        border: 1px solid rgba(99,102,241,0.28);
        transform: translateY(-1px);
      }

      /* Buttons */
      .stButton>button, .stDownloadButton>button {
        border-radius: 14px !important;
        border: 1px solid rgba(99,102,241,0.28) !important;
        background: linear-gradient(180deg, rgba(99,102,241,0.22), rgba(99,102,241,0.10)) !important;
        color: #0f172a !important;
        font-weight: 900 !important;
        padding: 0.75rem 1rem !important;
      }
      .stButton>button:hover, .stDownloadButton>button:hover {
        border: 1px solid rgba(124,58,237,0.35) !important;
        background: linear-gradient(180deg, rgba(124,58,237,0.25), rgba(124,58,237,0.10)) !important;
      }

      /* Dataframe */
      .stDataFrame { border-radius: 14px; overflow: hidden; }

      /* Tiny tip pill */
      .tip-pill{
        display:inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(37,99,235,0.10);
        border: 1px solid rgba(37,99,235,0.18);
        font-weight: 800;
        color: rgba(15,23,42,0.75) !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Load model + columns (safe path for Streamlit Cloud)
# ============================================================
APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "model.pkl"
COLS_PATH = APP_DIR / "columns.pkl"

@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not COLS_PATH.exists():
        st.error("⚠️ model.pkl / columns.pkl not found. Make sure they are beside app.py in the repo.")
        st.stop()
    model = joblib.load(MODEL_PATH)
    cols = joblib.load(COLS_PATH)
    cols = list(cols) if not isinstance(cols, list) else cols
    return model, cols

model, feature_cols = load_artifacts()

# ============================================================
# Helpers
# ============================================================
def money(x: float) -> str:
    return "${:,.2f}".format(float(x))

def validate_inputs(temp, fuel, cpi, unemp):
    issues = []
    if fuel < 0: issues.append("Fuel Price must be ≥ 0.")
    if cpi < 0: issues.append("CPI must be ≥ 0.")
    if not (-10 <= temp <= 120): issues.append("Temperature should be between -10°F and 120°F.")
    if not (0 <= unemp <= 25): issues.append("Unemployment should be between 0% and 25%.")
    return issues

def build_features(store, holiday_flag, temp, fuel, cpi, unemp, week_date):
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
    X = build_features(store, holiday_flag, temp, fuel, cpi, unemp, week_date)
    pred = float(model.predict(X)[0])
    return pred, X

# ============================================================
# Defaults
# Reset requirement: "set everything to 0 again"
# (Store cannot be 0 because dropdown; date cannot be 0 because it must be a date)
# So we reset numeric values to 0 + keep store=1 + keep a default date.
# ============================================================
RESET_DEFAULTS = {
    "store": 1,
    "holiday_label": "Non-holiday Week",
    "temp": 0.0,
    "fuel": 0.0,
    "cpi": 0.0,
    "unemp": 0.0,
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

# ============================================================
# Session State init
# ============================================================
for k, v in RESET_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

# Widget keys (separate to avoid session_state mutation errors)
WIDGET_MAP = {
    "store_w": "store",
    "week_date_w": "week_date",
    "holiday_label_w": "holiday_label",
    "temp_w": "temp",
    "fuel_w": "fuel",
    "cpi_w": "cpi",
    "unemp_w": "unemp",
}

for w_key, state_key in WIDGET_MAP.items():
    if w_key not in st.session_state:
        st.session_state[w_key] = st.session_state[state_key]

def sync_widgets_to_state():
    for w_key, state_key in WIDGET_MAP.items():
        st.session_state[state_key] = st.session_state[w_key]

# Preset queue (safe)
if "pending_preset" not in st.session_state:
    st.session_state.pending_preset = None

def queue_preset(name: str):
    st.session_state.pending_preset = name
    st.rerun()

def apply_pending_preset():
    name = st.session_state.pending_preset
    if not name:
        return

    preset = EXAMPLE if name == "example" else RESET_DEFAULTS

    st.session_state["store_w"] = preset["store"]
    st.session_state["week_date_w"] = preset["week_date"]
    st.session_state["holiday_label_w"] = preset["holiday_label"]
    st.session_state["temp_w"] = preset["temp"]
    st.session_state["fuel_w"] = preset["fuel"]
    st.session_state["cpi_w"] = preset["cpi"]
    st.session_state["unemp_w"] = preset["unemp"]

    # Reset also clears output + history
    if name == "reset":
        st.session_state.history = []
        st.session_state.last_result = None

    sync_widgets_to_state()
    st.session_state.pending_preset = None

apply_pending_preset()

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown('<div class="sb-title">📊 Walmart Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-sub">Forecasting + analytics</div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠 Overview", "🧮 Predict", "📈 Insights", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    if page == "🧮 Predict":
        st.markdown("### Inputs")

        st.selectbox("Store", list(range(1, 46)), key="store_w")
        st.date_input("Week Date", key="week_date_w")
        st.selectbox("Week Type", ["Non-holiday Week", "Holiday Week"], key="holiday_label_w")
        st.slider("Temperature (°F)", -5.0, 105.0, step=0.1, key="temp_w")

        with st.expander("Advanced Inputs", expanded=False):
            st.slider("Fuel Price", 0.0, 5.0, step=0.01, key="fuel_w")
            st.slider("CPI", 0.0, 230.0, step=0.1, key="cpi_w")
            st.slider("Unemployment (%)", 0.0, 25.0, step=0.01, key="unemp_w")

        sync_widgets_to_state()

        st.markdown("### Actions")
        c1, c2 = st.columns(2)
        with c1:
            predict_btn = st.button("Predict", use_container_width=True)
        with c2:
            compare_btn = st.button("Compare", use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            if st.button("Example", use_container_width=True):
                queue_preset("example")
        with c4:
            if st.button("Reset", use_container_width=True):
                queue_preset("reset")

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

# ============================================================
# Read inputs
# ============================================================
store = st.session_state.store
holiday_flag = 1 if st.session_state.holiday_label == "Holiday Week" else 0
temp = st.session_state.temp
fuel = st.session_state.fuel
cpi = st.session_state.cpi
unemp = st.session_state.unemp
week_date = st.session_state.week_date

# ============================================================
# Main header + obvious sidebar tip
# ============================================================
st.markdown("## Walmart Weekly Sales Forecasting")
st.markdown('<div class="muted">Predict weekly sales using store + economic indicators. Includes what-if comparison.</div>', unsafe_allow_html=True)
st.markdown('<span class="tip-pill">Tip: If you collapse the sidebar, use the small arrow (>) at the top-left to open it again.</span>', unsafe_allow_html=True)
st.markdown("")

# ============================================================
# Pages
# ============================================================
if page == "🏠 Overview":
    hist = pd.DataFrame(st.session_state.history) if len(st.session_state.history) else pd.DataFrame()
    if len(hist):
        hist["pred_sales"] = pd.to_numeric(hist["pred_sales"], errors="coerce")
        hist = hist.dropna(subset=["pred_sales"])

    last_pred = st.session_state.last_result

    k1, k2, k3, k4 = st.columns(4, gap="large")

    with k1:
        val = money(last_pred["pred"]) if (last_pred and last_pred.get("mode") == "single") else "—"
        st.markdown(f"""<div class="card"><div class="kpi-title">Latest Forecast</div><div class="kpi-value">{val}</div><div class="kpi-sub">Most recent</div></div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="card"><div class="kpi-title">Selected Store</div><div class="kpi-value">{store}</div><div class="kpi-sub">Current input</div></div>""", unsafe_allow_html=True)
    with k3:
        wt = "Holiday" if holiday_flag == 1 else "Non-holiday"
        st.markdown(f"""<div class="card"><div class="kpi-title">Week Type</div><div class="kpi-value">{wt}</div><div class="kpi-sub">Seasonality</div></div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class="card"><div class="kpi-title">Predictions</div><div class="kpi-value">{len(st.session_state.history)}</div><div class="kpi-sub">This session</div></div>""", unsafe_allow_html=True)

    st.markdown("### Forecast Trend")
    if len(hist) >= 2:
        st.line_chart(hist["pred_sales"])
    else:
        st.markdown('<div class="card-soft"><div class="muted">Make at least 2 predictions to see a trend.</div></div>', unsafe_allow_html=True)

    st.markdown("### Recent Predictions")
    if len(hist):
        st.dataframe(hist.tail(10).iloc[::-1], use_container_width=True)
    else:
        st.markdown('<div class="card-soft"><div class="muted">No predictions yet. Go to <b>Predict</b>.</div></div>', unsafe_allow_html=True)

elif page == "🧮 Predict":
    issues = validate_inputs(temp, fuel, cpi, unemp)
    if issues:
        st.warning("Input validation found issues:")
        for msg in issues:
            st.write(f"• {msg}")

    if predict_btn:
        if issues:
            st.error("Fix the issues above, then predict again.")
        else:
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

    if compare_btn:
        if issues:
            st.error("Fix the issues above, then compare again.")
        else:
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
            st.info("What-if comparison done ✅")

    left, right = st.columns([0.68, 0.32], gap="large")

    with left:
        st.markdown("### Output")
        res = st.session_state.last_result

        if res is None:
            st.markdown(
                """<div class="card-soft"><div class="muted">No output yet. Use sidebar inputs and click <b>Predict</b> or <b>Compare</b>.</div></div>""",
                unsafe_allow_html=True,
            )
        else:
            if res["mode"] == "single":
                a, b, c = st.columns(3, gap="large")
                with a:
                    st.markdown(f"""<div class="card"><div class="kpi-title">Predicted Weekly Sales</div><div class="kpi-value">{money(res["pred"])}</div><div class="kpi-sub">Estimated revenue</div></div>""", unsafe_allow_html=True)
                with b:
                    st.markdown(f"""<div class="card"><div class="kpi-title">Store</div><div class="kpi-value">{res["store"]}</div><div class="kpi-sub">Store selected</div></div>""", unsafe_allow_html=True)
                with c:
                    wt = "Holiday" if res["holiday_flag"] == 1 else "Non-holiday"
                    st.markdown(f"""<div class="card"><div class="kpi-title">Week Type</div><div class="kpi-value">{wt}</div><div class="kpi-sub">Seasonality</div></div>""", unsafe_allow_html=True)

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
              <div class="kpi-title">Tips</div>
              <div class="kpi-sub">
                <b>Validation:</b> prevents unrealistic inputs from skewing predictions.<br><br>
                <b>Compare:</b> shows business impact of holiday vs non-holiday weeks.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="card" style="margin-top:0.9rem;">
              <div class="kpi-title">Quick Summary</div>
              <div class="kpi-sub"><b>Store:</b> {store}</div>
              <div class="kpi-sub"><b>Week Date:</b> {week_date}</div>
              <div class="kpi-sub"><b>Week Type:</b> {'Holiday' if holiday_flag==1 else 'Non-holiday'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

elif page == "📈 Insights":
    st.markdown("## Insights")
    st.markdown('<div class="muted">Simple analytics from your session predictions.</div>', unsafe_allow_html=True)

    hist = pd.DataFrame(st.session_state.history) if len(st.session_state.history) else pd.DataFrame()
    if len(hist):
        hist["pred_sales"] = pd.to_numeric(hist["pred_sales"], errors="coerce")
        hist = hist.dropna(subset=["pred_sales"])

    a, b, c = st.columns(3, gap="large")
    with a:
        st.markdown(f"""<div class="card"><div class="kpi-title">Predictions</div><div class="kpi-value">{len(st.session_state.history)}</div><div class="kpi-sub">This session</div></div>""", unsafe_allow_html=True)
    with b:
        st.markdown(f"""<div class="card"><div class="kpi-title">Avg Forecast</div><div class="kpi-value">{money(hist["pred_sales"].mean()) if len(hist) else "—"}</div><div class="kpi-sub">Mean</div></div>""", unsafe_allow_html=True)
    with c:
        st.markdown(f"""<div class="card"><div class="kpi-title">Max Forecast</div><div class="kpi-value">{money(hist["pred_sales"].max()) if len(hist) else "—"}</div><div class="kpi-sub">Peak</div></div>""", unsafe_allow_html=True)

    if len(hist) >= 2:
        st.markdown("### Trend")
        st.line_chart(hist["pred_sales"])
    else:
        st.caption("Make at least 2 predictions to see analytics.")

else:
    st.markdown("## About")
    st.markdown(
        """
        <div class="card">
          <div class="kpi-title">Summary</div>
          <div class="kpi-sub">
            Predict weekly Walmart sales to support planning (inventory, staffing, promotions).<br>
            Includes validation, what-if comparison, and CSV export.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
