# app.py
import streamlit as st
import pandas as pd
import joblib
from datetime import date
from pathlib import Path

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Walmart Weekly Sales Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# Theme CSS (dark blue/black, clean)
# ----------------------------
st.markdown(
    """
    <style>
      /* App background */
      [data-testid="stAppViewContainer"]{
        background:
          radial-gradient(1000px 560px at 15% 0%, rgba(59,130,246,0.16), transparent 60%),
          radial-gradient(900px 520px at 90% 10%, rgba(99,102,241,0.12), transparent 60%),
          linear-gradient(180deg, #05060f 0%, #050812 45%, #04050d 100%);
      }
      .block-container{ padding-top: 2.4rem; max-width: 1400px; }

      /* Sidebar */
      [data-testid="stSidebar"]{
        background: linear-gradient(180deg, rgba(10,12,24,0.98) 0%, rgba(6,8,18,0.99) 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
      }

      /* Hide Streamlit footer/menu */
      footer {visibility: hidden;}
      #MainMenu {visibility: hidden;}

      /* Headings */
      h1, h2, h3 { color: rgba(255,255,255,0.92) !important; }
      p, li, label, .stMarkdown { color: rgba(255,255,255,0.80); }

      /* Cards */
      .card{
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.04);
        border-radius: 18px;
        padding: 1.05rem 1.1rem;
        box-shadow: 0 12px 28px rgba(0,0,0,0.35);
      }
      .muted{ color: rgba(255,255,255,0.70); }

      /* Inputs */
      [data-baseweb="select"] > div, .stNumberInput input, .stDateInput input, .stTextInput input {
        border-radius: 12px !important;
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        color: rgba(255,255,255,0.92) !important;
      }

      /* Buttons */
      .stButton > button{
        border-radius: 14px !important;
        border: 1px solid rgba(255,255,255,0.14) !important;
        background: linear-gradient(180deg, rgba(59,130,246,0.25), rgba(99,102,241,0.10)) !important;
        color: rgba(255,255,255,0.95) !important;
        padding: 0.75rem 1rem !important;
        font-weight: 700 !important;
      }
      .stButton > button:hover{
        border: 1px solid rgba(59,130,246,0.40) !important;
        background: linear-gradient(180deg, rgba(59,130,246,0.35), rgba(99,102,241,0.12)) !important;
      }

      /* Success/Info look nicer */
      [data-testid="stAlert"]{
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.05);
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Load model (pipeline)
# ----------------------------
@st.cache_resource
def load_model():
    base = Path(__file__).parent
    model_path = base / "model.pkl"
    if not model_path.exists():
        st.error("❌ model.pkl not found in the same folder as app.py")
        st.stop()
    return joblib.load(model_path)

pipe = load_model()

# ----------------------------
# Helpers
# ----------------------------
def money(x: float) -> str:
    return "${:,.2f}".format(float(x))

def validate_inputs(temp, fuel, cpi, unemp):
    issues = []
    if fuel <= 0:
        issues.append("Fuel Price must be > 0.")
    if cpi <= 0:
        issues.append("CPI must be > 0.")
    if not (-10 <= temp <= 120):
        issues.append("Temperature should be between -10°F and 120°F.")
    if not (0 <= unemp <= 25):
        issues.append("Unemployment should be between 0% and 25%.")
    return issues

def make_input_df(store, holiday_flag, temp, fuel, cpi, unemp, week_date):
    # IMPORTANT: These column names must match what you trained the pipeline on
    return pd.DataFrame([{
        "Store": int(store),
        "Holiday_Flag": int(holiday_flag),
        "Temperature": float(temp),
        "Fuel_Price": float(fuel),
        "CPI": float(cpi),
        "Unemployment": float(unemp),
        "Date": pd.to_datetime(week_date),
    }])

def predict_one(store, holiday_flag, temp, fuel, cpi, unemp, week_date):
    X = make_input_df(store, holiday_flag, temp, fuel, cpi, unemp, week_date)
    pred = float(pipe.predict(X)[0])
    return pred, X

# ----------------------------
# Defaults + Example
# ----------------------------
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

def apply_preset(preset: dict):
    # SAFE: update session_state for widget keys via callback
    for k, v in preset.items():
        st.session_state[k] = v

def reset_all():
    apply_preset(DEFAULTS)
    st.session_state.history = []
    st.session_state.last_result = None

if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.markdown("### 📊 Walmart Dashboard")
    st.markdown('<div class="muted">Forecasting + insights</div>', unsafe_allow_html=True)
    page = st.radio(
        "Navigation",
        ["🏠 Overview", "🧮 Predict", "📈 Insights", "🧪 Model", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    st.markdown("#### Inputs")
    st.selectbox("Store", list(range(1, 46)), key="store")
    st.date_input("Week Date", key="week_date")
    st.selectbox("Week Type", ["Non-holiday Week", "Holiday Week"], key="holiday_label")
    st.slider("Temperature (°F)", -5.0, 105.0, step=0.1, key="temp")

    with st.expander("Advanced Inputs", expanded=False):
        st.slider("Fuel Price", 0.1, 5.0, step=0.01, key="fuel")
        st.slider("CPI", 0.1, 300.0, step=0.1, key="cpi")
        st.slider("Unemployment (%)", 0.0, 25.0, step=0.01, key="unemp")

    st.markdown("#### Actions")
    c1, c2 = st.columns(2)
    with c1:
        predict_btn = st.button("Predict", use_container_width=True)
    with c2:
        compare_btn = st.button("Compare", use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.button("Example", use_container_width=True, on_click=apply_preset, args=(EXAMPLE,))
    with c4:
        st.button("Reset", use_container_width=True, on_click=reset_all)

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

# Read inputs
store = st.session_state.store
holiday_flag = 1 if st.session_state.holiday_label == "Holiday Week" else 0
temp = st.session_state.temp
fuel = st.session_state.fuel
cpi = st.session_state.cpi
unemp = st.session_state.unemp
week_date = st.session_state.week_date

# ----------------------------
# Header (NO top navigation bar)
# ----------------------------
st.title("Walmart Weekly Sales Forecasting")
st.caption("Predict weekly sales using store + economic indicators. Includes what-if comparison.")

# ----------------------------
# Pages
# ----------------------------
if page == "🏠 Overview":
    hist = pd.DataFrame(st.session_state.history) if st.session_state.history else pd.DataFrame()
    last = st.session_state.last_result

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        latest = money(last["pred"]) if (last and last.get("mode") == "single") else "—"
        st.markdown(f'<div class="card"><b>Latest Forecast</b><br><span style="font-size:1.6rem;font-weight:900;">{latest}</span><br><span class="muted">Most recent</span></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="card"><b>Selected Store</b><br><span style="font-size:1.6rem;font-weight:900;">{store}</span><br><span class="muted">Current input</span></div>', unsafe_allow_html=True)
    with k3:
        wt = "Holiday" if holiday_flag else "Non-holiday"
        st.markdown(f'<div class="card"><b>Week Type</b><br><span style="font-size:1.6rem;font-weight:900;">{wt}</span><br><span class="muted">Seasonality</span></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="card"><b>Predictions</b><br><span style="font-size:1.6rem;font-weight:900;">{len(st.session_state.history)}</span><br><span class="muted">This session</span></div>', unsafe_allow_html=True)

    st.subheader("Forecast Trend")
    if len(hist) >= 2:
        hist["pred_sales"] = pd.to_numeric(hist["pred_sales"], errors="coerce")
        st.line_chart(hist["pred_sales"])
    else:
        st.info("Make at least 2 predictions to see a trend.")

    st.subheader("Recent Predictions")
    if len(hist):
        st.dataframe(hist.tail(8).iloc[::-1], use_container_width=True)
    else:
        st.info("No predictions yet. Go to Predict.")

elif page == "🧮 Predict":
    issues = validate_inputs(temp, fuel, cpi, unemp)
    if issues:
        for msg in issues:
            st.warning(msg)

    if predict_btn and not issues:
        pred, X_used = predict_one(store, holiday_flag, temp, fuel, cpi, unemp, week_date)
        st.session_state.last_result = {"mode": "single", "pred": pred, "X": X_used}
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

    if compare_btn and not issues:
        pred_non, X_non = predict_one(store, 0, temp, fuel, cpi, unemp, week_date)
        pred_hol, X_hol = predict_one(store, 1, temp, fuel, cpi, unemp, week_date)
        diff = pred_hol - pred_non
        pct = (diff / pred_non * 100) if pred_non != 0 else 0.0
        st.session_state.last_result = {
            "mode": "compare",
            "pred_non": pred_non, "pred_hol": pred_hol,
            "diff": diff, "pct": pct,
            "X_non": X_non, "X_hol": X_hol
        }
        st.info("What-if comparison done ✅ (only Holiday_Flag changed)")

    left, right = st.columns([0.7, 0.3], gap="large")

    with left:
        st.subheader("Output")
        res = st.session_state.last_result

        if not res:
            st.markdown('<div class="card">No output yet. Use the sidebar and click <b>Predict</b> or <b>Compare</b>.</div>', unsafe_allow_html=True)
        else:
            if res["mode"] == "single":
                a, b, c = st.columns(3)
                with a:
                    st.markdown(f'<div class="card"><b>Predicted Weekly Sales</b><br><span style="font-size:1.8rem;font-weight:900;color:#9bb6ff;">{money(res["pred"])}</span><br><span class="muted">Estimated revenue</span></div>', unsafe_allow_html=True)
                with b:
                    st.markdown(f'<div class="card"><b>Store</b><br><span style="font-size:1.8rem;font-weight:900;color:#9bb6ff;">{store}</span><br><span class="muted">Store selected</span></div>', unsafe_allow_html=True)
                with c:
                    wt = "Holiday" if holiday_flag else "Non-holiday"
                    st.markdown(f'<div class="card"><b>Week Type</b><br><span style="font-size:1.8rem;font-weight:900;color:#9bb6ff;">{wt}</span><br><span class="muted">Seasonality</span></div>', unsafe_allow_html=True)

                with st.expander("See model inputs used"):
                    st.dataframe(res["X"], use_container_width=True)

            if res["mode"] == "compare":
                a, b, c = st.columns(3)
                with a:
                    st.markdown(f'<div class="card"><b>Non-holiday</b><br><span style="font-size:1.8rem;font-weight:900;color:#9bb6ff;">{money(res["pred_non"])}</span><br><span class="muted">Holiday_Flag = 0</span></div>', unsafe_allow_html=True)
                with b:
                    st.markdown(f'<div class="card"><b>Holiday</b><br><span style="font-size:1.8rem;font-weight:900;color:#9bb6ff;">{money(res["pred_hol"])}</span><br><span class="muted">Holiday_Flag = 1</span></div>', unsafe_allow_html=True)
                with c:
                    st.markdown(f'<div class="card"><b>Uplift</b><br><span style="font-size:1.8rem;font-weight:900;color:#9bb6ff;">{money(res["diff"])}</span><br><span class="muted">{res["pct"]:.2f}% change</span></div>', unsafe_allow_html=True)

                with st.expander("See inputs used (debug)"):
                    st.write("Non-holiday:")
                    st.dataframe(res["X_non"], use_container_width=True)
                    st.write("Holiday:")
                    st.dataframe(res["X_hol"], use_container_width=True)

        st.subheader("Session History")
        if len(st.session_state.history) == 0:
            st.caption("No predictions yet.")
        else:
            hist_df = pd.DataFrame(st.session_state.history).iloc[::-1].reset_index(drop=True)
            with st.expander("Show history table", expanded=False):
                st.dataframe(hist_df, use_container_width=True)

            hist_df["pred_sales"] = pd.to_numeric(hist_df["pred_sales"], errors="coerce")
            hist_df = hist_df.dropna(subset=["pred_sales"])
            if len(hist_df) >= 2:
                st.subheader("Trend (Predicted Sales)")
                st.line_chart(hist_df["pred_sales"])

    with right:
        st.subheader("Guidance")
        st.markdown(
            """
            <div class="card">
              <b>Tips</b><br><br>
              <b>Validation:</b> prevents unrealistic inputs from skewing predictions.<br><br>
              <b>Compare:</b> shows business impact of holiday vs non-holiday weeks.<br><br>
              <span class="muted">If Store changes don’t change outputs, your saved model probably isn’t the full pipeline.</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="card" style="margin-top:1rem;">
              <b>Quick Summary</b><br><br>
              <span class="muted">Store:</span> <b>{store}</b><br>
              <span class="muted">Week Date:</span> <b>{week_date}</b><br>
              <span class="muted">Week Type:</span> <b>{"Holiday" if holiday_flag else "Non-holiday"}</b>
            </div>
            """,
            unsafe_allow_html=True
        )

elif page == "📈 Insights":
    st.subheader("Insights")
    st.markdown('<div class="card">Use this page to write business interpretation for your report: inventory, staffing, holiday uplift, economic context.</div>', unsafe_allow_html=True)

elif page == "🧪 Model":
    st.subheader("Model")
    st.markdown('<div class="card">Put your MAE/RMSE results + predicted vs actual plot screenshot here for documentation.</div>', unsafe_allow_html=True)

else:
    st.subheader("About")
    st.markdown('<div class="card">ML pipeline + Streamlit deployment for Walmart weekly sales forecasting.</div>', unsafe_allow_html=True)
