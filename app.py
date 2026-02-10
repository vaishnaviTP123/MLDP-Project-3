import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date
from pathlib import Path

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Walmart Sales Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CSS (clean, no topbar)
# ============================================================
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800;900&display=swap');
      * { font-family: 'Outfit', sans-serif; }

      [data-testid="stAppViewContainer"]{
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%);
        background-attachment: fixed;
      }

      .block-container{
        padding-top: 1.0rem;
        padding-bottom: 2.0rem;
        max-width: 1500px;
      }

      /* Sidebar */
      [data-testid="stSidebar"]{
        background: linear-gradient(180deg, rgba(13,18,36,0.98) 0%, rgba(8,12,26,0.99) 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
      }

      /* Cards */
      .card{
        background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        padding: 1.2rem 1.25rem;
        height: 100%;
      }

      .label{
        color: rgba(255,255,255,0.55);
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.35rem;
      }

      .value{
        font-size: 1.85rem;
        font-weight: 950;
        background: linear-gradient(135deg, #ffffff 0%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.2rem 0;
        line-height: 1.15;
      }

      .sub{
        color: rgba(255,255,255,0.62);
        font-size: 0.9rem;
        font-weight: 600;
      }

      .panel{
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        padding: 1.25rem;
      }

      /* Buttons */
      .stButton>button{
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.72rem 1.0rem;
        font-weight: 850;
      }
      .stDownloadButton>button{
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.72rem 1.0rem;
        font-weight: 850;
      }

      footer {visibility: hidden;}
      #MainMenu {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Safe loading paths (this fixes Streamlit Cloud path issues)
# ============================================================
APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "model.pkl"
COLS_PATH = APP_DIR / "columns.pkl"

@st.cache_resource
def load_artifacts():
    # show exact directory content if missing
    if not MODEL_PATH.exists() or not COLS_PATH.exists():
        st.error("❌ model.pkl / columns.pkl not found in the deployed folder.")
        st.write("Expected paths:")
        st.code(str(MODEL_PATH))
        st.code(str(COLS_PATH))

        st.write("Files currently in app folder:")
        try:
            st.code("\n".join([p.name for p in sorted(APP_DIR.iterdir())]))
        except Exception as e:
            st.write("Could not list directory:", e)

        st.info(
            "Fix: Make sure model.pkl and columns.pkl are pushed to GitHub (not ignored) "
            "and redeploy. If they are large, GitHub might not include them."
        )
        st.stop()

    model = joblib.load(MODEL_PATH)
    cols = joblib.load(COLS_PATH)

    if isinstance(cols, (pd.Index, np.ndarray)):
        cols = list(cols)

    return model, cols

model, feature_cols = load_artifacts()

# ============================================================
# Helpers
# ============================================================
def money(x) -> str:
    try:
        return "${:,.2f}".format(float(x))
    except Exception:
        return "—"

def validate_inputs(temp, fuel, cpi, unemp):
    issues = []
    if fuel <= 0: issues.append("Fuel Price must be > 0")
    if cpi <= 0: issues.append("CPI must be > 0")
    if not (-10 <= temp <= 120): issues.append("Temperature should be -10°F to 120°F")
    if not (0 <= unemp <= 25): issues.append("Unemployment should be 0% to 25%")
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
# Session defaults
# ============================================================
DEFAULTS = {
    "store": 1,
    "holiday_flag": 0,
    "temp": 60.0,
    "fuel": 3.50,
    "cpi": 180.0,
    "unemp": 7.50,
    "week_date": date(2012, 2, 10),
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

if "history" not in st.session_state:
    st.session_state.history = []

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

def history_df():
    if not st.session_state.history:
        return pd.DataFrame()
    df = pd.DataFrame(st.session_state.history)
    df["pred_sales"] = pd.to_numeric(df["pred_sales"], errors="coerce")
    df = df.dropna(subset=["pred_sales"])
    return df

# ============================================================
# Sidebar Navigation (no top nav)
# ============================================================
with st.sidebar:
    st.markdown("## 📊 Walmart Sales")
    st.caption("Sales forecasting + analytics")

    page = st.radio(
        "Go to",
        ["🏠 Overview", "🎯 Predict", "📈 Analytics", "ℹ️ About"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    if page == "🎯 Predict":
        st.markdown("### Inputs")

        st.session_state.store = st.selectbox(
            "Store Number",
            list(range(1, 46)),
            index=max(0, min(44, int(st.session_state.store) - 1)),
        )

        st.session_state.week_date = st.date_input("Week Date", value=st.session_state.week_date)

        holiday_label = st.selectbox(
            "Week Type",
            ["Non-Holiday", "Holiday"],
            index=1 if st.session_state.holiday_flag == 1 else 0,
        )
        st.session_state.holiday_flag = 1 if holiday_label == "Holiday" else 0

        st.session_state.temp = st.slider("Temperature (°F)", -10.0, 120.0, float(st.session_state.temp), 0.5)

        with st.expander("Economic Indicators", expanded=True):
            st.session_state.fuel = st.number_input(
                "Fuel Price ($)",
                min_value=0.01,
                max_value=10.0,
                value=float(max(st.session_state.fuel, 0.01)),
                step=0.01,
            )
            st.session_state.cpi = st.number_input(
                "CPI",
                min_value=0.01,
                max_value=500.0,
                value=float(max(st.session_state.cpi, 0.01)),
                step=0.1,
            )
            st.session_state.unemp = st.number_input(
                "Unemployment (%)",
                min_value=0.0,
                max_value=25.0,
                value=float(min(max(st.session_state.unemp, 0.0), 25.0)),
                step=0.01,
            )

        st.markdown("### Actions")
        predict_btn = st.button("Predict", use_container_width=True)
        compare_btn = st.button("Compare Holiday vs Non-Holiday", use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Reset", use_container_width=True):
                for k, v in DEFAULTS.items():
                    st.session_state[k] = v
                st.rerun()
        with c2:
            if st.session_state.history:
                df = pd.DataFrame(st.session_state.history)
                st.download_button(
                    "Export CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="forecast_history.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.button("Export CSV", disabled=True, use_container_width=True)

    else:
        predict_btn = False
        compare_btn = False

# ============================================================
# Pages
# ============================================================
hist = history_df()

if page == "🏠 Overview":
    st.title("Overview")
    st.caption("Quick snapshot of your session predictions.")

    c1, c2, c3, c4 = st.columns(4, gap="large")

    with c1:
        st.markdown(
            f"""
            <div class="card">
              <div class="label">Predictions</div>
              <div class="value">{len(hist)}</div>
              <div class="sub">This session</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        avg_val = hist["pred_sales"].mean() if len(hist) else 0
        st.markdown(
            f"""
            <div class="card">
              <div class="label">Average</div>
              <div class="value">{money(avg_val)}</div>
              <div class="sub">Mean forecast</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        max_val = hist["pred_sales"].max() if len(hist) else 0
        st.markdown(
            f"""
            <div class="card">
              <div class="label">Peak</div>
              <div class="value">{money(max_val)}</div>
              <div class="sub">Highest forecast</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        last_val = st.session_state.last_prediction if st.session_state.last_prediction is not None else 0
        st.markdown(
            f"""
            <div class="card">
              <div class="label">Latest</div>
              <div class="value">{money(last_val)}</div>
              <div class="sub">Most recent</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Trend")
    if len(hist) >= 2:
        st.line_chart(hist["pred_sales"].reset_index(drop=True), use_container_width=True)
    else:
        st.info("Make at least 2 predictions to see a trend chart.")

    st.markdown("### Recent History")
    if len(hist):
        show = hist.tail(10).iloc[::-1].reset_index(drop=True).copy()
        st.dataframe(show, use_container_width=True, height=360)
    else:
        st.info("No predictions yet. Go to Predict.")

elif page == "🎯 Predict":
    st.title("Sales Forecasting")
    st.caption("Predict weekly sales using store + economic indicators.")

    issues = validate_inputs(st.session_state.temp, st.session_state.fuel, st.session_state.cpi, st.session_state.unemp)
    if issues:
        st.warning("Please fix these before predicting:")
        for m in issues:
            st.write(f"• {m}")

    if predict_btn and not issues:
        pred, X_used = predict_one(
            st.session_state.store,
            st.session_state.holiday_flag,
            st.session_state.temp,
            st.session_state.fuel,
            st.session_state.cpi,
            st.session_state.unemp,
            st.session_state.week_date,
        )
        st.session_state.last_prediction = pred
        st.session_state.history.append(
            {
                "date": str(st.session_state.week_date),
                "store": int(st.session_state.store),
                "holiday_flag": int(st.session_state.holiday_flag),
                "temperature": float(st.session_state.temp),
                "fuel_price": float(st.session_state.fuel),
                "cpi": float(st.session_state.cpi),
                "unemployment": float(st.session_state.unemp),
                "pred_sales": float(pred),
            }
        )

        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            st.markdown(
                f"""<div class="card"><div class="label">Predicted Sales</div><div class="value">{money(pred)}</div><div class="sub">Weekly</div></div>""",
                unsafe_allow_html=True,
            )
        with c2:
            wt = "Holiday" if st.session_state.holiday_flag else "Non-Holiday"
            st.markdown(
                f"""<div class="card"><div class="label">Week Type</div><div class="value" style="font-size:1.4rem;">{wt}</div><div class="sub">{st.session_state.week_date}</div></div>""",
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f"""<div class="card"><div class="label">Store</div><div class="value">{st.session_state.store}</div><div class="sub">Selected</div></div>""",
                unsafe_allow_html=True,
            )

        with st.expander("Show model features used"):
            st.dataframe(X_used, use_container_width=True)

    if compare_btn and not issues:
        pred_non, _ = predict_one(
            st.session_state.store, 0,
            st.session_state.temp, st.session_state.fuel,
            st.session_state.cpi, st.session_state.unemp,
            st.session_state.week_date
        )
        pred_hol, _ = predict_one(
            st.session_state.store, 1,
            st.session_state.temp, st.session_state.fuel,
            st.session_state.cpi, st.session_state.unemp,
            st.session_state.week_date
        )
        diff = pred_hol - pred_non
        pct = (diff / pred_non * 100) if pred_non != 0 else 0.0

        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            st.markdown(
                f"""<div class="card"><div class="label">Non-Holiday</div><div class="value">{money(pred_non)}</div><div class="sub">Baseline</div></div>""",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""<div class="card"><div class="label">Holiday</div><div class="value">{money(pred_hol)}</div><div class="sub">Holiday_Flag=1</div></div>""",
                unsafe_allow_html=True,
            )
        with c3:
            sign = "+" if diff > 0 else ""
            st.markdown(
                f"""<div class="card"><div class="label">Impact</div><div class="value">{money(abs(diff))}</div><div class="sub">{sign}{pct:.1f}%</div></div>""",
                unsafe_allow_html=True,
            )

        comp = pd.DataFrame({"Scenario": ["Non-Holiday", "Holiday"], "Sales": [pred_non, pred_hol]})
        st.bar_chart(comp.set_index("Scenario"), use_container_width=True)

elif page == "📈 Analytics":
    st.title("Analytics")
    st.caption("Session analytics from your predictions.")

    if len(hist) >= 3:
        st.markdown("### Stats")
        c1, c2, c3, c4, c5 = st.columns(5)
        stats = [
            ("Mean", hist["pred_sales"].mean()),
            ("Median", hist["pred_sales"].median()),
            ("Std", hist["pred_sales"].std()),
            ("Min", hist["pred_sales"].min()),
            ("Max", hist["pred_sales"].max()),
        ]
        for col, (lab, val) in zip([c1, c2, c3, c4, c5], stats):
            with col:
                st.markdown(
                    f"""<div class="card"><div class="label">{lab}</div><div class="value" style="font-size:1.25rem;">{money(val)}</div></div>""",
                    unsafe_allow_html=True,
                )

        st.markdown("### Trend")
        st.line_chart(hist["pred_sales"].reset_index(drop=True), use_container_width=True)

        st.markdown("### Full History")
        st.dataframe(hist.iloc[::-1].reset_index(drop=True), use_container_width=True, height=420)
    else:
        st.info("Make at least 3 predictions to unlock analytics.")

else:
    st.title("About")
    st.markdown(
        """
        <div class="panel">
          <b>Goal:</b> Predict weekly Walmart sales for planning (inventory, staffing, promotions).<br><br>
          <b>Files required in GitHub for Streamlit Cloud:</b> <code>model.pkl</code>, <code>columns.pkl</code>, <code>app.py</code>, <code>requirements.txt</code>.
        </div>
        """,
        unsafe_allow_html=True,
    )
