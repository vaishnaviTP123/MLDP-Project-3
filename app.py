import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Walmart Sales Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# CSS (kept your look, cleaned slightly)
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
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1600px;
      }

      footer {visibility: hidden;}
      #MainMenu {visibility: hidden;}

      .main-header{
        background: linear-gradient(135deg, rgba(99,102,241,0.10) 0%, rgba(168,85,247,0.08) 100%);
        border: 1px solid rgba(139,92,246,0.20);
        border-radius: 24px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(18px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.30);
      }
      .header-title{
        font-size: 2.0rem;
        font-weight: 900;
        background: linear-gradient(135deg, #a78bfa 0%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
        margin: 0;
      }
      .header-subtitle{
        color: rgba(255,255,255,0.62);
        font-size: 0.95rem;
        margin-top: 0.3rem;
        font-weight: 400;
      }
      .status-badge{
        display:inline-block;
        padding: 6px 16px;
        background: rgba(16,185,129,0.15);
        border: 1px solid rgba(16,185,129,0.30);
        border-radius: 20px;
        color: #10b981;
        font-size: 0.85rem;
        font-weight: 800;
      }

      .section-title{
        font-size: 1.25rem;
        font-weight: 900;
        color: #ffffff;
        margin: 0.2rem 0 1.0rem 0;
        display:flex;
        align-items:center;
        gap:10px;
      }
      .section-title::before{
        content:'';
        width: 4px;
        height: 22px;
        background: linear-gradient(180deg, #8b5cf6, #ec4899);
        border-radius: 2px;
      }

      .input-section{
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 20px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
      }

      .metric-card{
        background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 20px;
        padding: 1.3rem;
        height: 100%;
      }
      .metric-label{
        color: rgba(255,255,255,0.55);
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.4rem;
      }
      .metric-value{
        font-size: 1.75rem;
        font-weight: 950;
        background: linear-gradient(135deg, #ffffff 0%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.2rem 0;
        line-height: 1.15;
      }
      .metric-change{
        color: rgba(255,255,255,0.62);
        font-size: 0.85rem;
        font-weight: 600;
      }

      .metric-change.positive{ color: #10b981; }
      .metric-change.negative{ color: #ef4444; }

      .stButton>button{
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.1rem;
        font-weight: 800;
      }
      .stDownloadButton>button{
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.1rem;
        font-weight: 800;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Load model + feature columns (safe)
# ============================================================
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("model.pkl")
        cols = joblib.load("columns.pkl")
        # ensure it's a list
        if isinstance(cols, (pd.Index, np.ndarray)):
            cols = list(cols)
        return model, cols
    except Exception as e:
        st.error("⚠️ Model files not found / cannot load. Make sure model.pkl and columns.pkl are in the same folder as app.py")
        st.exception(e)
        st.stop()

model, feature_cols = load_artifacts()

# ============================================================
# Helpers
# ============================================================
def money(x: float) -> str:
    try:
        return "${:,.2f}".format(float(x))
    except Exception:
        return "—"

def validate_inputs(temp, fuel, cpi, unemp):
    issues = []
    if fuel <= 0:
        issues.append("⚠️ Fuel Price must be > 0")
    if cpi <= 0:
        issues.append("⚠️ CPI must be > 0")
    if not (-10 <= temp <= 120):
        issues.append("⚠️ Temperature should be between -10°F and 120°F")
    if not (0 <= unemp <= 25):
        issues.append("⚠️ Unemployment should be between 0% and 25%")
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
# Session init (ONE set of keys only)
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

# ============================================================
# Header
# ============================================================
st.markdown(
    """
    <div class="main-header">
        <div style="display:flex; justify-content:space-between; align-items:center; gap:16px;">
            <div>
                <div class="header-title">📊 Walmart Sales Intelligence Platform</div>
                <div class="header-subtitle">Advanced predictive analytics for retail operations planning</div>
            </div>
            <div class="status-badge">● ONLINE</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Tabs
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs(["📈 Dashboard", "🎯 Forecast Engine", "📊 Analytics", "⚙️ About"])

# ============================================================
# Prepare history DF safely
# ============================================================
def get_history_df():
    if len(st.session_state.history) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(st.session_state.history)
    if "pred_sales" in df.columns:
        df["pred_sales"] = pd.to_numeric(df["pred_sales"], errors="coerce")
        df = df.dropna(subset=["pred_sales"])
    return df

# ============================================================
# TAB 1: Dashboard
# ============================================================
with tab1:
    st.markdown('<div class="section-title">Key Performance Indicators</div>', unsafe_allow_html=True)

    hist = get_history_df()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">Total Forecasts</div>
              <div class="metric-value">{len(hist)}</div>
              <div class="metric-change">This session</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        avg_forecast = hist["pred_sales"].mean() if len(hist) else 0
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">Avg Forecast</div>
              <div class="metric-value">{money(avg_forecast)}</div>
              <div class="metric-change">Mean prediction</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        max_forecast = hist["pred_sales"].max() if len(hist) else 0
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">Peak Forecast</div>
              <div class="metric-value">{money(max_forecast)}</div>
              <div class="metric-change">Maximum value</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        last_value = st.session_state.last_prediction if st.session_state.last_prediction is not None else 0
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">Latest Forecast</div>
              <div class="metric-value">{money(last_value)}</div>
              <div class="metric-change">Most recent</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")

    left, right = st.columns([2, 1], gap="large")

    with left:
        st.markdown('<div class="section-title">Forecast Trend Analysis</div>', unsafe_allow_html=True)
        if len(hist) >= 2:
            chart_df = hist.copy().reset_index(drop=True)
            st.line_chart(chart_df["pred_sales"], use_container_width=True)
        else:
            st.info("📊 Run at least 2 predictions to see the trend chart.")

    with right:
        st.markdown('<div class="section-title">Session Summary</div>', unsafe_allow_html=True)
        if len(hist) > 0:
            std_val = hist["pred_sales"].std()
            std_val = 0 if pd.isna(std_val) else std_val
            st.markdown(
                f"""
                <div class="input-section">
                  <div style="margin-bottom: 1rem;">
                    <div class="metric-label">Predictions Made</div>
                    <div class="metric-value" style="font-size:1.4rem;">{len(hist)}</div>
                  </div>
                  <div style="margin-bottom: 1rem;">
                    <div class="metric-label">Std Dev</div>
                    <div class="metric-value" style="font-size:1.25rem;">{money(std_val)}</div>
                  </div>
                  <div>
                    <div class="metric-label">Range</div>
                    <div class="metric-change">{money(hist["pred_sales"].min())} - {money(hist["pred_sales"].max())}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("🎯 No predictions yet. Go to Forecast Engine tab.")

    if len(hist):
        st.markdown('<div class="section-title">Recent Forecast History</div>', unsafe_allow_html=True)
        display_hist = hist.tail(10).iloc[::-1].reset_index(drop=True).copy()
        display_hist["pred_sales"] = display_hist["pred_sales"].apply(money)
        st.dataframe(display_hist, use_container_width=True, height=360)

# ============================================================
# TAB 2: Forecast Engine
# ============================================================
with tab2:
    st.markdown('<div class="section-title">Forecast Configuration</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("#### 🏪 Store & Date Selection")

        a, b, c = st.columns(3)
        with a:
            st.session_state.store = st.selectbox(
                "Store Number",
                list(range(1, 46)),
                index=max(0, min(44, int(st.session_state.store) - 1)),
                key="store_widget",
            )

        with b:
            st.session_state.week_date = st.date_input(
                "Week Date",
                value=st.session_state.week_date,
                key="date_widget",
            )

        with c:
            holiday_label = st.selectbox(
                "Week Type",
                ["Non-Holiday", "Holiday"],
                index=1 if int(st.session_state.holiday_flag) == 1 else 0,
                key="holiday_widget",
            )
            st.session_state.holiday_flag = 1 if holiday_label == "Holiday" else 0

        st.markdown("#### 🌡️ Environmental Factors")
        st.session_state.temp = st.slider(
            "Temperature (°F)",
            -10.0,
            120.0,
            value=float(st.session_state.temp),
            step=0.5,
            key="temp_widget",
        )

        st.markdown("#### 💰 Economic Indicators")
        e1, e2, e3 = st.columns(3)

        with e1:
            st.session_state.fuel = st.number_input(
                "Fuel Price ($)",
                min_value=0.01,
                max_value=10.0,
                value=float(max(st.session_state.fuel, 0.01)),
                step=0.01,
                key="fuel_widget",
            )
        with e2:
            st.session_state.cpi = st.number_input(
                "CPI",
                min_value=0.01,
                max_value=500.0,
                value=float(max(st.session_state.cpi, 0.01)),
                step=0.1,
                key="cpi_widget",
            )
        with e3:
            st.session_state.unemp = st.number_input(
                "Unemployment (%)",
                min_value=0.0,
                max_value=25.0,
                value=float(min(max(st.session_state.unemp, 0.0), 25.0)),
                step=0.01,
                key="unemp_widget",
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # Buttons
        b1, b2, b3, b4 = st.columns(4, gap="small")
        with b1:
            predict_btn = st.button("🎯 Generate Forecast", use_container_width=True)
        with b2:
            compare_btn = st.button("📊 Compare Scenarios", use_container_width=True)
        with b3:
            reset_btn = st.button("🔄 Reset Inputs", use_container_width=True)
        with b4:
            export_btn = st.button("💾 Export History", use_container_width=True)

        if reset_btn:
            for k, v in DEFAULTS.items():
                st.session_state[k] = v
            st.success("✅ Inputs reset to defaults")
            st.rerun()

        if export_btn:
            if len(st.session_state.history):
                hist_df = pd.DataFrame(st.session_state.history)
                csv = hist_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV",
                    data=csv,
                    file_name="forecast_history.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.info("No history to export yet.")

    with col2:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("#### 📋 Current Configuration")
        st.markdown(
            f"""
            **Store:** {st.session_state.store}  
            **Date:** {st.session_state.week_date}  
            **Type:** {'Holiday' if st.session_state.holiday_flag else 'Non-Holiday'}  
            **Temperature:** {st.session_state.temp}°F  
            **Fuel Price:** ${st.session_state.fuel:.2f}  
            **CPI:** {st.session_state.cpi:.1f}  
            **Unemployment:** {st.session_state.unemp:.2f}%
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("#### ℹ️ Quick Guide")
        st.markdown(
            """
            **Generate Forecast:** Predict sales for current inputs  
            **Compare Scenarios:** Holiday vs non-holiday impact  
            **Reset:** Restore default values  
            **Export:** Download your prediction history
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Validation
    issues = validate_inputs(
        st.session_state.temp,
        st.session_state.fuel,
        st.session_state.cpi,
        st.session_state.unemp,
    )
    if issues:
        for msg in issues:
            st.warning(msg)

    # Predict
    if predict_btn:
        if issues:
            st.error("Fix the warnings above, then try again.")
        else:
            try:
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

                st.success("✅ Forecast generated successfully!")

                st.markdown('<div class="section-title">Forecast Result</div>', unsafe_allow_html=True)
                r1, r2, r3 = st.columns(3, gap="large")

                with r1:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                          <div class="metric-label">Predicted Weekly Sales</div>
                          <div class="metric-value">{money(pred)}</div>
                          <div class="metric-change">Store {st.session_state.store}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with r2:
                    wt = "Holiday Week" if st.session_state.holiday_flag else "Non-Holiday Week"
                    st.markdown(
                        f"""
                        <div class="metric-card">
                          <div class="metric-label">Week Type</div>
                          <div class="metric-value" style="font-size:1.25rem;">{wt}</div>
                          <div class="metric-change">{st.session_state.week_date}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with r3:
                    st.markdown(
                        """
                        <div class="metric-card">
                          <div class="metric-label">Note</div>
                          <div class="metric-value" style="font-size:1.25rem;">Model Output</div>
                          <div class="metric-change">Use Compare to show interactivity</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with st.expander("🔍 View Model Features Used"):
                    st.dataframe(X_used, use_container_width=True)

            except Exception as e:
                st.error("❌ Prediction failed.")
                st.exception(e)

    # Compare
    if compare_btn:
        if issues:
            st.error("Fix the warnings above, then try again.")
        else:
            try:
                pred_non, X_non = predict_one(
                    st.session_state.store,
                    0,
                    st.session_state.temp,
                    st.session_state.fuel,
                    st.session_state.cpi,
                    st.session_state.unemp,
                    st.session_state.week_date,
                )
                pred_hol, X_hol = predict_one(
                    st.session_state.store,
                    1,
                    st.session_state.temp,
                    st.session_state.fuel,
                    st.session_state.cpi,
                    st.session_state.unemp,
                    st.session_state.week_date,
                )

                diff = pred_hol - pred_non
                pct = (diff / pred_non * 100) if pred_non != 0 else 0.0

                st.success("✅ Scenario comparison completed!")

                st.markdown('<div class="section-title">Holiday Impact Analysis</div>', unsafe_allow_html=True)
                r1, r2, r3 = st.columns(3, gap="large")

                with r1:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                          <div class="metric-label">Non-Holiday Week</div>
                          <div class="metric-value">{money(pred_non)}</div>
                          <div class="metric-change">Baseline</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with r2:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                          <div class="metric-label">Holiday Week</div>
                          <div class="metric-value">{money(pred_hol)}</div>
                          <div class="metric-change">Holiday_Flag = 1</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with r3:
                    cls = "positive" if diff > 0 else "negative"
                    sign = "+" if diff > 0 else ""
                    st.markdown(
                        f"""
                        <div class="metric-card">
                          <div class="metric-label">Impact</div>
                          <div class="metric-value">{money(abs(diff))}</div>
                          <div class="metric-change {cls}">{sign}{pct:.1f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                comp_df = pd.DataFrame({"Scenario": ["Non-Holiday", "Holiday"], "Sales": [pred_non, pred_hol]})
                st.bar_chart(comp_df.set_index("Scenario"), use_container_width=True)

                with st.expander("🔍 View Feature Differences"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Non-Holiday Features**")
                        st.dataframe(X_non, use_container_width=True)
                    with c2:
                        st.markdown("**Holiday Features**")
                        st.dataframe(X_hol, use_container_width=True)

            except Exception as e:
                st.error("❌ Comparison failed.")
                st.exception(e)

# ============================================================
# TAB 3: Analytics
# ============================================================
with tab3:
    st.markdown('<div class="section-title">Advanced Analytics & Insights</div>', unsafe_allow_html=True)

    hist_df = get_history_df()

    if len(hist_df) >= 3:
        # stats
        cols = st.columns(5, gap="small")
        stats = [
            ("Mean", hist_df["pred_sales"].mean()),
            ("Median", hist_df["pred_sales"].median()),
            ("Std Dev", hist_df["pred_sales"].std()),
            ("Min", hist_df["pred_sales"].min()),
            ("Max", hist_df["pred_sales"].max()),
        ]
        for i, (label, val) in enumerate(stats):
            val = 0 if pd.isna(val) else val
            with cols[i]:
                st.markdown(
                    f"""
                    <div class="metric-card">
                      <div class="metric-label">{label}</div>
                      <div class="metric-value" style="font-size:1.25rem;">{money(val)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("")

        left, right = st.columns(2, gap="large")
        with left:
            st.markdown('<div class="section-title">Sales Over Time</div>', unsafe_allow_html=True)
            st.line_chart(hist_df["pred_sales"].reset_index(drop=True), use_container_width=True)

        with right:
            st.markdown('<div class="section-title">Store Comparison</div>', unsafe_allow_html=True)
            if "store" in hist_df.columns and hist_df["store"].nunique() > 1:
                store_avg = hist_df.groupby("store")["pred_sales"].mean().reset_index()
                store_avg = store_avg.sort_values("pred_sales", ascending=False)
                st.bar_chart(store_avg.set_index("store"), use_container_width=True)
            else:
                st.info("Make predictions for multiple stores to see store comparison.")

        st.markdown('<div class="section-title">Full History</div>', unsafe_allow_html=True)
        show_df = hist_df.copy().iloc[::-1].reset_index(drop=True)
        st.dataframe(show_df, use_container_width=True, height=420)

    else:
        st.info("📊 Generate at least 3 predictions to unlock analytics.")
        st.markdown(
            """
            <div class="input-section">
              <b>What you'll get here:</b><br><br>
              • Mean/Median/Std Dev and Min/Max<br>
              • Trend chart over time<br>
              • Store ranking chart (if you predict multiple stores)<br>
              • Full history table
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================
# TAB 4: About
# ============================================================
with tab4:
    st.markdown('<div class="section-title">About This Platform</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown(
            """
            <div class="input-section">
              <h3>🎯 Purpose</h3>
              <p>Predict weekly Walmart sales for operational planning (inventory, staffing, and promotions).</p>

              <h3>🔧 Technology</h3>
              <ul>
                <li><b>UI:</b> Streamlit</li>
                <li><b>Model:</b> scikit-learn (loaded from <code>model.pkl</code>)</li>
                <li><b>Features:</b> loaded from <code>columns.pkl</code></li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="input-section">
              <h3>✅ Features</h3>
              <ul>
                <li>Single forecast prediction</li>
                <li>Holiday vs non-holiday comparison</li>
                <li>Session history tracking</li>
                <li>Analytics summary + charts</li>
                <li>Export prediction history</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
