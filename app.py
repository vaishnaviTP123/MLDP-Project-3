import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date, datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Walmart Sales Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# Enhanced Dashboard CSS with modern design
# ============================================================
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');
      
      /* ====== Background & Base ====== */
      * { font-family: 'Outfit', sans-serif; }
      
      [data-testid="stAppViewContainer"]{
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%);
        background-attachment: fixed;
      }
      
      .block-container { 
        padding-top: 1rem; 
        padding-bottom: 2rem; 
        max-width: 1600px; 
      }
      
      /* ====== Header Bar ====== */
      .main-header {
        background: linear-gradient(135deg, rgba(99,102,241,0.1) 0%, rgba(168,85,247,0.08) 100%);
        border: 1px solid rgba(139,92,246,0.2);
        border-radius: 24px;
        padding: 1.8rem 2rem;
        margin-bottom: 2rem;
        backdrop-filter: blur(20px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
      }
      
      .header-title {
        font-size: 2.2rem;
        font-weight: 900;
        background: linear-gradient(135deg, #a78bfa 0%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
        margin: 0;
      }
      
      .header-subtitle {
        color: rgba(255,255,255,0.6);
        font-size: 0.95rem;
        margin-top: 0.3rem;
        font-weight: 400;
      }
      
      /* ====== Tabs Styling ====== */
      .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.03);
        border-radius: 16px;
        padding: 8px;
        border: 1px solid rgba(255,255,255,0.08);
      }
      
      .stTabs [data-baseweb="tab"] {
        height: 50px;
        background: transparent;
        border-radius: 12px;
        color: rgba(255,255,255,0.5);
        font-weight: 600;
        font-size: 0.95rem;
        border: none;
        padding: 0 24px;
        transition: all 0.3s ease;
      }
      
      .stTabs [data-baseweb="tab"]:hover {
        background: rgba(139,92,246,0.1);
        color: rgba(255,255,255,0.8);
      }
      
      .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(139,92,246,0.2) 0%, rgba(236,72,153,0.2) 100%);
        color: #ffffff !important;
        border: 1px solid rgba(139,92,246,0.3);
      }
      
      /* ====== Cards ====== */
      .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 1.5rem;
        height: 100%;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
      }
      
      .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #8b5cf6, #ec4899);
        opacity: 0;
        transition: opacity 0.3s ease;
      }
      
      .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(139,92,246,0.3);
        box-shadow: 0 12px 40px rgba(139,92,246,0.2);
      }
      
      .metric-card:hover::before {
        opacity: 1;
      }
      
      .metric-label {
        color: rgba(255,255,255,0.5);
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
      }
      
      .metric-value {
        font-size: 2rem;
        font-weight: 900;
        background: linear-gradient(135deg, #ffffff 0%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.3rem 0;
      }
      
      .metric-change {
        color: rgba(255,255,255,0.6);
        font-size: 0.85rem;
        font-weight: 500;
      }
      
      .metric-change.positive {
        color: #10b981;
      }
      
      .metric-change.negative {
        color: #ef4444;
      }
      
      /* ====== Input Sections ====== */
      .input-section {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 1.8rem;
        margin-bottom: 1.5rem;
      }
      
      .section-title {
        font-size: 1.3rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 10px;
      }
      
      .section-title::before {
        content: '';
        width: 4px;
        height: 24px;
        background: linear-gradient(180deg, #8b5cf6, #ec4899);
        border-radius: 2px;
      }
      
      /* ====== Buttons ====== */
      .stButton>button {
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 700;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(139,92,246,0.3);
      }
      
      .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(139,92,246,0.4);
        background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%);
      }
      
      .stButton>button:active {
        transform: translateY(0);
      }
      
      /* ====== Inputs ====== */
      .stNumberInput input, .stDateInput input, .stSelectbox select {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 12px !important;
        color: white !important;
        font-weight: 500;
        padding: 0.75rem !important;
      }
      
      .stNumberInput input:focus, .stDateInput input:focus, .stSelectbox select:focus {
        border-color: rgba(139,92,246,0.5) !important;
        box-shadow: 0 0 0 3px rgba(139,92,246,0.1) !important;
      }
      
      /* ====== Slider ====== */
      .stSlider [data-baseweb="slider"] {
        margin-top: 1rem;
      }
      
      .stSlider [data-testid="stTickBar"] {
        background: rgba(255,255,255,0.1);
      }
      
      /* ====== Charts ====== */
      .chart-container {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 1.5rem;
        margin-top: 1rem;
      }
      
      /* ====== Status Badge ====== */
      .status-badge {
        display: inline-block;
        padding: 6px 16px;
        background: rgba(16,185,129,0.15);
        border: 1px solid rgba(16,185,129,0.3);
        border-radius: 20px;
        color: #10b981;
        font-size: 0.85rem;
        font-weight: 700;
        margin-left: 1rem;
      }
      
      /* ====== Alert Boxes ====== */
      .stAlert {
        border-radius: 16px;
        border-left: 4px solid;
        background: rgba(255,255,255,0.05);
      }
      
      /* Hide Streamlit branding */
      footer {visibility: hidden;}
      #MainMenu {visibility: hidden;}
      
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Load model + feature columns
# ============================================================
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("model.pkl")
        cols = joblib.load("columns.pkl")
        return model, cols
    except:
        st.error("⚠️ Model files not found. Please ensure model.pkl and columns.pkl are in the app directory.")
        st.stop()

model, feature_cols = load_artifacts()

# ============================================================
# Helper Functions
# ============================================================
def money(x: float) -> str:
    return "${:,.2f}".format(float(x))

def validate_inputs(temp, fuel, cpi, unemp):
    issues = []
    if fuel <= 0:
        issues.append("⚠️ Fuel Price must be greater than 0")
    if cpi <= 0:
        issues.append("⚠️ CPI must be greater than 0")
    if not (-10 <= temp <= 120):
        issues.append("⚠️ Temperature should be between -10°F and 120°F")
    if not (0 <= unemp <= 25):
        issues.append("⚠️ Unemployment rate should be between 0% and 25%")
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

def create_gauge_chart(value, title, max_value=None):
    """Create a gauge chart for metrics"""
    if max_value is None:
        max_value = value * 1.5
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 16, 'color': 'white'}},
        number={'prefix': "$", 'font': {'size': 32, 'color': 'white'}},
        gauge={
            'axis': {'range': [None, max_value], 'tickcolor': "white"},
            'bar': {'color': "#8b5cf6"},
            'bgcolor': "rgba(255,255,255,0.1)",
            'borderwidth': 2,
            'bordercolor': "rgba(255,255,255,0.2)",
            'steps': [
                {'range': [0, max_value*0.33], 'color': 'rgba(239,68,68,0.2)'},
                {'range': [max_value*0.33, max_value*0.66], 'color': 'rgba(251,191,36,0.2)'},
                {'range': [max_value*0.66, max_value], 'color': 'rgba(16,185,129,0.2)'}
            ],
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white', 'family': 'Outfit'},
        height=250,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def create_trend_chart(df):
    """Create an advanced trend chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        y=df['pred_sales'],
        mode='lines+markers',
        name='Predicted Sales',
        line=dict(color='#8b5cf6', width=3),
        marker=dict(size=8, color='#ec4899', line=dict(color='white', width=2)),
        fill='tozeroy',
        fillcolor='rgba(139,92,246,0.1)'
    ))
    
    fig.update_layout(
        title="Sales Forecast Trend",
        title_font=dict(size=20, color='white', family='Outfit'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Outfit'),
        hovermode='x unified',
        showlegend=False,
        height=350,
        margin=dict(l=20, r=20, t=60, b=40),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            title="Prediction #",
            title_font=dict(size=12, color='rgba(255,255,255,0.6)')
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            title="Sales ($)",
            title_font=dict(size=12, color='rgba(255,255,255,0.6)'),
            tickformat='$,.0f'
        )
    )
    
    return fig

def create_comparison_chart(non_holiday, holiday):
    """Create a comparison chart for holiday vs non-holiday"""
    fig = go.Figure()
    
    categories = ['Non-Holiday', 'Holiday']
    values = [non_holiday, holiday]
    colors = ['#6366f1', '#ec4899']
    
    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker=dict(
            color=colors,
            line=dict(color='white', width=2)
        ),
        text=[money(v) for v in values],
        textposition='outside',
        textfont=dict(size=16, color='white', family='Outfit', weight='bold')
    ))
    
    fig.update_layout(
        title="Holiday Impact Analysis",
        title_font=dict(size=20, color='white', family='Outfit'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Outfit'),
        height=400,
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=40),
        xaxis=dict(
            showgrid=False,
            title_font=dict(size=12, color='rgba(255,255,255,0.6)')
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            title="Predicted Sales ($)",
            title_font=dict(size=12, color='rgba(255,255,255,0.6)'),
            tickformat='$,.0f'
        )
    )
    
    return fig

# ============================================================
# Session State Initialization
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
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div class="header-title">📊 Walmart Sales Intelligence Platform</div>
                <div class="header-subtitle">Advanced predictive analytics for retail operations planning</div>
            </div>
            <div class="status-badge">● ONLINE</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# Main Tabs
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs(["📈 Dashboard", "🎯 Forecast Engine", "📊 Analytics", "⚙️ About"])

# ============================================================
# TAB 1: DASHBOARD
# ============================================================
with tab1:
    # Key Metrics Row
    st.markdown('<div class="section-title">Key Performance Indicators</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    hist = pd.DataFrame(st.session_state.history) if len(st.session_state.history) else pd.DataFrame()
    if len(hist):
        hist["pred_sales"] = pd.to_numeric(hist["pred_sales"], errors="coerce")
        hist = hist.dropna(subset=["pred_sales"])
    
    with col1:
        total_predictions = len(st.session_state.history)
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Total Forecasts</div>
                <div class="metric-value">{total_predictions}</div>
                <div class="metric-change">This session</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        avg_forecast = hist["pred_sales"].mean() if len(hist) else 0
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Avg Forecast</div>
                <div class="metric-value">{money(avg_forecast)}</div>
                <div class="metric-change">Mean prediction</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col3:
        max_forecast = hist["pred_sales"].max() if len(hist) else 0
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Peak Forecast</div>
                <div class="metric-value">{money(max_forecast)}</div>
                <div class="metric-change">Maximum value</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col4:
        last_pred = st.session_state.last_prediction
        last_value = last_pred if last_pred else 0
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Latest Forecast</div>
                <div class="metric-value">{money(last_value)}</div>
                <div class="metric-change">Most recent</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts Row
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="section-title">Forecast Trend Analysis</div>', unsafe_allow_html=True)
        if len(hist) >= 2:
            st.plotly_chart(create_trend_chart(hist), use_container_width=True)
        else:
            st.info("📊 Run at least 2 predictions to see trend visualization")
    
    with col2:
        st.markdown('<div class="section-title">Latest Prediction</div>', unsafe_allow_html=True)
        if last_pred:
            st.plotly_chart(create_gauge_chart(last_pred, "Weekly Sales"), use_container_width=True)
        else:
            st.info("🎯 No predictions yet. Go to Forecast Engine to start.")
    
    # Recent History Table
    if len(hist):
        st.markdown('<div class="section-title">Recent Forecast History</div>', unsafe_allow_html=True)
        display_hist = hist.tail(10).iloc[::-1].reset_index(drop=True)
        display_hist['pred_sales'] = display_hist['pred_sales'].apply(money)
        st.dataframe(display_hist, use_container_width=True, height=400)

# ============================================================
# TAB 2: FORECAST ENGINE
# ============================================================
with tab2:
    # Input Section
    st.markdown('<div class="section-title">Forecast Configuration</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("#### 🏪 Store & Date Selection")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            store = st.selectbox("Store Number", list(range(1, 46)), 
                               index=st.session_state.store - 1,
                               key="store_input")
            st.session_state.store = store
        
        with c2:
            week_date = st.date_input("Week Date", 
                                     value=st.session_state.week_date,
                                     key="date_input")
            st.session_state.week_date = week_date
        
        with c3:
            holiday = st.selectbox("Week Type", 
                                  ["Non-Holiday", "Holiday"],
                                  index=st.session_state.holiday_flag,
                                  key="holiday_input")
            st.session_state.holiday_flag = 1 if holiday == "Holiday" else 0
        
        st.markdown("#### 🌡️ Environmental Factors")
        temp = st.slider("Temperature (°F)", -5.0, 105.0, 
                        value=st.session_state.temp, 
                        step=0.5,
                        key="temp_input")
        st.session_state.temp = temp
        
        st.markdown("#### 💰 Economic Indicators")
        ec1, ec2, ec3 = st.columns(3)
        
        with ec1:
            fuel = st.number_input("Fuel Price ($)", min_value=0.0, max_value=10.0, 
                                  value=st.session_state.fuel, step=0.01,
                                  key="fuel_input")
            st.session_state.fuel = fuel
        
        with ec2:
            cpi = st.number_input("CPI", min_value=0.0, max_value=500.0, 
                                value=st.session_state.cpi, step=0.1,
                                key="cpi_input")
            st.session_state.cpi = cpi
        
        with ec3:
            unemp = st.number_input("Unemployment (%)", min_value=0.0, max_value=30.0, 
                                   value=st.session_state.unemp, step=0.01,
                                   key="unemp_input")
            st.session_state.unemp = unemp
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Action Buttons
        st.markdown("<br>", unsafe_allow_html=True)
        b1, b2, b3, b4 = st.columns(4)
        
        with b1:
            predict_btn = st.button("🎯 Generate Forecast", use_container_width=True)
        
        with b2:
            compare_btn = st.button("📊 Compare Scenarios", use_container_width=True)
        
        with b3:
            if st.button("🔄 Reset Inputs", use_container_width=True):
                for k, v in DEFAULTS.items():
                    st.session_state[k] = v
                st.success("✅ Inputs reset to defaults")
                st.rerun()
        
        with b4:
            if len(st.session_state.history):
                hist_df = pd.DataFrame(st.session_state.history)
                csv = hist_df.to_csv(index=False).encode('utf-8')
                st.download_button("💾 Export History", data=csv, 
                                 file_name="forecast_history.csv",
                                 mime="text/csv",
                                 use_container_width=True)
    
    with col2:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("#### 📋 Current Configuration")
        st.markdown(f"""
        **Store:** {st.session_state.store}  
        **Date:** {st.session_state.week_date}  
        **Type:** {'Holiday' if st.session_state.holiday_flag else 'Non-Holiday'}  
        **Temperature:** {st.session_state.temp}°F  
        **Fuel Price:** ${st.session_state.fuel}  
        **CPI:** {st.session_state.cpi}  
        **Unemployment:** {st.session_state.unemp}%
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="input-section" style="margin-top: 1rem;">', unsafe_allow_html=True)
        st.markdown("#### ℹ️ Quick Guide")
        st.markdown("""
        **Generate Forecast:** Predict sales for current inputs  
        **Compare Scenarios:** See holiday vs non-holiday impact  
        **Reset:** Restore default values  
        **Export:** Download all predictions
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Validation
    issues = validate_inputs(st.session_state.temp, st.session_state.fuel, 
                            st.session_state.cpi, st.session_state.unemp)
    
    if issues:
        for issue in issues:
            st.warning(issue)
    
    # Prediction Logic
    if predict_btn and not issues:
        try:
            pred, X_used = predict_one(
                st.session_state.store,
                st.session_state.holiday_flag,
                st.session_state.temp,
                st.session_state.fuel,
                st.session_state.cpi,
                st.session_state.unemp,
                st.session_state.week_date
            )
            
            st.session_state.last_prediction = pred
            
            st.session_state.history.append({
                "date": str(st.session_state.week_date),
                "store": int(st.session_state.store),
                "holiday_flag": int(st.session_state.holiday_flag),
                "temperature": float(st.session_state.temp),
                "fuel_price": float(st.session_state.fuel),
                "cpi": float(st.session_state.cpi),
                "unemployment": float(st.session_state.unemp),
                "pred_sales": float(pred),
            })
            
            st.success(f"✅ Forecast generated successfully!")
            
            # Display result
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-title">Forecast Result</div>', unsafe_allow_html=True)
            
            r1, r2, r3 = st.columns(3)
            with r1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Predicted Weekly Sales</div>
                        <div class="metric-value">{money(pred)}</div>
                        <div class="metric-change">Store {st.session_state.store}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            with r2:
                week_type = "Holiday Week" if st.session_state.holiday_flag else "Non-Holiday Week"
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Week Type</div>
                        <div class="metric-value" style="font-size: 1.5rem;">{week_type}</div>
                        <div class="metric-change">{st.session_state.week_date}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            with r3:
                st.plotly_chart(create_gauge_chart(pred, "Sales Forecast"), use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
    
    # Comparison Logic
    if compare_btn and not issues:
        try:
            pred_non, X_non = predict_one(
                st.session_state.store, 0,
                st.session_state.temp, st.session_state.fuel,
                st.session_state.cpi, st.session_state.unemp,
                st.session_state.week_date
            )
            
            pred_hol, X_hol = predict_one(
                st.session_state.store, 1,
                st.session_state.temp, st.session_state.fuel,
                st.session_state.cpi, st.session_state.unemp,
                st.session_state.week_date
            )
            
            diff = pred_hol - pred_non
            pct = (diff / pred_non * 100) if pred_non != 0 else 0.0
            
            st.success("✅ Scenario comparison completed!")
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-title">Holiday Impact Analysis</div>', unsafe_allow_html=True)
            
            c1, c2 = st.columns([1, 1])
            
            with c1:
                r1, r2, r3 = st.columns(3)
                
                with r1:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Non-Holiday Week</div>
                            <div class="metric-value">{money(pred_non)}</div>
                            <div class="metric-change">Baseline</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                with r2:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Holiday Week</div>
                            <div class="metric-value">{money(pred_hol)}</div>
                            <div class="metric-change">With uplift</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                with r3:
                    change_class = "positive" if diff > 0 else "negative"
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Impact</div>
                            <div class="metric-value">{money(abs(diff))}</div>
                            <div class="metric-change {change_class}">{'+' if diff > 0 else ''}{pct:.1f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            
            with c2:
                st.plotly_chart(create_comparison_chart(pred_non, pred_hol), use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Comparison failed: {str(e)}")

# ============================================================
# TAB 3: ANALYTICS
# ============================================================
with tab3:
    st.markdown('<div class="section-title">Advanced Analytics & Insights</div>', unsafe_allow_html=True)
    
    if len(st.session_state.history) >= 5:
        hist_df = pd.DataFrame(st.session_state.history)
        hist_df["pred_sales"] = pd.to_numeric(hist_df["pred_sales"], errors="coerce")
        hist_df = hist_df.dropna(subset=["pred_sales"])
        
        # Statistical Summary
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Mean</div>
                    <div class="metric-value" style="font-size: 1.4rem;">{money(hist_df["pred_sales"].mean())}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Median</div>
                    <div class="metric-value" style="font-size: 1.4rem;">{money(hist_df["pred_sales"].median())}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Std Dev</div>
                    <div class="metric-value" style="font-size: 1.4rem;">{money(hist_df["pred_sales"].std())}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Min</div>
                    <div class="metric-value" style="font-size: 1.4rem;">{money(hist_df["pred_sales"].min())}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col5:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Max</div>
                    <div class="metric-value" style="font-size: 1.4rem;">{money(hist_df["pred_sales"].max())}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Distribution Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="section-title">Sales Distribution</div>', unsafe_allow_html=True)
            fig = px.histogram(hist_df, x="pred_sales", nbins=20,
                             title="Forecast Distribution",
                             labels={"pred_sales": "Predicted Sales ($)"})
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', family='Outfit'),
                title_font=dict(size=18, color='white'),
                showlegend=False,
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
            )
            fig.update_traces(marker_color='#8b5cf6')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown('<div class="section-title">Store Performance</div>', unsafe_allow_html=True)
            store_avg = hist_df.groupby('store')['pred_sales'].mean().reset_index()
            store_avg = store_avg.sort_values('pred_sales', ascending=False).head(10)
            
            fig = px.bar(store_avg, x='store', y='pred_sales',
                        title="Top 10 Stores by Avg Forecast",
                        labels={"store": "Store", "pred_sales": "Avg Sales ($)"})
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', family='Outfit'),
                title_font=dict(size=18, color='white'),
                showlegend=False,
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickformat='$,.0f')
            )
            fig.update_traces(marker_color='#ec4899')
            st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.info("📊 Generate at least 5 predictions to see advanced analytics")

# ============================================================
# TAB 4: ABOUT
# ============================================================
with tab4:
    st.markdown('<div class="section-title">About This Platform</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            """
            <div class="input-section">
                <h3>🎯 Purpose</h3>
                <p>The Walmart Sales Intelligence Platform leverages machine learning to predict weekly sales 
                across different stores, enabling data-driven decisions for inventory management, staffing, 
                and promotional planning.</p>
                
                <h3>🔧 Technology Stack</h3>
                <ul>
                    <li><strong>Framework:</strong> Streamlit</li>
                    <li><strong>ML Model:</strong> Scikit-learn (Random Forest/Gradient Boosting)</li>
                    <li><strong>Visualization:</strong> Plotly</li>
                    <li><strong>Data Processing:</strong> Pandas, NumPy</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            """
            <div class="input-section">
                <h3>📊 Features</h3>
                <ul>
                    <li>Real-time sales forecasting</li>
                    <li>Holiday impact analysis</li>
                    <li>Multi-store comparison</li>
                    <li>Economic indicator integration</li>
                    <li>Historical trend tracking</li>
                    <li>Export & reporting capabilities</li>
                </ul>
                
                <h3>💡 Use Cases</h3>
                <ul>
                    <li><strong>Inventory Planning:</strong> Stock optimization</li>
                    <li><strong>Workforce Management:</strong> Staff scheduling</li>
                    <li><strong>Promotional Strategy:</strong> Campaign timing</li>
                    <li><strong>Financial Forecasting:</strong> Revenue planning</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class="input-section">
            <h3>📈 Model Information</h3>
            <p>The prediction model was trained on historical Walmart sales data including:</p>
            <ul>
                <li>Store-specific characteristics</li>
                <li>Temporal features (year, month, week)</li>
                <li>Holiday indicators</li>
                <li>Economic indicators (CPI, unemployment)</li>
                <li>Environmental factors (temperature, fuel prices)</li>
            </ul>
            <p><strong>Note:</strong> Model performance metrics and validation results should be documented 
            separately for production deployment.</p>
        </div>
        """,
        unsafe_allow_html=True
    )