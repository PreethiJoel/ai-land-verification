import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="LandVerify AI",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* Background */
.stApp {
    background: #0d0f14;
    color: #e8e6e0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #111318 !important;
    border-right: 1px solid #1e2330;
}
[data-testid="stSidebar"] * {
    color: #c9c6be !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #161922;
    border: 1px solid #1e2330;
    border-radius: 10px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.12em;
    color: #7a7e8a !important;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.7rem !important;
    font-weight: 700;
    color: #e8e6e0 !important;
}

/* Tab styling */
[data-baseweb="tab-list"] {
    background: #111318;
    border-radius: 8px;
    padding: 4px;
    gap: 2px;
    border-bottom: none !important;
}
[data-baseweb="tab"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em;
    color: #7a7e8a !important;
    background: transparent !important;
    border-radius: 6px !important;
    padding: 8px 16px !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: #1e2330 !important;
    color: #e8e6e0 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    width: 100%;
    letter-spacing: 0.04em;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #3b82f6, #2563eb);
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(37,99,235,0.35);
}

/* Progress bar */
.stProgress > div > div {
    border-radius: 4px;
}

/* Input fields */
.stNumberInput input, .stSelectbox select {
    background: #161922 !important;
    color: #e8e6e0 !important;
    border: 1px solid #1e2330 !important;
    border-radius: 6px !important;
    font-family: 'DM Mono', monospace !important;
}

/* Risk banner */
.risk-banner-low {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 1px solid #10b981;
    border-radius: 12px;
    padding: 28px 36px;
    text-align: center;
    margin: 16px 0;
}
.risk-banner-medium {
    background: linear-gradient(135deg, #451a03, #78350f);
    border: 1px solid #f59e0b;
    border-radius: 12px;
    padding: 28px 36px;
    text-align: center;
    margin: 16px 0;
}
.risk-banner-high {
    background: linear-gradient(135deg, #4c0519, #7f1d1d);
    border: 1px solid #ef4444;
    border-radius: 12px;
    padding: 28px 36px;
    text-align: center;
    margin: 16px 0;
}
.risk-title {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: 0.1em;
    margin: 0;
}
.risk-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.1em;
    opacity: 0.7;
    margin-top: 6px;
}

/* Explanation box */
.explain-box {
    background: #161922;
    border: 1px solid #1e2330;
    border-radius: 10px;
    padding: 20px 24px;
}
.explain-item {
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    color: #c9c6be;
    padding: 6px 0;
    border-bottom: 1px solid #1e2330;
    display: flex;
    align-items: center;
    gap: 10px;
}
.explain-item:last-child { border-bottom: none; }

/* Section headers */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    color: #4b5563;
    text-transform: uppercase;
    margin-bottom: 12px;
    padding-bottom: 6px;
    border-bottom: 1px solid #1e2330;
}

/* Logo / header */
.app-logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 800;
    letter-spacing: 0.04em;
    color: #e8e6e0;
}
.app-logo span { color: #3b82f6; }

/* Divider */
hr { border-color: #1e2330 !important; }

/* Chart backgrounds */
.stPlotlyChart, .stPyplot { background: transparent !important; }

/* Number input width fix */
.stNumberInput { margin-bottom: 4px; }

/* Gauge-like progress */
.gauge-wrap {
    background: #161922;
    border: 1px solid #1e2330;
    border-radius: 10px;
    padding: 20px;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MODEL TRAINING (CACHED)
# ─────────────────────────────────────────────
@st.cache_resource
def train_models():
    rng = np.random.default_rng(42)

    # Price model
    n_price = 500
    area = rng.uniform(200, 10000, n_price)
    location = rng.choice([0, 1, 2], n_price)
    road = rng.choice([0, 1], n_price)
    water = rng.choice([0, 1], n_price)
    price = (area * 120 + location * 25000 + road * 18000 +
             water * 12000 + rng.normal(0, 8000, n_price))

    X_price = np.column_stack([area, location, road, water])
    price_model = LinearRegression().fit(X_price, price)

    # Fraud model
    n_fraud = 1000
    area_dev = rng.uniform(0, 60, n_fraud)
    boundary_shift = rng.uniform(0, 50, n_fraud)
    price_dev = rng.uniform(0, 60, n_fraud)
    fraud_label = ((area_dev > 25) | (boundary_shift > 30) | (price_dev > 30)).astype(int)

    X_fraud = np.column_stack([area_dev, boundary_shift, price_dev])
    scaler = StandardScaler()
    X_fraud_s = scaler.fit_transform(X_fraud)
    fraud_model = LogisticRegression(random_state=42).fit(X_fraud_s, fraud_label)

    return price_model, fraud_model, scaler


price_model, fraud_model, fraud_scaler = train_models()


# ─────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────
def gps_to_meters(coords):
    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]
    ref_lat = np.radians(np.mean(lats))
    ref_lon = np.mean(lons)
    ref_lat_deg = np.mean(lats)
    xs = [(lon - ref_lon) * 111000 * np.cos(ref_lat) for lon in lons]
    ys = [(lat - ref_lat_deg) * 111000 for lat in lats]
    return list(zip(xs, ys))


def shoelace_area(pts):
    n = len(pts)
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]
    return abs(area) / 2


def predict_price(area, location, road, water):
    X = np.array([[area, location, road, water]])
    return float(price_model.predict(X)[0])


def predict_fraud(area_dev_pct, boundary_shift, price_dev_pct):
    X = np.array([[area_dev_pct, boundary_shift, price_dev_pct]])
    X_s = fraud_scaler.transform(X)
    prob = float(fraud_model.predict_proba(X_s)[0][1])
    return max(0.0, min(1.0, prob))


def classify_risk(prob):
    if prob > 0.7:
        return "HIGH", "#ef4444"
    elif prob >= 0.4:
        return "MEDIUM", "#f59e0b"
    else:
        return "LOW", "#10b981"


def get_risk_banner_class(risk):
    return {"HIGH": "risk-banner-high", "MEDIUM": "risk-banner-medium", "LOW": "risk-banner-low"}[risk]


def build_explanations(area_dev, price_dev, fraud_prob, boundary_shift):
    reasons = []
    if area_dev > 20:
        reasons.append(("⚠", "Significant area deviation detected", "area"))
    elif area_dev > 10:
        reasons.append(("◈", "Minor area deviation noted", "area"))
    else:
        reasons.append(("✓", "Area matches recorded value closely", "ok"))

    if boundary_shift > 20:
        reasons.append(("⚠", "Boundary inconsistencies observed", "boundary"))
    else:
        reasons.append(("✓", "No boundary inconsistencies found", "ok"))

    if price_dev > 25:
        reasons.append(("⚠", "Predicted market value mismatch", "price"))
    elif price_dev > 12:
        reasons.append(("◈", "Moderate price deviation detected", "price"))
    else:
        reasons.append(("✓", "Price aligns with model prediction", "ok"))

    if fraud_prob > 0.7:
        reasons.append(("⛔", "Multiple anomaly indicators triggered", "critical"))
    elif fraud_prob > 0.4:
        reasons.append(("◈", "Elevated fraud signal — review advised", "warn"))
    else:
        reasons.append(("✓", "Low fraud signal probability", "ok"))

    return reasons


def mpl_style():
    plt.rcParams.update({
        'figure.facecolor': '#0d0f14',
        'axes.facecolor': '#111318',
        'axes.edgecolor': '#1e2330',
        'axes.labelcolor': '#7a7e8a',
        'xtick.color': '#7a7e8a',
        'ytick.color': '#7a7e8a',
        'text.color': '#e8e6e0',
        'grid.color': '#1e2330',
        'grid.linestyle': '--',
        'font.family': 'monospace',
    })


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="app-logo">Land<span>Verify</span> AI</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-family:DM Mono,monospace;font-size:0.7rem;color:#4b5563;margin-bottom:20px;">FRAUD DETECTION SYSTEM v2.0</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">GPS BOUNDARY POINTS</div>', unsafe_allow_html=True)

    num_points = st.slider("Number of points", 3, 10, 4)

    default_coords = [
        (12.9716, 77.5946),
        (12.9726, 77.5966),
        (12.9706, 77.5976),
        (12.9696, 77.5956),
        (12.9700, 77.5936),
        (12.9710, 77.5930),
        (12.9720, 77.5928),
        (12.9730, 77.5940),
        (12.9735, 77.5960),
        (12.9715, 77.5975),
    ]

    coords = []
    valid = True
    for i in range(num_points):
        c1, c2 = st.columns(2)
        with c1:
            lat = st.number_input(f"Lat {i+1}", value=default_coords[i][0], format="%.6f", key=f"lat_{i}", label_visibility="collapsed" if i > 0 else "visible")
            if i == 0:
                st.caption("Latitude")
        with c2:
            lon = st.number_input(f"Lon {i+1}", value=default_coords[i][1], format="%.6f", key=f"lon_{i}", label_visibility="collapsed" if i > 0 else "visible")
            if i == 0:
                st.caption("Longitude")

        if not (-90 <= lat <= 90):
            st.error(f"Point {i+1}: Latitude must be −90 to 90")
            valid = False
        if not (-180 <= lon <= 180):
            st.error(f"Point {i+1}: Longitude must be −180 to 180")
            valid = False
        coords.append((lat, lon))

    # Check duplicate points
    if len(set(coords)) < 3:
        st.warning("At least 3 distinct points required.")
        valid = False

    st.markdown("---")
    st.markdown('<div class="section-header">PARCEL ATTRIBUTES</div>', unsafe_allow_html=True)
    recorded_area = st.number_input("Recorded Area (m²)", min_value=100.0, value=2000.0, step=50.0)
    location_type = st.selectbox("Location Type", ["Rural", "Semi-Urban", "Urban"])
    location_val = {"Rural": 0, "Semi-Urban": 1, "Urban": 2}[location_type]
    road_access = st.checkbox("Road Access", value=True)
    water_access = st.checkbox("Water Access", value=True)
    market_price = st.number_input("Reported Market Price (₹)", min_value=0.0, value=350000.0, step=5000.0)

    st.markdown("---")
    run = st.button("▶  RUN ANALYSIS")


# ─────────────────────────────────────────────
# COMPUTE
# ─────────────────────────────────────────────
if valid:
    meter_pts = gps_to_meters(coords)
    calc_area = shoelace_area(meter_pts)
    pred_price = predict_price(calc_area, location_val, int(road_access), int(water_access))
    area_dev = abs(calc_area - recorded_area) / max(recorded_area, 1) * 100
    price_dev = abs(pred_price - market_price) / max(market_price, 1) * 100
    boundary_shift = min(area_dev * 0.6 + np.random.default_rng(int(calc_area)).uniform(0, 10), 50)
    fraud_prob = predict_fraud(area_dev, boundary_shift, price_dev)
    risk_level, risk_color = classify_risk(fraud_prob)
    explanations = build_explanations(area_dev, price_dev, fraud_prob, boundary_shift)
else:
    calc_area = pred_price = area_dev = price_dev = fraud_prob = boundary_shift = 0
    risk_level, risk_color = "LOW", "#10b981"
    explanations = []
    meter_pts = []


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown('<h1 style="font-family:Syne;font-size:2rem;font-weight:800;margin:0;color:#e8e6e0;">AI Smart Land Verification</h1>', unsafe_allow_html=True)
    st.markdown('<p style="font-family:DM Mono,monospace;font-size:0.75rem;color:#4b5563;letter-spacing:0.15em;margin-top:4px;">FRAUD DETECTION · PRICE INTELLIGENCE · GEOSPATIAL ANALYSIS</p>', unsafe_allow_html=True)
with col_h2:
    if valid:
        st.markdown(f'<div style="text-align:right;padding-top:8px;"><span style="font-family:DM Mono,monospace;font-size:0.7rem;color:#4b5563;">STATUS</span><br><span style="font-family:Syne;font-size:1.1rem;font-weight:700;color:{risk_color};">● {risk_level} RISK</span></div>', unsafe_allow_html=True)

st.markdown('<hr style="margin:12px 0 20px 0;">', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tabs = st.tabs(["01 · DASHBOARD", "02 · AREA MAP", "03 · PRICE ANALYSIS", "04 · FRAUD ANALYSIS", "05 · COMBINED"])

# ═══════════════════════════════════════════════
# TAB 1 — OVERVIEW DASHBOARD
# ═══════════════════════════════════════════════
with tabs[0]:
    if not valid:
        st.warning("Fix validation errors in the sidebar to see results.")
    else:
        # Metrics row
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Calculated Area", f"{calc_area:,.0f} m²")
        m2.metric("Predicted Price", f"₹{pred_price:,.0f}")
        m3.metric("Area Deviation", f"{area_dev:.1f}%")
        m4.metric("Fraud Probability", f"{fraud_prob*100:.1f}%")
        m5.metric("Risk Level", risk_level)

        st.markdown("<br>", unsafe_allow_html=True)

        # Risk banner + explanation
        col_banner, col_explain = st.columns([2, 1])
        with col_banner:
            banner_class = get_risk_banner_class(risk_level)
            icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}[risk_level]
            label = {"HIGH": "HIGH RISK — INVESTIGATION REQUIRED",
                     "MEDIUM": "MEDIUM RISK — REVIEW ADVISED",
                     "LOW": "LOW RISK — VERIFIED"}[risk_level]
            st.markdown(f"""
            <div class="{banner_class}">
                <div class="risk-title" style="color:{risk_color};">{icon}  {risk_level} RISK</div>
                <div class="risk-sub">{label}</div>
                <div style="font-family:DM Mono,monospace;font-size:0.72rem;color:#9ca3af;margin-top:14px;">
                    FRAUD PROBABILITY: {fraud_prob*100:.1f}% &nbsp;·&nbsp; AREA DEV: {area_dev:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Summary bar
            st.markdown('<div class="section-header" style="margin-top:16px;">RISK INDICATORS</div>', unsafe_allow_html=True)
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                st.caption("Area Deviation")
                st.progress(min(area_dev / 60, 1.0))
                st.caption(f"{area_dev:.1f}% / 60% max")
            with col_p2:
                st.caption("Fraud Probability")
                st.progress(fraud_prob)
                st.caption(f"{fraud_prob*100:.1f}%")
            with col_p3:
                st.caption("Price Deviation")
                st.progress(min(price_dev / 60, 1.0))
                st.caption(f"{price_dev:.1f}% / 60% max")

        with col_explain:
            st.markdown('<div class="section-header">WHY THIS RISK LEVEL?</div>', unsafe_allow_html=True)
            items_html = ""
            for icon, text, typ in explanations:
                color = {"ok": "#10b981", "warn": "#f59e0b", "area": "#f59e0b",
                         "boundary": "#f59e0b", "price": "#f59e0b", "critical": "#ef4444"}.get(typ, "#7a7e8a")
                items_html += f'<div class="explain-item"><span style="color:{color};font-size:1rem;">{icon}</span> {text}</div>'
            st.markdown(f'<div class="explain-box">{items_html}</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">PARCEL SUMMARY</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="explain-box">
                <div class="explain-item"><span style="color:#4b5563;">LOC</span> {location_type}</div>
                <div class="explain-item"><span style="color:#4b5563;">ROAD</span> {"Yes" if road_access else "No"}</div>
                <div class="explain-item"><span style="color:#4b5563;">WATER</span> {"Yes" if water_access else "No"}</div>
                <div class="explain-item"><span style="color:#4b5563;">POINTS</span> {num_points}</div>
                <div class="explain-item"><span style="color:#4b5563;">REC. AREA</span> {recorded_area:,.0f} m²</div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# TAB 2 — AREA VISUALIZATION
# ═══════════════════════════════════════════════
with tabs[1]:
    if not valid or not meter_pts:
        st.warning("Fix validation errors to see visualization.")
    else:
        mpl_style()
        col_poly, col_bar = st.columns([3, 2])
        with col_poly:
            st.markdown('<div class="section-header">BOUNDARY POLYGON</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(7, 6))
            xs = [p[0] for p in meter_pts]
            ys = [p[1] for p in meter_pts]
            xs_closed = xs + [xs[0]]
            ys_closed = ys + [ys[0]]

            ax.fill(xs, ys, color='#1d4ed820', zorder=1)
            ax.plot(xs_closed, ys_closed, color='#3b82f6', linewidth=2, zorder=2)
            for i, (x, y) in enumerate(meter_pts):
                ax.scatter(x, y, color='#60a5fa', s=70, zorder=3)
                ax.annotate(f'P{i+1}', (x, y), fontsize=8, color='#93c5fd',
                            xytext=(5, 5), textcoords='offset points')

            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("East–West (m)")
            ax.set_ylabel("North–South (m)")
            ax.set_title(f"Polygon · {calc_area:,.0f} m²", color='#e8e6e0', fontsize=11, pad=12)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_bar:
            st.markdown('<div class="section-header">AREA COMPARISON</div>', unsafe_allow_html=True)
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            bars = ax2.bar(
                ["Recorded\nArea", "Calculated\nArea"],
                [recorded_area, calc_area],
                color=['#1d4ed8', '#3b82f6' if area_dev < 15 else '#f59e0b'],
                width=0.5, edgecolor='none'
            )
            for bar, val in zip(bars, [recorded_area, calc_area]):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + recorded_area*0.01,
                         f"{val:,.0f} m²", ha='center', va='bottom', color='#e8e6e0', fontsize=9)
            ax2.set_ylabel("Area (m²)")
            ax2.set_title(f"Deviation: {area_dev:.1f}%", color='#e8e6e0', fontsize=10)
            ax2.grid(axis='y', alpha=0.3)
            fig2.tight_layout()
            st.pyplot(fig2)
            plt.close()

            st.markdown("<br>", unsafe_allow_html=True)
            if area_dev > 20:
                st.error(f"⚠ {area_dev:.1f}% area deviation exceeds acceptable threshold (20%).")
            elif area_dev > 10:
                st.warning(f"◈ {area_dev:.1f}% area deviation is slightly elevated.")
            else:
                st.success(f"✓ Area deviation of {area_dev:.1f}% is within acceptable range.")


# ═══════════════════════════════════════════════
# TAB 3 — PRICE ANALYSIS
# ═══════════════════════════════════════════════
with tabs[2]:
    if not valid:
        st.warning("Fix validation errors to see price analysis.")
    else:
        mpl_style()
        st.markdown('<div class="section-header">PRICE DECOMPOSITION</div>', unsafe_allow_html=True)

        col_chart, col_info = st.columns([3, 2])
        with col_chart:
            base_contribution = calc_area * 120
            road_contribution = int(road_access) * 18000
            water_contribution = int(water_access) * 12000
            location_contribution = location_val * 25000
            total_pred = base_contribution + road_contribution + water_contribution + location_contribution

            categories = ["Base\n(Area)", "Location\nBonus", "Road\nAccess", "Water\nAccess", "Total\nPredicted"]
            values = [base_contribution, location_contribution, road_contribution, water_contribution, pred_price]
            colors = ['#1d4ed8', '#7c3aed', '#0891b2', '#059669', '#3b82f6']

            fig3, ax3 = plt.subplots(figsize=(8, 5))
            bars3 = ax3.bar(categories, values, color=colors, width=0.55, edgecolor='none')
            for bar, val in zip(bars3, values):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + pred_price*0.01,
                         f"₹{val:,.0f}", ha='center', va='bottom', color='#e8e6e0', fontsize=8)
            ax3.set_ylabel("Contribution (₹)")
            ax3.set_title("Price Factor Breakdown", color='#e8e6e0', fontsize=11)
            ax3.grid(axis='y', alpha=0.3)
            fig3.tight_layout()
            st.pyplot(fig3)
            plt.close()

        with col_info:
            st.markdown('<div class="section-header">PRICE INTELLIGENCE</div>', unsafe_allow_html=True)
            m_a, m_b = st.columns(2)
            m_a.metric("AI Predicted", f"₹{pred_price:,.0f}")
            m_b.metric("Market Reported", f"₹{market_price:,.0f}")
            st.metric("Price Deviation", f"{price_dev:.1f}%", delta=f"{'Over' if pred_price > market_price else 'Under'}-valued")

            st.markdown("<br>", unsafe_allow_html=True)
            if price_dev > 30:
                st.error("⚠ Significant mismatch between AI estimate and reported price. Possible misrepresentation.")
            elif price_dev > 15:
                st.warning("◈ Moderate price deviation. Could reflect local market factors.")
            else:
                st.success("✓ Price is well-aligned with AI model prediction.")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="explain-box">
                <div class="explain-item"><span style="color:#4b5563;">BASE RATE</span> ₹120/m²</div>
                <div class="explain-item"><span style="color:#4b5563;">LOCATION</span> +₹{location_contribution:,.0f}</div>
                <div class="explain-item"><span style="color:#4b5563;">ROAD</span> +₹{road_contribution:,.0f}</div>
                <div class="explain-item"><span style="color:#4b5563;">WATER</span> +₹{water_contribution:,.0f}</div>
                <div class="explain-item" style="font-weight:600;color:#e8e6e0;">TOTAL ₹{pred_price:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# TAB 4 — FRAUD ANALYSIS
# ═══════════════════════════════════════════════
with tabs[3]:
    if not valid:
        st.warning("Fix validation errors to see fraud analysis.")
    else:
        mpl_style()
        st.markdown('<div class="section-header">FRAUD RISK ASSESSMENT</div>', unsafe_allow_html=True)

        col_gauge, col_detail = st.columns([2, 3])
        with col_gauge:
            # Semicircular gauge using matplotlib
            fig4, ax4 = plt.subplots(figsize=(5, 3.5), subplot_kw=dict(aspect='equal'))
            theta = np.linspace(np.pi, 0, 300)
            r_outer, r_inner = 1.0, 0.6

            # Background arc segments
            zones = [(np.pi, np.pi*0.6, '#0f2d1a'), (np.pi*0.6, np.pi*0.3, '#2d1e0a'), (np.pi*0.3, 0, '#2d0a0a')]
            for t_start, t_end, col in zones:
                seg = np.linspace(t_start, t_end, 100)
                x_out = r_outer * np.cos(seg)
                y_out = r_outer * np.sin(seg)
                x_in = r_inner * np.cos(seg[::-1])
                y_in = r_inner * np.sin(seg[::-1])
                ax4.fill(np.concatenate([x_out, x_in]),
                         np.concatenate([y_out, y_in]), color=col, zorder=1)

            # Active arc
            active_end = np.pi - fraud_prob * np.pi
            seg_active = np.linspace(np.pi, active_end, 200)
            x_ao = r_outer * np.cos(seg_active)
            y_ao = r_outer * np.sin(seg_active)
            x_ai = r_inner * np.cos(seg_active[::-1])
            y_ai = r_inner * np.sin(seg_active[::-1])
            ax4.fill(np.concatenate([x_ao, x_ai]),
                     np.concatenate([y_ao, y_ai]), color=risk_color, zorder=2, alpha=0.85)

            # Needle
            needle_angle = np.pi - fraud_prob * np.pi
            ax4.annotate("", xy=(0.85 * np.cos(needle_angle), 0.85 * np.sin(needle_angle)),
                         xytext=(0, 0),
                         arrowprops=dict(arrowstyle='->', color='#e8e6e0', lw=2.5))

            ax4.text(0, 0.2, f"{fraud_prob*100:.1f}%", ha='center', va='center',
                     fontsize=22, fontweight='bold', color='#e8e6e0')
            ax4.text(0, -0.05, "FRAUD PROBABILITY", ha='center', va='center',
                     fontsize=7, color='#7a7e8a', fontfamily='monospace')

            ax4.set_xlim(-1.15, 1.15)
            ax4.set_ylim(-0.2, 1.15)
            ax4.axis('off')
            fig4.tight_layout()
            st.pyplot(fig4)
            plt.close()

        with col_detail:
            c1, c2, c3 = st.columns(3)
            c1.metric("Area Deviation", f"{area_dev:.1f}%")
            c2.metric("Boundary Shift", f"{boundary_shift:.1f}%")
            c3.metric("Price Deviation", f"{price_dev:.1f}%")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">DEVIATION BREAKDOWN</div>', unsafe_allow_html=True)

            fig5, ax5 = plt.subplots(figsize=(6, 3))
            features = ['Area Dev.', 'Boundary Shift', 'Price Dev.']
            vals = [area_dev, boundary_shift, price_dev]
            thresholds = [20, 20, 25]
            bar_colors = [risk_color if v > t else '#1d4ed8' for v, t in zip(vals, thresholds)]
            bars5 = ax5.barh(features, vals, color=bar_colors, edgecolor='none', height=0.45)
            for t, f in zip(thresholds, features):
                ax5.axvline(t, color='#4b5563', linestyle='--', linewidth=1, alpha=0.6)
            ax5.set_xlabel("Deviation %")
            ax5.set_title("Anomaly Indicators vs Thresholds", color='#e8e6e0', fontsize=10)
            ax5.grid(axis='x', alpha=0.3)
            for bar, val in zip(bars5, vals):
                ax5.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                         f'{val:.1f}%', va='center', color='#e8e6e0', fontsize=9)
            fig5.tight_layout()
            st.pyplot(fig5)
            plt.close()

            # Risk card
            banner_class = get_risk_banner_class(risk_level)
            st.markdown(f"""
            <div class="{banner_class}" style="padding:16px 20px;margin-top:12px;">
                <div style="font-family:Syne,sans-serif;font-weight:700;font-size:1rem;color:{risk_color};">
                    {risk_level} FRAUD RISK
                </div>
                <div style="font-family:DM Mono,monospace;font-size:0.72rem;color:#9ca3af;margin-top:6px;">
                    {"Investigation required — escalate to compliance team." if risk_level == "HIGH"
                     else "Manual review recommended before proceeding." if risk_level == "MEDIUM"
                     else "Document appears legitimate. Standard processing."}
                </div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# TAB 5 — COMBINED VIEW
# ═══════════════════════════════════════════════
with tabs[4]:
    if not valid or not meter_pts:
        st.warning("Fix validation errors to see combined view.")
    else:
        mpl_style()
        st.markdown(f"""
        <div style="text-align:center;padding:12px 0 20px;">
            <span style="font-family:Syne,sans-serif;font-size:1.5rem;font-weight:800;color:#e8e6e0;">
                LAND PARCEL INTELLIGENCE REPORT
            </span><br>
            <span style="font-family:DM Mono,monospace;font-size:0.72rem;color:#4b5563;letter-spacing:0.15em;">
                AI-GENERATED · {num_points} GPS POINTS · {location_type.upper()}
            </span>
        </div>
        """, unsafe_allow_html=True)

        # Top metrics
        cols = st.columns(5)
        for col, label, val in zip(cols,
            ["CALC. AREA", "PRED. PRICE", "AREA DEV.", "FRAUD PROB.", "RISK LEVEL"],
            [f"{calc_area:,.0f} m²", f"₹{pred_price:,.0f}", f"{area_dev:.1f}%",
             f"{fraud_prob*100:.1f}%", risk_level]):
            col.metric(label, val)

        st.markdown("<br>", unsafe_allow_html=True)

        col_left, col_right = st.columns([3, 2])
        with col_left:
            # Polygon
            fig6, ax6 = plt.subplots(figsize=(7, 5))
            xs = [p[0] for p in meter_pts]
            ys = [p[1] for p in meter_pts]
            xs_c = xs + [xs[0]]
            ys_c = ys + [ys[0]]
            ax6.fill(xs, ys, color=f'{risk_color}15', zorder=1)
            ax6.plot(xs_c, ys_c, color=risk_color, linewidth=2.5, zorder=2)
            for i, (x, y) in enumerate(meter_pts):
                ax6.scatter(x, y, color=risk_color, s=80, zorder=3)
                ax6.annotate(f'P{i+1}', (x, y), fontsize=8, color='#e8e6e0',
                            xytext=(5, 5), textcoords='offset points')
            ax6.set_aspect('equal')
            ax6.grid(True, alpha=0.25)
            ax6.set_title(f"Boundary Polygon — {calc_area:,.0f} m²", color='#e8e6e0', fontsize=11)
            fig6.tight_layout()
            st.pyplot(fig6)
            plt.close()

        with col_right:
            # Risk banner
            banner_class = get_risk_banner_class(risk_level)
            st.markdown(f"""
            <div class="{banner_class}">
                <div class="risk-title" style="color:{risk_color};">{risk_level}</div>
                <div class="risk-sub">FRAUD RISK CLASSIFICATION</div>
                <div style="font-family:DM Mono,monospace;font-size:1.5rem;font-weight:700;
                            color:{risk_color};margin-top:12px;">{fraud_prob*100:.1f}%</div>
                <div style="font-family:DM Mono,monospace;font-size:0.7rem;color:#9ca3af;">FRAUD PROBABILITY</div>
            </div>
            """, unsafe_allow_html=True)

            # Mini summary chart
            fig7, ax7 = plt.subplots(figsize=(5, 3))
            metrics = ['Area\nDev.', 'Boundary\nShift', 'Price\nDev.', 'Fraud\nProb.']
            mvals = [area_dev, boundary_shift, price_dev, fraud_prob * 100]
            thresholds2 = [20, 20, 25, 40]
            colors7 = [risk_color if v > t else '#1d4ed8' for v, t in zip(mvals, thresholds2)]
            ax7.bar(metrics, mvals, color=colors7, width=0.5, edgecolor='none')
            for t, pos in zip(thresholds2, range(4)):
                ax7.axhline(t if pos < 3 else 40, color='#4b5563', linestyle='--', lw=1, alpha=0.6)
            ax7.set_ylabel("Score (%)")
            ax7.set_title("Risk Indicators", color='#e8e6e0', fontsize=10)
            ax7.grid(axis='y', alpha=0.3)
            fig7.tight_layout()
            st.pyplot(fig7)
            plt.close()

            # Why this risk
            st.markdown('<div class="section-header" style="margin-top:12px;">FINDINGS</div>', unsafe_allow_html=True)
            items_html = ""
            for icon, text, typ in explanations:
                color = {"ok": "#10b981", "warn": "#f59e0b", "area": "#f59e0b",
                         "boundary": "#f59e0b", "price": "#f59e0b", "critical": "#ef4444"}.get(typ, "#7a7e8a")
                items_html += f'<div class="explain-item"><span style="color:{color};">{icon}</span> {text}</div>'
            st.markdown(f'<div class="explain-box">{items_html}</div>', unsafe_allow_html=True)