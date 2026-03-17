import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────── CONFIG ───────────────────────
st.set_page_config(
    page_title="Smart Sales Customer Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────── CUSTOM CSS ───────────────────────
st.markdown("""
<style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Target Streamlit's root elements to apply the dark slate theme */
    .stApp {
        background-color: #0B1120;
        font-family: 'Inter', sans-serif;
        color: #F8FAFC;
    }
    
    /* Remove default Streamlit top padding to match Stitch header spacing */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1400px !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid #1E293B !important;
    }
    
    /* ── Cards (Stitch Style) ── */
    .metric-card {
        background: linear-gradient(180deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.7) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(51, 65, 85, 0.5);
        border-top: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 16px; /* Reduced padding to fit 6 columns */
        margin-bottom: 16px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -4px rgba(0, 0, 0, 0.2);
        transition: all 0.25s ease;
        position: relative;
        overflow: hidden;
    }
    
    /* Accent glow on hover */
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #07b6d5, transparent);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.4), 0 10px 10px -5px rgba(0, 0, 0, 0.2);
        border-color: rgba(7, 182, 213, 0.3);
    }
    .metric-card:hover::before {
        opacity: 1;
    }
    
    /* Card Top Label */
    .metric-label {
        font-size: 0.75rem; /* Smaller for narrow columns */
        font-weight: 600;
        color: #94A3B8;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 8px;
        display: block;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    /* Card Big Number */
    .metric-value {
        font-size: 2rem; /* Scaled down slightly to prevent huge numbers wrapping */
        font-weight: 700;
        color: #F8FAFC;
        line-height: 1.1;
        letter-spacing: -0.02em;
        font-feature-settings: "tnum" on, "lnum" on; /* tabular lining numbers */
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    /* Card Subtext / Delta */
    .metric-delta {
        font-size: 0.875rem;
        color: #07b6d5; /* cyan accent */
        margin-top: 10px;
        font-weight: 500;
        display: inline-flex;
        align-items: center;
        background: rgba(7, 182, 213, 0.1);
        padding: 2px 8px;
        border-radius: 4px;
    }

    /* ── Headers ── */
    .page-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.03em;
        margin-bottom: 4px;
        background: linear-gradient(to right, #F8FAFC, #94A3B8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .page-subtitle {
        font-size: 1.125rem;
        color: #64748B;
        margin-bottom: 32px;
        font-weight: 500;
    }

    /* ── Score badge (Pill shape) ── */
    .score-badge {
        display: inline-flex;
        align-items: center;
        padding: 6px 14px;
        border-radius: 9999px; /* full pill */
        font-weight: 600;
        font-size: 0.75rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-top: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .badge-green  { background: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-yellow { background: rgba(245, 158, 11, 0.15); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-red    { background: rgba(239, 68, 68, 0.15); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.3); }

    /* ── Forms and Inputs ── */
    div[data-baseweb="select"] > div {
        background-color: #0F172A !important;
        border: 1px solid #1E293B !important;
        border-radius: 8px !important;
        color: #F8FAFC !important;
    }
    div[data-baseweb="select"] * {
        color: #F8FAFC !important;
    }
    input[type="number"], input[type="text"] {
        background-color: #0F172A !important;
        border: 1px solid #1E293B !important;
        border-radius: 8px !important;
        color: #F8FAFC !important;
    }
    input[type="number"]:focus, input[type="text"]:focus {
        border-color: #07b6d5 !important;
        box-shadow: 0 0 0 1px #07b6d5 !important;
    }
    
    /* Button styling to match Stitch */
    div.stButton > button {
        background: linear-gradient(135deg, #07b6d5 0%, #0284c7 100%) !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.5rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 6px -1px rgba(7, 182, 213, 0.3) !important;
    }
    div.stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 8px -1px rgba(7, 182, 213, 0.4) !important;
    }

    /* ── Sidebar brand ── */
    .sidebar-brand {
        padding: 24px 0 32px 0;
        border-bottom: 1px solid #1E293B;
        margin-bottom: 24px;
    }
    .sidebar-brand h2 {
        font-size: 1.5rem;
        font-weight: 800;
        color: #F8FAFC;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 8px;
        letter-spacing: -0.03em;
    }
    .sidebar-brand h2 span {
        color: #07b6d5;
    }
    .sidebar-brand p {
        font-size: 0.8rem;
        color: #64748B;
        margin: 6px 0 0 0;
        font-weight: 500;
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }
    
    /* ── Advanced Streamlit Hiding ── */
    /* Hide the top header completely */
    header {visibility: hidden !important;}
    
    /* Hide the deploy button and main menu */
    .stAppDeployButton {display: none !important;}
    #MainMenu {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    
    /* Hide toolbar (the '...' menu) on charts */
    [data-testid="stToolbar"] {visibility: hidden !important;}
    
    /* Remove padding around the main container to allow edge-to-edge if needed */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
        max-width: 1600px !important;
    }

    /* ── Dividers ── */
    hr {
        border-top: 1px solid rgba(51, 65, 85, 0.4) !important;
        margin: 2.5rem 0 !important;
    }
    
    /* Hide sidebar collapse button background for cleaner look */
    [data-testid="collapsedControl"] {
        color: #94A3B8 !important;
        background-color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────── PLOTLY THEME ───────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#94A3B8", size=13),
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#F8FAFC")),
)
GRID_AXIS = dict(gridcolor="#1E293B", zerolinecolor="#1E293B", title_font=dict(color="#F8FAFC"), tickfont=dict(color="#94A3B8"))
COLOR_SEQ = ["#07b6d5", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#F97316"]

# ─────────────────────── DATA & MODEL LOADING ───────────────────────
@st.cache_resource
def load_models():
    """Load all trained models and preprocessing objects."""
    paths = {
        'scaler': 'pkl/scaler.pkl',
        'pca': 'pkl/pca.pkl',
        'kmeans': 'pkl/kmeans_model.pkl',
        'classification': 'pkl/Classification_Model.pkl',
        'regression': 'pkl/Regression_Model.pkl',
        'encoder': 'pkl/gender_encoder.pkl',
    }
    models = {}
    for name, path in paths.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
        else:
            st.error(f"Model file not found: {path}")
            return None
    return models


@st.cache_data
def load_data():
    """Load the customer dataset."""
    path = "Dataset/customer_data.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def preprocess_input(data, models):
    """Transform raw input data into the format expected by the models."""
    input_df = pd.DataFrame([data])

    # Convert numeric fields to int where necessary (as done in Data_Preprocessing.py)
    input_df['Avg_Monthly_Spend'] = input_df['Avg_Monthly_Spend'].astype(int)
    input_df['Last_Month_Spend'] = input_df['Last_Month_Spend'].astype(int)

    # Encode Gender using the loaded LabelEncoder
    try:
        input_df['Gender'] = models['encoder'].transform(input_df['Gender'])
    except Exception:
        # Fallback if transform fails (should not happen if data is 'Male'/'Female')
        input_df['Gender'] = input_df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)

    # Encode Location (One-Hot) - matching Data_Preprocessing.py name mapping
    loc = input_df.pop('Location').values[0]
    input_df['Location_Suburban'] = 1 if loc == 'Suburban' else 0
    input_df['Location_Urban'] = 1 if loc == 'Urban' else 0

    # Numeric features to scale (must matches order in Data_Preprocessing.py numeric_cols)
    # numeric_cols in training were: Age, Tenure_Months, Avg_Monthly_Spend, Last_Month_Spend, 
    # Num_Transactions, Days_Since_Last_Purchase, Support_Tickets
    numeric_features = [
        'Age', 'Tenure_Months', 'Avg_Monthly_Spend', 'Last_Month_Spend',
        'Num_Transactions', 'Days_Since_Last_Purchase', 'Support_Tickets',
    ]
    input_df[numeric_features] = models['scaler'].transform(input_df[numeric_features])

    # Reorder features for PCA (must match training X columns order)
    # Training X columns order was: Age, Gender, Tenure_Months, Avg_Monthly_Spend, 
    # Last_Month_Spend, Num_Transactions, Days_Since_Last_Purchase, Support_Tickets, 
    # Location_Suburban, Location_Urban
    pca_cols = [
        'Age', 'Gender', 'Tenure_Months', 'Avg_Monthly_Spend', 'Last_Month_Spend',
        'Num_Transactions', 'Days_Since_Last_Purchase', 'Support_Tickets',
        'Location_Suburban', 'Location_Urban',
    ]
    input_df = input_df[pca_cols]

    # PCA Transform
    return models['pca'].transform(input_df)


@st.cache_data
def enrich_dataset(_models, df):
    """Add predicted cluster & churn columns to the entire dataset."""
    enriched = df.copy()
    pca_results = []
    clusters = []
    churn_preds = []
    spend_preds = []

    for _, row in enriched.iterrows():
        raw = {
            'Age': row['Age'],
            'Gender': row['Gender'],
            'Tenure_Months': row['Tenure_Months'],
            'Avg_Monthly_Spend': row['Avg_Monthly_Spend'],
            'Last_Month_Spend': row['Last_Month_Spend'],
            'Num_Transactions': row['Num_Transactions'],
            'Days_Since_Last_Purchase': row['Days_Since_Last_Purchase'],
            'Support_Tickets': row['Support_Tickets'],
            'Location': row['Location'],
        }
        processed = preprocess_input(raw, _models)
        pca_results.append(processed[0])
        clusters.append(_models['kmeans'].predict(processed)[0])
        churn_preds.append(_models['classification'].predict(processed)[0])
        spend_preds.append(_models['regression'].predict(processed)[0])

    enriched['Segment'] = clusters
    enriched['Predicted_Churn'] = churn_preds
    enriched['Predicted_Next_Spend'] = spend_preds
    return enriched


def render_metric_card(label, value, delta=""):
    """Render a styled metric card."""
    delta_html = f'<div class="metric-delta">{delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def health_score(row):
    """Compute a 0-100 health score for a single customer."""
    score = 50
    # Spend contribution (higher is better)
    score += min((row['Avg_Monthly_Spend'] / 20), 20)
    # Recency penalty (more days since purchase = worse)
    score -= min((row['Days_Since_Last_Purchase'] / 15), 20)
    # Tenure bonus
    score += min((row['Tenure_Months'] / 3), 10)
    # Ticket penalty
    score -= min((row['Support_Tickets'] * 3), 15)
    # Churn penalty
    if row.get('Predicted_Churn', 0) == 1:
        score -= 10
    return max(0, min(100, round(score)))


# ─────────────────────── LOAD EVERYTHING ───────────────────────
models = load_models()
df_raw = load_data()

if models is None:
    st.error("⚠️ Models could not be loaded. Ensure all `.pkl` files are in the `pkl/` directory.")
    st.stop()
if df_raw is None:
    st.error("⚠️ Dataset not found at `Dataset/customer_data.csv`.")
    st.stop()

df = enrich_dataset(models, df_raw)

# ─────────────────────── SIDEBAR NAV ───────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <h2>🧠 SCIS</h2>
        <p>Smart Customer Intelligence System</p>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "📊 Overview Dashboard",
            "🔍 Customer Analyzer",
            "📈 Segment Analytics",
            "⚠️ Churn Intelligence",
            "💰 Revenue Forecasting",
        ],
        label_visibility="collapsed",
    )

# ═══════════════════════════════════════════════════════════════
#  PAGE 1 – OVERVIEW DASHBOARD
# ═══════════════════════════════════════════════════════════════
if page == "📊 Overview Dashboard":
    st.markdown('<div class="page-header">Overview Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Executive summary of your customer portfolio</div>', unsafe_allow_html=True)

    # ── KPI Row ──
    total_customers = len(df)
    total_revenue = df['Avg_Monthly_Spend'].sum()
    avg_spend = df['Avg_Monthly_Spend'].mean()
    churn_rate = df['Predicted_Churn'].mean() * 100
    avg_tenure = df['Tenure_Months'].mean()
    avg_tickets = df['Support_Tickets'].mean()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        render_metric_card("Total Customers", f"{total_customers:,}", "Active portfolio")
    with c2:
        render_metric_card("Monthly Revenue", f"${total_revenue:,.0f}", "Aggregate spend")
    with c3:
        render_metric_card("Avg Monthly Spend", f"${avg_spend:,.2f}", "Per customer")
    with c4:
        render_metric_card("Churn Risk Rate", f"{churn_rate:.1f}%",
                           f"{df['Predicted_Churn'].sum()} at risk")
    with c5:
        render_metric_card("Avg Tenure", f"{avg_tenure:.1f} mo", "Customer loyalty")
    with c6:
        render_metric_card("Avg Tickets", f"{avg_tickets:.1f}", "Support load")

    st.divider()

    # ── Row 2: Revenue Trend + Customer Health Matrix ──
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Revenue Distribution by Tenure")
        # Create tenure buckets for a revenue trend view
        df_trend = df.copy()
        df_trend['Tenure_Bucket'] = pd.cut(
            df_trend['Tenure_Months'],
            bins=[0, 6, 12, 18, 24, 36, 60, 100],
            labels=['0-6m', '6-12m', '12-18m', '18-24m', '24-36m', '36-60m', '60m+'],
        )
        trend_data = df_trend.groupby('Tenure_Bucket', observed=True).agg(
            Avg_Spend=('Avg_Monthly_Spend', 'mean'),
            Count=('CustomerID', 'count'),
        ).reset_index()

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Bar(
            x=trend_data['Tenure_Bucket'], y=trend_data['Avg_Spend'],
            marker_color='#07b6d5', name='Avg Spend',
            hovertemplate='%{x}<br>Avg Spend: $%{y:.2f}<extra></extra>',
        ))
        fig_trend.add_trace(go.Scatter(
            x=trend_data['Tenure_Bucket'], y=trend_data['Count'],
            mode='lines+markers', yaxis='y2', name='Customer Count',
            line=dict(color='#06d6a0', width=2),
            marker=dict(size=7),
        ))
        fig_trend.update_layout(
            **PLOTLY_LAYOUT,
            height=380,
        )
        fig_trend.update_layout(
            yaxis=dict(title='Avg Spend ($)', **GRID_AXIS),
            yaxis2=dict(title='# Customers', overlaying='y', side='right', **GRID_AXIS),
            legend=dict(orientation='h', y=-0.18),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_right:
        st.subheader("Customer Health Matrix")
        fig_health = px.scatter(
            df, x='Tenure_Months', y='Avg_Monthly_Spend',
            color=df['Predicted_Churn'].map({0: 'Low Risk', 1: 'High Risk'}),
            color_discrete_map={'Low Risk': '#10B981', 'High Risk': '#EF4444'},
            opacity=0.65, hover_data=['CustomerID'],
        )
        fig_health.update_layout(**PLOTLY_LAYOUT, height=380, legend_title_text='Churn Risk')
        st.plotly_chart(fig_health, use_container_width=True)

    st.divider()

    # ── Row 3: Segment Distribution + Churn Risk Heatmap ──
    col_seg, col_heat = st.columns(2)

    with col_seg:
        st.subheader("Segment Distribution")
        seg_counts = df['Segment'].value_counts().sort_index().reset_index()
        seg_counts.columns = ['Segment', 'Count']
        seg_counts['Segment'] = seg_counts['Segment'].apply(lambda x: f'Group {x}')
        fig_seg = px.pie(
            seg_counts, names='Segment', values='Count',
            color_discrete_sequence=COLOR_SEQ,
            hole=0.45,
        )
        fig_seg.update_layout(**PLOTLY_LAYOUT, height=380)
        fig_seg.update_traces(textinfo='percent+label', textfont_size=13)
        st.plotly_chart(fig_seg, use_container_width=True)

    with col_heat:
        st.subheader("Churn Risk Heatmap")
        # Heatmap: Location × Segment with avg churn rate
        heat_data = df.groupby(['Location', 'Segment'])['Predicted_Churn'].mean().reset_index()
        heat_pivot = heat_data.pivot(index='Location', columns='Segment', values='Predicted_Churn')
        heat_pivot.columns = [f'Group {c}' for c in heat_pivot.columns]

        fig_heat = px.imshow(
            heat_pivot.values,
            labels=dict(x='Segment', y='Location', color='Churn Rate'),
            x=heat_pivot.columns.tolist(),
            y=heat_pivot.index.tolist(),
            color_continuous_scale=[[0, '#10B981'], [0.5, '#F59E0B'], [1, '#EF4444']],
            aspect='auto',
            text_auto='.0%',
        )
        fig_heat.update_layout(**PLOTLY_LAYOUT, height=380)
        st.plotly_chart(fig_heat, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
#  PAGE 2 – CUSTOMER ANALYZER
# ═══════════════════════════════════════════════════════════════
elif page == "🔍 Customer Analyzer":
    st.markdown('<div class="page-header">Customer Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Deep individual profiling with ML-powered insights</div>', unsafe_allow_html=True)

    # ── Input mode ──
    input_mode = st.radio("Select Input Mode", ["Existing Customer", "New Customer (Manual Entry)"], horizontal=True)

    defaults = {
        'Gender': "Male", 'Age': 30, 'Location': "Urban", 'Tenure': 12,
        'Avg_Spend': 500, 'Last_Spend': 450, 'Transactions': 5,
        'Days_Since': 10, 'Tickets': 1,
    }
    selected_customer_id = None

    if input_mode == "Existing Customer":
        col_sel, _ = st.columns([1, 2])
        with col_sel:
            selected_customer_id = st.selectbox("Select Customer", df['CustomerID'].tolist())
        if selected_customer_id:
            cust = df[df['CustomerID'] == selected_customer_id].iloc[0]
            defaults.update({
                'Gender': cust['Gender'], 'Age': int(cust['Age']),
                'Location': cust['Location'], 'Tenure': int(cust['Tenure_Months']),
                'Avg_Spend': int(cust['Avg_Monthly_Spend']),
                'Last_Spend': int(cust['Last_Month_Spend']),
                'Transactions': int(cust['Num_Transactions']),
                'Days_Since': int(cust['Days_Since_Last_Purchase']),
                'Tickets': int(cust['Support_Tickets']),
            })

    # ── Input form ──
    with st.form("analyzer_form"):
        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            gender = st.selectbox("Gender", ["Male", "Female"],
                                  index=0 if defaults['Gender'] == "Male" else 1)
            age = st.number_input("Age", 18, 100, defaults['Age'])
        with fc2:
            loc_options = ["Urban", "Suburban", "Rural"]
            try:
                loc_idx = loc_options.index(defaults['Location'])
            except ValueError:
                loc_idx = 0
            location = st.selectbox("Location", loc_options, index=loc_idx)
            tenure = st.number_input("Tenure (Months)", 0, 200, defaults['Tenure'])
        with fc3:
            avg_spend = st.number_input("Avg Monthly Spend ($)", 0, 10000, defaults['Avg_Spend'])
            last_spend = st.number_input("Last Month Spend ($)", 0, 10000, defaults['Last_Spend'])
        with fc4:
            transactions = st.number_input("Num Transactions", 0, 500, defaults['Transactions'])
            days_since = st.number_input("Days Since Last Purchase", 0, 365, defaults['Days_Since'])

        support_tickets = st.number_input("Support Tickets", 0, 50, defaults['Tickets'])
        submit = st.form_submit_button("🔍  Analyze Customer", use_container_width=True)

    if submit:
        raw_data = {
            'Age': age, 'Gender': gender, 'Tenure_Months': tenure,
            'Avg_Monthly_Spend': avg_spend, 'Last_Month_Spend': last_spend,
            'Num_Transactions': transactions, 'Days_Since_Last_Purchase': days_since,
            'Support_Tickets': support_tickets, 'Location': location,
        }

        try:
            processed = preprocess_input(raw_data, models)
            cluster = models['kmeans'].predict(processed)[0]
            churn = models['classification'].predict(processed)[0]
            next_spend = models['regression'].predict(processed)[0]

            # Health score
            row_for_score = {**raw_data, 'Predicted_Churn': churn}
            h_score = health_score(row_for_score)

            st.divider()
            title_suffix = f" — {selected_customer_id}" if selected_customer_id else ""
            st.subheader(f"Analysis Results{title_suffix}")

            # ── KPI cards ──
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                render_metric_card("Customer Segment", f"Group {cluster}", "K-Means cluster")
            with k2:
                risk_label = "⚠️ High Risk" if churn == 1 else "✅ Low Risk"
                render_metric_card("Churn Risk", risk_label, "Classification model")
            with k3:
                render_metric_card("Predicted Next Spend", f"${next_spend:.2f}", "Regression model")
            with k4:
                badge = "badge-green" if h_score >= 65 else ("badge-yellow" if h_score >= 40 else "badge-red")
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">HEALTH SCORE</div>
                    <div class="metric-value">{h_score}/100</div>
                    <span class="score-badge {badge}">{'Healthy' if h_score >= 65 else ('At Risk' if h_score >= 40 else 'Critical')}</span>
                </div>
                """, unsafe_allow_html=True)

            st.divider()

            # ── ML Insight Explainer ──
            st.subheader("📐 Explainable ML Insights")
            exp1, exp2 = st.columns(2)

            with exp1:
                st.markdown("**Key Factors Driving This Prediction**")
                factors = []
                if days_since > 60:
                    factors.append(("🔴", "High recency gap", f"{days_since} days since last purchase"))
                elif days_since < 15:
                    factors.append(("🟢", "Recent activity", f"Purchased {days_since} days ago"))
                if support_tickets >= 4:
                    factors.append(("🔴", "High support load", f"{support_tickets} tickets filed"))
                elif support_tickets <= 1:
                    factors.append(("🟢", "Low support needs", f"Only {support_tickets} ticket(s)"))
                if avg_spend > df['Avg_Monthly_Spend'].quantile(0.75):
                    factors.append(("🟢", "High spender", f"${avg_spend} vs avg ${df['Avg_Monthly_Spend'].mean():.0f}"))
                elif avg_spend < df['Avg_Monthly_Spend'].quantile(0.25):
                    factors.append(("🔴", "Low spender", f"${avg_spend} vs avg ${df['Avg_Monthly_Spend'].mean():.0f}"))
                if tenure > 24:
                    factors.append(("🟢", "Loyal customer", f"{tenure} months tenure"))
                elif tenure < 6:
                    factors.append(("🟡", "New customer", f"Only {tenure} months tenure"))

                if not factors:
                    factors.append(("🟡", "Average profile", "No extreme indicators detected"))

                for icon, title, detail in factors:
                    st.markdown(f"{icon} **{title}** — {detail}")

            with exp2:
                st.markdown("**Customer vs. Population**")
                metrics_compare = {
                    'Avg Spend': (avg_spend, df['Avg_Monthly_Spend'].mean()),
                    'Tenure': (tenure, df['Tenure_Months'].mean()),
                    'Transactions': (transactions, df['Num_Transactions'].mean()),
                    'Support Tickets': (support_tickets, df['Support_Tickets'].mean()),
                }
                compare_df = pd.DataFrame([
                    {'Metric': k, 'Customer': v[0], 'Population Avg': round(v[1], 1)}
                    for k, v in metrics_compare.items()
                ])
                fig_compare = go.Figure()
                fig_compare.add_trace(go.Bar(
                    name='Customer', x=compare_df['Metric'], y=compare_df['Customer'],
                    marker_color='#07b6d5',
                ))
                fig_compare.add_trace(go.Bar(
                    name='Population Avg', x=compare_df['Metric'], y=compare_df['Population Avg'],
                    marker_color='#334155',
                ))
                fig_compare.update_layout(**PLOTLY_LAYOUT, barmode='group', height=300)
                st.plotly_chart(fig_compare, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")


# ═══════════════════════════════════════════════════════════════
#  PAGE 3 – SEGMENT ANALYTICS
# ═══════════════════════════════════════════════════════════════
elif page == "📈 Segment Analytics":
    st.markdown('<div class="page-header">Segment Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Comparative intelligence across customer groups</div>', unsafe_allow_html=True)

    n_segments = df['Segment'].nunique()
    segment_cols = st.columns(n_segments)

    for i, seg_id in enumerate(sorted(df['Segment'].unique())):
        seg_df = df[df['Segment'] == seg_id]
        with segment_cols[i]:
            render_metric_card(f"Group {seg_id}", f"{len(seg_df)} customers", "")
            st.markdown(f"**Avg Spend:** ${seg_df['Avg_Monthly_Spend'].mean():.2f}")
            st.markdown(f"**Avg Tenure:** {seg_df['Tenure_Months'].mean():.1f} mo")
            st.markdown(f"**Churn Rate:** {seg_df['Predicted_Churn'].mean()*100:.1f}%")
            st.markdown(f"**Avg Tickets:** {seg_df['Support_Tickets'].mean():.1f}")

    st.divider()

    # ── Comparative Charts ──
    ch1, ch2 = st.columns(2)

    with ch1:
        st.subheader("Spend Distribution by Segment")
        fig_box = px.box(
            df, x=df['Segment'].apply(lambda s: f'Group {s}'),
            y='Avg_Monthly_Spend', color=df['Segment'].apply(lambda s: f'Group {s}'),
            color_discrete_sequence=COLOR_SEQ,
        )
        fig_box.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=380,
                              xaxis_title='Segment', yaxis_title='Avg Monthly Spend ($)')
        st.plotly_chart(fig_box, use_container_width=True)

    with ch2:
        st.subheader("Age Distribution by Segment")
        fig_age = px.histogram(
            df, x='Age', color=df['Segment'].apply(lambda s: f'Group {s}'),
            barmode='overlay', color_discrete_sequence=COLOR_SEQ, opacity=0.7,
        )
        fig_age.update_layout(**PLOTLY_LAYOUT, height=380,
                              xaxis_title='Age', yaxis_title='Count')
        st.plotly_chart(fig_age, use_container_width=True)

    ch3, ch4 = st.columns(2)

    with ch3:
        st.subheader("Location Breakdown per Segment")
        loc_seg = df.groupby([df['Segment'].apply(lambda s: f'Group {s}'), 'Location']).size().reset_index(name='Count')
        fig_loc = px.bar(loc_seg, x='Segment', y='Count', color='Location',
                         barmode='group', color_discrete_sequence=COLOR_SEQ)
        fig_loc.update_layout(**PLOTLY_LAYOUT, height=380)
        st.plotly_chart(fig_loc, use_container_width=True)

    with ch4:
        st.subheader("Churn Rate by Segment")
        churn_seg = df.groupby('Segment')['Predicted_Churn'].mean().reset_index()
        churn_seg['Segment'] = churn_seg['Segment'].apply(lambda s: f'Group {s}')
        churn_seg['Churn_Pct'] = churn_seg['Predicted_Churn'] * 100
        fig_churn_seg = px.bar(churn_seg, x='Segment', y='Churn_Pct',
                                color='Segment', color_discrete_sequence=COLOR_SEQ,
                                text_auto='.1f')
        fig_churn_seg.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=380,
                                     yaxis_title='Churn Rate (%)')
        st.plotly_chart(fig_churn_seg, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
#  PAGE 4 – CHURN INTELLIGENCE
# ═══════════════════════════════════════════════════════════════
elif page == "⚠️ Churn Intelligence":
    st.markdown('<div class="page-header">Churn Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Risk analysis & prioritized retention outreach</div>', unsafe_allow_html=True)

    at_risk = df[df['Predicted_Churn'] == 1].copy()
    safe = df[df['Predicted_Churn'] == 0]

    # ── KPI row ──
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_metric_card("Customers at Risk", f"{len(at_risk)}", f"{len(at_risk)/len(df)*100:.1f}% of total")
    with k2:
        render_metric_card("Revenue at Risk", f"${at_risk['Avg_Monthly_Spend'].sum():,.0f}", "Monthly exposure")
    with k3:
        render_metric_card("Avg Spend (at risk)", f"${at_risk['Avg_Monthly_Spend'].mean():.2f}" if len(at_risk) else "–", "")
    with k4:
        render_metric_card("Avg Tickets (at risk)", f"{at_risk['Support_Tickets'].mean():.1f}" if len(at_risk) else "–", "")

    st.divider()

    # ── Risk Factors ──
    rf1, rf2 = st.columns(2)

    with rf1:
        st.subheader("Support Tickets vs Churn")
        fig_tickets = px.histogram(
            df, x='Support_Tickets',
            color=df['Predicted_Churn'].map({0: 'Safe', 1: 'At Risk'}),
            barmode='overlay', color_discrete_map={'Safe': '#06d6a0', 'At Risk': '#f87171'},
            opacity=0.75,
        )
        fig_tickets.update_layout(**PLOTLY_LAYOUT, height=350, xaxis_title='Support Tickets',
                                   yaxis_title='Count')
        st.plotly_chart(fig_tickets, use_container_width=True)

    with rf2:
        st.subheader("Days Since Last Purchase vs Churn")
        fig_days = px.histogram(
            df, x='Days_Since_Last_Purchase',
            color=df['Predicted_Churn'].map({0: 'Safe', 1: 'At Risk'}),
            barmode='overlay', color_discrete_map={'Safe': '#06d6a0', 'At Risk': '#f87171'},
            opacity=0.75,
        )
        fig_days.update_layout(**PLOTLY_LAYOUT, height=350,
                                xaxis_title='Days Since Last Purchase', yaxis_title='Count')
        st.plotly_chart(fig_days, use_container_width=True)

    st.divider()

    # ── Prioritized Retention Table ──
    st.subheader("🚨 Prioritized Retention Outreach")
    if len(at_risk) > 0:
        at_risk['Health_Score'] = at_risk.apply(health_score, axis=1)
        at_risk_sorted = at_risk.sort_values('Health_Score', ascending=True)

        display_cols = ['CustomerID', 'Health_Score', 'Avg_Monthly_Spend', 'Last_Month_Spend',
                        'Days_Since_Last_Purchase', 'Support_Tickets', 'Segment', 'Predicted_Next_Spend']
        st.dataframe(
            at_risk_sorted[display_cols].reset_index(drop=True),
            use_container_width=True,
            height=400,
            column_config={
                'CustomerID': st.column_config.TextColumn('Customer'),
                'Health_Score': st.column_config.ProgressColumn('Health', min_value=0, max_value=100, format="%d"),
                'Avg_Monthly_Spend': st.column_config.NumberColumn('Avg Spend', format="$%.2f"),
                'Last_Month_Spend': st.column_config.NumberColumn('Last Spend', format="$%.2f"),
                'Predicted_Next_Spend': st.column_config.NumberColumn('Pred. Spend', format="$%.2f"),
            },
        )
    else:
        st.success("No customers currently flagged as at-risk. 🎉")


# ═══════════════════════════════════════════════════════════════
#  PAGE 5 – REVENUE FORECASTING
# ═══════════════════════════════════════════════════════════════
elif page == "💰 Revenue Forecasting":
    st.markdown('<div class="page-header">Revenue Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Advanced spend prediction, cohort analysis & trend forecasting</div>', unsafe_allow_html=True)

    # ── KPI Row ──
    total_current = df['Avg_Monthly_Spend'].sum()
    total_predicted = df['Predicted_Next_Spend'].sum()
    delta_pct = ((total_predicted - total_current) / total_current) * 100 if total_current else 0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_metric_card("Current Monthly Revenue", f"${total_current:,.0f}", "Based on Avg Spend")
    with k2:
        render_metric_card("Predicted Next Month", f"${total_predicted:,.0f}",
                           f"{'↑' if delta_pct >= 0 else '↓'} {abs(delta_pct):.1f}% change")
    with k3:
        render_metric_card("Revenue Delta", f"${total_predicted - total_current:,.0f}",
                           "Predicted change")
    with k4:
        avg_pred = df['Predicted_Next_Spend'].mean()
        render_metric_card("Avg Predicted Spend", f"${avg_pred:.2f}", "Per customer")

    st.divider()

    # ── Actual vs Predicted Spend ──
    rv1, rv2 = st.columns(2)

    with rv1:
        st.subheader("Actual vs Predicted Spend Distribution")
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Histogram(x=df['Avg_Monthly_Spend'], name='Current Avg Spend',
                                        marker_color='#07b6d5', opacity=0.7, nbinsx=30))
        fig_cmp.add_trace(go.Histogram(x=df['Predicted_Next_Spend'], name='Predicted Next Spend',
                                        marker_color='#06d6a0', opacity=0.7, nbinsx=30))
        fig_cmp.update_layout(**PLOTLY_LAYOUT, barmode='overlay', height=380,
                               xaxis_title='Spend ($)', yaxis_title='Count')
        st.plotly_chart(fig_cmp, use_container_width=True)

    with rv2:
        st.subheader("Spend Scatter: Current vs Predicted")
        fig_scatter = px.scatter(
            df, x='Avg_Monthly_Spend', y='Predicted_Next_Spend',
            color=df['Segment'].apply(lambda s: f'Group {s}'),
            color_discrete_sequence=COLOR_SEQ, opacity=0.6,
            hover_data=['CustomerID'],
        )
        # Add 45-degree reference line
        max_val = max(df['Avg_Monthly_Spend'].max(), df['Predicted_Next_Spend'].max())
        fig_scatter.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val], mode='lines',
            line=dict(dash='dash', color='#475569', width=1),
            name='Parity Line',
        ))
        fig_scatter.update_layout(**PLOTLY_LAYOUT, height=380,
                                   xaxis_title='Current Avg Spend ($)',
                                   yaxis_title='Predicted Next Spend ($)')
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.divider()

    # ── Cohort Analysis ──
    st.subheader("Cohort Analysis — Revenue by Tenure & Segment")
    df_cohort = df.copy()
    df_cohort['Tenure_Bucket'] = pd.cut(
        df_cohort['Tenure_Months'],
        bins=[0, 6, 12, 24, 48, 200],
        labels=['0-6m', '6-12m', '12-24m', '24-48m', '48m+'],
    )
    cohort_data = df_cohort.groupby(
        [df_cohort['Segment'].apply(lambda s: f'Group {s}'), 'Tenure_Bucket'], observed=True,
    ).agg(
        Avg_Spend=('Avg_Monthly_Spend', 'mean'),
        Predicted_Spend=('Predicted_Next_Spend', 'mean'),
        Count=('CustomerID', 'count'),
    ).reset_index()

    fig_cohort = px.bar(
        cohort_data, x='Tenure_Bucket', y='Predicted_Spend', color='Segment',
        barmode='group', color_discrete_sequence=COLOR_SEQ,
        text_auto='.0f',
    )
    fig_cohort.update_layout(**PLOTLY_LAYOUT, height=400,
                              xaxis_title='Tenure Cohort', yaxis_title='Predicted Avg Spend ($)')
    st.plotly_chart(fig_cohort, use_container_width=True)

    # ── Feature Impact on Revenue ──
    st.subheader("Feature Correlation with Spend")
    numeric_cols = ['Age', 'Tenure_Months', 'Num_Transactions',
                    'Days_Since_Last_Purchase', 'Support_Tickets', 'Last_Month_Spend']
    corr_values = df[numeric_cols + ['Avg_Monthly_Spend']].corr()['Avg_Monthly_Spend'].drop('Avg_Monthly_Spend')
    corr_df = corr_values.reset_index()
    corr_df.columns = ['Feature', 'Correlation']
    corr_df = corr_df.sort_values('Correlation', ascending=True)

    fig_corr = px.bar(
        corr_df, x='Correlation', y='Feature', orientation='h',
        color='Correlation',
        color_continuous_scale=[[0, '#f87171'], [0.5, '#fbbf24'], [1, '#06d6a0']],
    )
    fig_corr.update_layout(**PLOTLY_LAYOUT, height=320, xaxis_title='Correlation with Avg Monthly Spend')
    st.plotly_chart(fig_corr, use_container_width=True)
