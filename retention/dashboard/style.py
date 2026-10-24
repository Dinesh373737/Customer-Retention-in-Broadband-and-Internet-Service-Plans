"""Shared visual language for the Streamlit dashboard."""
import streamlit as st

BLUE = "#3269c7"
TEAL = "#0d9488"
CORAL = "#e57769"
PALETTE = [BLUE, TEAL, CORAL, "#9c80c9", "#d7ac58"]


def inject_style():
    st.markdown("""
    <style>
    html,body,[class*="css"],.stApp {font-family:'DM Sans',sans-serif;color:#20334d;}
    .stApp {background:#f5f8fc;}
    h1,h2,h3 {font-family:'Manrope',sans-serif;letter-spacing:-.035em;}
    h1 {font-size:2.4rem!important;font-weight:800!important;margin-bottom:.3rem!important;}
    h2 {font-size:1.4rem!important;} h3 {font-size:1.1rem!important;}
    [data-testid="stMainBlockContainer"] {padding:2.5rem 3rem 3rem;max-width:1580px;}
    [data-testid="stSidebar"] {background:#fff;border-right:1px solid #e1e8f1;}
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap:.7rem;}
    [data-testid="stMetric"] {background:#fff;border:1px solid #e1e8f1;border-radius:13px;padding:1.1rem 1.2rem;min-height:112px;}
    [data-testid="stMetricLabel"] {color:#6d7d91;font-size:.78rem;text-transform:uppercase;letter-spacing:.05em;}
    [data-testid="stMetricValue"] {font-family:'Manrope',sans-serif;font-size:1.9rem;font-weight:800;color:#213958;}
    [data-testid="stPlotlyChart"] {border:1px solid #e4eaf2;border-radius:12px;background:white;overflow:hidden;}
    [data-testid="stDataFrame"] {border-radius:10px;}
    [data-testid="stAlert"] {border-radius:10px;}
    .eyebrow {font-size:.69rem;color:#087f83;letter-spacing:.15em;font-weight:700;margin-bottom:8px;}
    .subline {color:#708198;font-size:.95rem;line-height:1.6;margin-bottom:1.5rem;}
    .brand {font-family:'Manrope',sans-serif;font-size:1.48rem;font-weight:800;letter-spacing:-.06em;color:#183e64;margin-top:.6rem;}
    .brand span {color:#0d9488;}.brand-note {font-size:.68rem;color:#8a98aa;letter-spacing:.1em;margin:2px 0 25px;}
    .status {background:#edf7f7;border:1px solid #cfe9e6;border-radius:8px;padding:10px 12px;font-size:.73rem;color:#336a69;margin:12px 0;line-height:1.6;}
    .insight {background:#fff;border:1px solid #e2e9f2;border-left:3px solid #0d9488;border-radius:9px;padding:14px 17px;margin-bottom:10px;line-height:1.6;font-size:.9rem;}
    .step {background:#fff;border:1px solid #dce6f1;border-radius:10px;padding:16px;margin:8px 0;min-height:130px;}
    .step-num {font-size:.75rem;color:#0d9488;font-weight:700;}.step h4 {font-size:1rem;margin:8px 0;color:#23466b;}.step p{font-size:.8rem;color:#708198;margin:0;}
    .footer {border-top:1px solid #dce5f0;margin-top:2rem;padding-top:1rem;font-size:.72rem;color:#7e8ca0;}
    .stButton>button[kind="primary"] {background:#167c91;border-color:#167c91;}
    @media(max-width:900px){[data-testid="stMainBlockContainer"]{padding:1.5rem 1rem;}h1{font-size:1.9rem!important;}}
    </style>
    """, unsafe_allow_html=True)


def chart(fig, height=340):
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=20, r=20, t=50, b=30),
        font=dict(family="DM Sans, Arial", color="#40546d", size=12),
        title_font=dict(size=15, color="#203c5b"), paper_bgcolor="white",
        plot_bgcolor="white", colorway=PALETTE, legend_title_text="",
        legend=dict(orientation="h", y=-.22, x=0),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#edf1f6", zeroline=False)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar":False})


def page_header(title, subtitle, label="IBM TELCO SAMPLE · HISTORICAL ANALYSIS"):
    st.markdown(f'<div class="eyebrow">{label}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="subline">{subtitle}</div>', unsafe_allow_html=True)
