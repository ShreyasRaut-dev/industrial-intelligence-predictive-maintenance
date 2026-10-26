from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Safe imports from local modules
try:
    from train_model import (
        load_artifacts, encode_features, normalize_ai4i_columns,
        NUMERIC_FEATURES, MODEL_FEATURES, TARGET, CATEGORICAL_FEATURE,
        REPORT_DIR, DATA_PATH, METRICS_PATH, HISTORY_PATH, HISTORY_PLOT_PATH,
        CONFUSION_PATH, CONFUSION_PLOT_PATH, ROC_PLOT_PATH, PR_PLOT_PATH,
        IMPORTANCE_PATH, CLASS_REPORT_PATH
    )
    from custom_trainer import (
        inspect_dataset_for_training, train_custom_neural_network,
        list_custom_models, predict_with_custom_model, CUSTOM_MODELS_DIR
    )
except Exception:
    pass

# Page Configuration - Wide layout with collapsed sidebar
st.set_page_config(
    page_title="AUREL // Industrial Fault Detection",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# ------------------------------------------------------------------------------
if "has_loaded" not in st.session_state:
    st.session_state.has_loaded = False
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "show_mode_modal" not in st.session_state:
    st.session_state.show_mode_modal = False
if "selected_nav" not in st.session_state:
    st.session_state.selected_nav = "⚡ Mode A: Pretrained Model"

# ------------------------------------------------------------------------------
# 1. ENTRANCE LOADING ANIMATION (Shown once on website entrance)
# ------------------------------------------------------------------------------
if not st.session_state.has_loaded:
    splash_placeholder = st.empty()
    with splash_placeholder.container():
        st.markdown("""
        <div style="max-width: 520px; margin: 120px auto 30px auto; text-align: center; background: #FFFFFF; border: 1px solid #E7E7E4; border-radius: 16px; padding: 48px 36px; box-shadow: 0 8px 32px rgba(0,0,0,0.06);">
            <div style="width: 68px; height: 68px; margin: 0 auto 20px auto; border-radius: 16px; background: linear-gradient(135deg, #E99A35 0%, #D98622 100%); color: #1A1A1A; font-size: 34px; display: flex; align-items: center; justify-content: center; box-shadow: 0 6px 18px rgba(233,154,53,0.38);">🏭</div>
            <h1 style="color: #1A1A1A; margin: 0; font-weight: 900; font-size: 1.85rem; letter-spacing: -0.02em;">INDUSTRIAL INTELLIGENCE</h1>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(1.2)

    splash_placeholder.empty()
    st.session_state.has_loaded = True
    st.session_state.show_mode_modal = False
    st.rerun()

# ------------------------------------------------------------------------------
# 2. DYNAMIC THEME ENGINE (Vibrant Light & Dark Modes)
# ------------------------------------------------------------------------------
is_dark = st.session_state.dark_mode

# Theme Variables based on Industrial Modern Design System:
# Light: Soft cool-toned gray-blue (Tailwind slate-50). Dark: Deep midnight blue / rich charcoal (slate-900).
bg_color = "#0F172A" if is_dark else "#F8FAFC"
card_bg = "#1E293B" if is_dark else "#FFFFFF"
card_border = "rgba(51, 65, 85, 0.8)" if is_dark else "rgba(226, 232, 240, 0.9)"
border_color = "#334155" if is_dark else "#E2E8F0"

# Typography
text_primary = "#F8FAFC" if is_dark else "#0F172A"
text_muted = "#94A3B8" if is_dark else "#64748B"

# Form Surfaces & Controls
input_bg = "#1E293B" if is_dark else "#F1F5F9"
input_border = "#475569" if is_dark else "#CBD5E1"
stat_chip_bg = "rgba(30, 41, 59, 0.75)" if is_dark else "rgba(241, 245, 249, 0.95)"

# Status Accents (Emerald Green, Amber, Crimson Red)
badge_bg = "rgba(16, 185, 129, 0.14)" if is_dark else "rgba(16, 185, 129, 0.12)"
badge_border = "rgba(16, 185, 129, 0.35)" if is_dark else "rgba(16, 185, 129, 0.3)"
badge_text = "#10B981" if is_dark else "#059669"

# Drop Shadows (Crisp white with soft indigo drop shadow in Light mode; layered navy depth in Dark mode)
card_shadow = "0 10px 25px -5px rgba(0, 0, 0, 0.4), 0 8px 10px -6px rgba(0, 0, 0, 0.25)" if is_dark else "0 10px 25px -5px rgba(99, 102, 241, 0.08), 0 8px 10px -6px rgba(99, 102, 241, 0.04)"
assessment_shadow = "0 16px 36px -6px rgba(99, 102, 241, 0.3), 0 0 20px rgba(6, 182, 212, 0.18)" if is_dark else "0 16px 36px -6px rgba(99, 102, 241, 0.14), 0 0 16px rgba(99, 102, 241, 0.08)"

# Background Mesh Gradients (Vibrant Indigo, Cyan, Violet & Emerald Ambient Orbs)
if is_dark:
    bg_gradient = """
        background-color: #0B1120 !important;
        background-image: 
            radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.24) 0px, transparent 45%),
            radial-gradient(at 100% 0%, rgba(6, 182, 212, 0.22) 0px, transparent 45%),
            radial-gradient(at 50% 35%, rgba(139, 92, 246, 0.14) 0px, transparent 55%),
            radial-gradient(at 90% 85%, rgba(16, 185, 129, 0.15) 0px, transparent 45%),
            radial-gradient(at 10% 90%, rgba(245, 158, 11, 0.11) 0px, transparent 40%) !important;
    """
    card_bg_glass = "rgba(30, 41, 59, 0.88)"
else:
    bg_gradient = """
        background-color: #F8FAFC !important;
        background-image: 
            radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 45%),
            radial-gradient(at 100% 0%, rgba(6, 182, 212, 0.14) 0px, transparent 45%),
            radial-gradient(at 50% 35%, rgba(139, 92, 246, 0.08) 0px, transparent 55%),
            radial-gradient(at 90% 85%, rgba(16, 185, 129, 0.10) 0px, transparent 45%),
            radial-gradient(at 10% 90%, rgba(245, 158, 11, 0.07) 0px, transparent 40%) !important;
    """
    card_bg_glass = "rgba(255, 255, 255, 0.92)"

st.markdown(f"""
<style>
/* Smooth Global Transitions */
* {{
    transition: background-color 0.3s ease-in-out, border-color 0.3s ease-in-out, color 0.3s ease-in-out, box-shadow 0.3s ease-in-out;
}}

/* App Canvas & Layout */
header[data-testid="stHeader"] {{
    display: none !important;
    height: 0 !important;
}}

.stApp {{
    {bg_gradient}
    background-attachment: fixed !important;
    background-size: cover !important;
    color: {text_primary} !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}}

.block-container {{
    padding-top: 2.2rem !important;
    padding-bottom: 3rem !important;
    max-width: 1360px !important;
}}

h1, h2, h3, h4, h5, h6 {{
    color: {text_primary} !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em !important;
}}

p, span, label {{
    color: {text_primary};
}}

.stCaption, small, [data-testid="stCaptionContainer"] {{
    color: {text_muted} !important;
}}

/* Top Branding Bar */
.brand-header {{
    background: {card_bg_glass};
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid {card_border};
    border-radius: 16px;
    padding: 18px 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.9rem;
    box-shadow: {card_shadow};
}}
.brand-left {{
    display: flex;
    align-items: center;
    gap: 16px;
}}
.brand-logo {{
    width: 50px;
    height: 50px;
    border-radius: 14px;
    background: linear-gradient(135deg, #6366F1 0%, #06B6D4 100%);
    color: #FFFFFF;
    font-size: 26px;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.3);
}}
.brand-title {{
    font-size: 1.35rem;
    font-weight: 900;
    color: {text_primary};
    letter-spacing: -0.02em;
    margin: 0;
}}
.brand-sub {{
    font-size: 0.83rem;
    color: {text_muted};
    margin-top: 3px;
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: 8px;
}}
.brand-right {{
    display: flex;
    align-items: center;
    gap: 10px;
}}
.stat-pill {{
    background: {stat_chip_bg};
    border: 1px solid {border_color};
    padding: 7px 16px;
    border-radius: 20px;
    font-size: 0.8rem;
    color: {text_muted};
    font-weight: 600;
}}
.stat-pill b {{
    color: {text_primary};
}}
.brand-badge {{
    background: {badge_bg};
    border: 1px solid {badge_border};
    color: {badge_text};
    padding: 7px 16px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 800;
    letter-spacing: 0.04em;
    display: inline-flex;
    align-items: center;
    gap: 8px;
}}
.pulse-dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: {badge_text};
    display: inline-block;
    box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.6);
    animation: statusPulse 2s infinite cubic-bezier(0.66, 0, 0, 1);
}}
@keyframes statusPulse {{
    to {{ box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }}
}}

/* Segmented Top Navigation Bar (Mode A, Mode B, Custom) */
div[data-testid="stPills"] {{
    background: {card_bg_glass} !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid {border_color} !important;
    border-radius: 14px !important;
    padding: 6px !important;
    margin-bottom: 1.6rem !important;
    box-shadow: {card_shadow} !important;
    display: flex !important;
    width: 100% !important;
    overflow-x: auto !important;
}}

div[data-testid="stPills"] > div {{
    display: flex !important;
    width: 100% !important;
    justify-content: space-between !important;
    gap: 8px !important;
    min-width: 600px !important;
}}

div[data-testid="stPills"] button {{
    flex: 1 1 0 !important;
    min-width: fit-content !important;
    text-align: center !important;
    background: transparent !important;
    border: 1px solid rgba(99, 102, 241, 0.25) !important;
    color: {text_muted} !important;
    font-weight: 600 !important;
    font-size: 0.93rem !important;
    border-radius: 10px !important;
    padding: 10px 16px !important;
    white-space: nowrap !important;
    overflow: visible !important;
    text-overflow: clip !important;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    cursor: pointer !important;
}}

div[data-testid="stPills"] button p,
div[data-testid="stPills"] button span,
div[data-testid="stPills"] button div {{
    overflow: visible !important;
    text-overflow: clip !important;
    white-space: nowrap !important;
    font-size: 0.93rem !important;
    text-align: center !important;
    margin: 0 !important;
    padding: 0 !important;
}}

div[data-testid="stPills"] button:hover {{
    color: #06B6D4 !important;
    background: rgba(99, 102, 241, 0.08) !important;
    border-color: rgba(6, 182, 212, 0.45) !important;
    transform: translateY(-1px) !important;
}}

div[data-testid="stPills"] button[aria-selected="true"] {{
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 50%, #06B6D4 100%) !important;
    color: #FFFFFF !important;
    font-weight: 800 !important;
    border: none !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.42) !important;
    transform: translateY(-1px) !important;
}}

/* Surface Panels & Cards */
.aurel-panel {{
    background: {card_bg_glass};
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid {card_border};
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 1.5rem;
    box-shadow: {card_shadow};
}}

.aurel-card {{
    background: {card_bg_glass};
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid {card_border};
    border-radius: 14px;
    padding: 22px;
    height: 100%;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: {card_shadow};
}}
.aurel-card:hover {{
    border-color: #6366F1;
    transform: translateY(-2px);
    box-shadow: 0 12px 28px rgba(99, 102, 241, 0.16);
}}

/* Metric Cards */
div[data-testid="stMetric"] {{
    background: {card_bg_glass} !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid {card_border} !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
    box-shadow: {card_shadow} !important;
    transition: all 0.3s ease-in-out !important;
}}
div[data-testid="stMetric"]:hover {{
    border-color: rgba(99, 102, 241, 0.5) !important;
    transform: translateY(-2px) !important;
}}
div[data-testid="stMetricLabel"] p {{
    color: {text_muted} !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}}
div[data-testid="stMetricValue"] {{
    color: {text_primary} !important;
    font-size: 1.65rem !important;
    font-weight: 900 !important;
}}

/* Primary Action Buttons */
button[kind="primary"], .stButton > button[kind="primary"], [data-testid="stBaseButton-primary"] {{
    background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 0.7rem 1.5rem !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4) !important;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    letter-spacing: 0.01em !important;
}}
button[kind="primary"]:hover, .stButton > button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {{
    transform: translateY(-2px) !important;
    background: linear-gradient(135deg, #4F46E5 0%, #4338CA 100%) !important;
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.55) !important;
    color: #FFFFFF !important;
}}

/* Secondary Buttons ("Switch Mode", "Dark Mode", "Download") with Colored Soft Borders */
button[kind="secondary"], .stButton > button, [data-testid="stBaseButton-secondary"], [data-testid="stDownloadButton"] > button {{
    background: {card_bg} !important;
    color: {text_primary} !important;
    border: 1px solid rgba(99, 102, 241, 0.35) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.2rem !important;
    box-shadow: {card_shadow} !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}}
button[kind="secondary"]:hover, .stButton > button:hover, [data-testid="stBaseButton-secondary"]:hover, [data-testid="stDownloadButton"] > button:hover {{
    border-color: #6366F1 !important;
    background: rgba(99, 102, 241, 0.12) !important;
    color: {'#818CF8' if is_dark else '#4F46E5'} !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.22) !important;
}}

/* Input Forms (Left Column) */
div[data-baseweb="input"], div[data-baseweb="select"] > div {{
    background: {input_bg} !important;
    border: 1px solid {input_border} !important;
    border-radius: 10px !important;
    color: {text_primary} !important;
    transition: all 0.3s ease-in-out !important;
}}
div[data-baseweb="input"]:focus-within, div[data-baseweb="select"]:focus-within > div {{
    border-color: {'#06B6D4' if is_dark else '#6366F1'} !important;
    box-shadow: 0 0 0 3px {'rgba(6, 182, 212, 0.25)' if is_dark else 'rgba(99, 102, 241, 0.2)'} !important;
}}

/* Number Stepper +/- Buttons */
div[data-testid="stNumberInput"] button {{
    border-radius: 6px !important;
    background: transparent !important;
    color: {text_muted} !important;
    transition: all 0.2s ease-in-out !important;
}}
div[data-testid="stNumberInput"] button:hover {{
    color: #06B6D4 !important;
    background: rgba(6, 182, 212, 0.15) !important;
}}

/* Shielded Notification Cards */
.notice-card {{
    border-radius: 12px;
    padding: 16px 20px;
    margin: 1rem 0;
    border: 1px solid transparent;
    box-shadow: {card_shadow};
}}
.notice-warning {{
    background: {'rgba(245, 158, 11, 0.1)' if is_dark else '#FFFDF7'};
    border-color: {'rgba(245, 158, 11, 0.3)' if is_dark else '#F7DCB0'};
    border-left: 4px solid #F59E0B;
}}
.notice-success {{
    background: {'rgba(16, 185, 129, 0.1)' if is_dark else '#F0F7F2'};
    border-color: {'rgba(16, 185, 129, 0.3)' if is_dark else '#C6E2D0'};
    border-left: 4px solid #10B981;
}}
.notice-danger {{
    background: {'rgba(239, 68, 68, 0.1)' if is_dark else '#FDF4F4'};
    border-color: {'rgba(239, 68, 68, 0.3)' if is_dark else '#F5C2C0'};
    border-left: 4px solid #EF4444;
}}
.notice-title {{
    font-weight: 800;
    font-size: 0.95rem;
    margin-bottom: 4px;
    color: {text_primary};
}}
.notice-body {{
    font-size: 0.88rem;
    color: {text_muted};
    line-height: 1.5;
}}

/* Diagnostic Assessment Card (Right Column - The Focal Point) */
.meter-wrapper {{
    background: {card_bg};
    border: 1px solid {card_border};
    border-radius: 16px;
    padding: 26px;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: {assessment_shadow};
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}}
.meter-wrapper::before {{
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 5px;
    background: linear-gradient(90deg, #6366F1 0%, #8B5CF6 50%, #06B6D4 100%);
}}
.meter-value {{
    font-size: 3.5rem;
    font-weight: 900;
    letter-spacing: -0.03em;
    margin: 10px 0;
}}
.meter-bar {{
    width: 100%;
    height: 14px;
    background: {'#334155' if is_dark else '#E2E8F0'};
    border-radius: 7px;
    overflow: hidden;
    margin: 16px 0;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.06);
}}
.meter-fill-normal {{
    background: linear-gradient(90deg, #10B981 0%, #06B6D4 100%);
    height: 100%;
    border-radius: 7px;
    box-shadow: 0 0 12px rgba(16, 185, 129, 0.5);
    transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}}
.meter-fill-fail {{
    background: linear-gradient(90deg, #F59E0B 0%, #EF4444 100%);
    height: 100%;
    border-radius: 7px;
    box-shadow: 0 0 12px rgba(239, 68, 68, 0.5);
    transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}}

/* Dataframe and Tables */
div[data-testid="stDataFrame"] {{
    border: 1px solid {border_color} !important;
    border-radius: 12px !important;
    background-color: {card_bg} !important;
}}
div[data-testid="stExpander"] {{
    background-color: {card_bg} !important;
    border: 1px solid {border_color} !important;
    border-radius: 12px !important;
}}
pre, code {{
    background-color: {input_bg} !important;
    border: 1px solid {border_color} !important;
    border-radius: 8px !important;
    color: {text_primary} !important;
}}

/* Modal Dialog Window */
div[data-testid="stDialog"] > div:first-child {{
    background-color: {card_bg} !important;
    border: 1px solid {card_border} !important;
    border-radius: 16px !important;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.35) !important;
}}
div[role="dialog"] {{
    background-color: {card_bg} !important;
    border: 1px solid {card_border} !important;
    border-radius: 16px !important;
}}
</style>
""", unsafe_allow_html=True)


# Error Shielding Function
def show_friendly_notice(title: str, message: str, hint: str = None, level: str = "warning"):
    css_class = "notice-warning" if level == "warning" else ("notice-danger" if level == "danger" else "notice-success")
    icon = "⚠️" if level == "warning" else ("🚨" if level == "danger" else "✅")
    hint_html = f"<div style='margin-top:8px;font-size:0.83rem;color:{text_muted};'><b>Actionable Hint:</b> {hint}</div>" if hint else ""
    st.markdown(f"""
    <div class="notice-card {css_class}">
        <div class="notice-title">{icon} {title}</div>
        <div class="notice-body">{message}</div>
        {hint_html}
    </div>
    """, unsafe_allow_html=True)


# Load Baseline AI4I Model with Shielding
@st.cache_resource
def get_safe_ai4i_artifacts():
    try:
        return load_artifacts(), True, ""
    except Exception as exc:
        return None, False, str(exc)

ai4i_data, baseline_loaded, load_err = get_safe_ai4i_artifacts()
if baseline_loaded:
    model, scaler, metrics, df_ai4i = ai4i_data
else:
    model, scaler, metrics, df_ai4i = None, None, {}, pd.DataFrame()


# ------------------------------------------------------------------------------
# 3. INTERACTIVE MODE SELECTION DIALOG (Opens upon entry or via Switch Mode)
# ------------------------------------------------------------------------------
@st.dialog("⚙️ Select Operational Workflow", width="large")
def select_mode_dialog():
    st.markdown(f"""
    <div style="margin-bottom: 22px; text-align: left;">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
            <span style="font-size: 1.25rem;">🏭</span>
            <span style="font-size: 0.82rem; font-weight: 700; color: #E99A35; letter-spacing: 0.08em; text-transform: uppercase;">AUREL Industrial Intelligence</span>
        </div>
        <h3 style="margin: 0 0 6px 0; font-size: 1.35rem; font-weight: 800; color: {text_primary};">Choose Diagnostic Mode</h3>
        <p style="color: {text_muted}; font-size: 0.9rem; margin: 0; line-height: 1.5;">
            Select an operational mode to begin real-time fault diagnosis or train a new deep learning model:
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown(f"""
        <div class="aurel-card" style="display: flex; flex-direction: column; justify-content: space-between; height: 350px; margin-bottom: 16px;">
            <div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <span style="background: rgba(233,154,53,0.15); color: #E99A35; font-size: 0.75rem; font-weight: 700; padding: 3px 10px; border-radius: 20px; border: 1px solid rgba(233,154,53,0.3);">PRE-TRAINED MODEL</span>
                    <span style="font-size: 0.76rem; color: {text_muted}; font-weight: 600;">10,000 AI4I Records</span>
                </div>
                <h4 style="margin: 0 0 8px 0; font-size: 1.15rem; font-weight: 800; color: {text_primary};">⚡ Mode A: AI4I Predictor</h4>
                <p style="color: {text_muted}; font-size: 0.86rem; line-height: 1.5; margin-bottom: 14px; min-height: 48px;">
                    Deploy the verified TensorFlow neural network for immediate fault inference, probability estimation, and diagnostic curve inspection.
                </p>
                <div style="background: {input_bg}; border: 1px solid {border_color}; border-radius: 8px; padding: 12px; margin-bottom: 8px;">
                    <div style="font-size: 0.82rem; color: {text_primary}; margin-bottom: 6px; display: flex; align-items: center; gap: 6px;">
                        <span style="color: #E99A35; font-weight: 700;">✓</span> <b>Milling Sensors:</b> Air/Proc Temp, Speed, Torque, Wear
                    </div>
                    <div style="font-size: 0.82rem; color: {text_primary}; margin-bottom: 6px; display: flex; align-items: center; gap: 6px;">
                        <span style="color: #E99A35; font-weight: 700;">✓</span> <b>Dual Inference:</b> Single Machine Dial + Batch CSV
                    </div>
                    <div style="font-size: 0.82rem; color: {text_primary}; display: flex; align-items: center; gap: 6px;">
                        <span style="color: #E99A35; font-weight: 700;">✓</span> <b>Integrated Curves:</b> ROC, PR, & Confusion Matrix
                    </div>
                </div>
            </div>
            <div style="font-size: 0.78rem; color: {text_muted}; text-align: center; font-weight: 600;">
                Pre-calibrated Threshold &tau; = 0.90 &bull; ROC-AUC 0.973
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("⚡ Launch Mode A (Predict)", type="primary", use_container_width=True, key="dlg_btn_a"):
            st.session_state.selected_nav = "⚡ Mode A: Pretrained Model"
            st.session_state.show_mode_modal = False
            st.rerun()

    with col_b:
        st.markdown(f"""
        <div class="aurel-card" style="display: flex; flex-direction: column; justify-content: space-between; height: 350px; margin-bottom: 16px;">
            <div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <span style="background: rgba(233,154,53,0.15); color: #E99A35; font-size: 0.75rem; font-weight: 700; padding: 3px 10px; border-radius: 20px; border: 1px solid rgba(233,154,53,0.3);">CUSTOM TRAINING</span>
                    <span style="font-size: 0.76rem; color: {text_muted}; font-weight: 600;">Automated Pipeline</span>
                </div>
                <h4 style="margin: 0 0 8px 0; font-size: 1.15rem; font-weight: 800; color: {text_primary};">🧪 Mode B: Train New Model</h4>
                <p style="color: {text_muted}; font-size: 0.86rem; line-height: 1.5; margin-bottom: 14px; min-height: 48px;">
                    Upload custom machinery datasets (water pumps, compressors, turbines) and build an isolated, dedicated neural network.
                </p>
                <div style="background: {input_bg}; border: 1px solid {border_color}; border-radius: 8px; padding: 12px; margin-bottom: 8px;">
                    <div style="font-size: 0.82rem; color: {text_primary}; margin-bottom: 6px; display: flex; align-items: center; gap: 6px;">
                        <span style="color: #E99A35; font-weight: 700;">✓</span> <b>Any Machinery:</b> Pumps, Turbines, Bearings, Engines
                    </div>
                    <div style="font-size: 0.82rem; color: {text_primary}; margin-bottom: 6px; display: flex; align-items: center; gap: 6px;">
                        <span style="color: #E99A35; font-weight: 700;">✓</span> <b>Quality Guard:</b> Binary target & class imbalance audit
                    </div>
                    <div style="font-size: 0.82rem; color: {text_primary}; display: flex; align-items: center; gap: 6px;">
                        <span style="color: #E99A35; font-weight: 700;">✓</span> <b>Zero Overwrite:</b> Isolated storage in dedicated folder
                    </div>
                </div>
            </div>
            <div style="font-size: 0.78rem; color: {text_muted}; text-align: center; font-weight: 600;">
                Includes Built-in Water Pump Sample Dataset
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🧪 Launch Mode B (Train)", type="primary", use_container_width=True, key="dlg_btn_b"):
            st.session_state.selected_nav = "🧪 Mode B: Train New Dataset"
            st.session_state.show_mode_modal = False
            st.rerun()

    st.markdown(f"""
    <div style="text-align: center; color: {text_muted}; font-size: 0.82rem; margin-top: 18px; border-top: 1px solid {border_color}; padding-top: 12px;">
        💡 <b>Workflow Navigation:</b> You can switch between modes at any time from the top navigation bar or the <b>⚙️ Switch Mode</b> button.
    </div>
    """, unsafe_allow_html=True)

# Automatically trigger the modal if flagged
if st.session_state.show_mode_modal:
    select_mode_dialog()


# ------------------------------------------------------------------------------
# 4. TOP BRAND BAR WITH MODE SWITCHER & DARK MODE TOGGLE
# ------------------------------------------------------------------------------
top_col1, top_col2 = st.columns([2.5, 1.5])

with top_col1:
    st.markdown(f"""
    <div class="brand-left" style="margin-bottom:12px;">
        <div class="brand-logo">🏭</div>
        <div>
            <div class="brand-title">INDUSTRIAL INTELLIGENCE</div>
            <div class="brand-sub">
                <span style="color:{'#38BDF8' if is_dark else '#0284C7'}; font-weight:700;">Deep Learning Neural Network (MLP)</span>
                <span>•</span>
                <span>Predictive Maintenance Platform</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with top_col2:
    btn_m_col, btn_t_col = st.columns([1, 1])
    with btn_m_col:
        if st.button("⚙️ Switch Mode", key="btn_switch_mode", use_container_width=True):
            st.session_state.show_mode_modal = True
            st.rerun()
    with btn_t_col:
        dark_label = "☀️ Light Mode" if is_dark else "🌙 Dark Mode"
        if st.button(dark_label, key="btn_dark_toggle", use_container_width=True):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

# Status Chips Bar
st.markdown(f"""
<div style="display:flex; justify-content:flex-end; gap:10px; margin-bottom:14px; flex-wrap:wrap;">
    <div class="stat-pill">Model: <b style="color:{'#818CF8' if is_dark else '#4F46E5'};">TensorFlow MLP</b></div>
    <div class="stat-pill">Cutoff: <b style="color:{'#38BDF8' if is_dark else '#0284C7'};">τ = 0.90</b></div>
    <div class="brand-badge"><span class="pulse-dot"></span> SYSTEM ONLINE</div>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 5. SEGMENTED TOP NAVIGATION (st.pills)
# ------------------------------------------------------------------------------
NAV_OPTIONS = [
    "⚡ Mode A: Pretrained Model",
    "🧪 Mode B: Train New Dataset",
    "🔮 Custom Model Inference"
]

# Ensure valid default
default_nav = st.session_state.selected_nav if st.session_state.selected_nav in NAV_OPTIONS else "⚡ Mode A: Pretrained Model"

selected_nav = st.pills(
    label="Navigation Menu",
    options=NAV_OPTIONS,
    default=default_nav,
    label_visibility="collapsed"
)

# Update state if changed by pill click
if selected_nav != st.session_state.selected_nav:
    st.session_state.selected_nav = selected_nav


# Prediction Helper for AI4I Model
def safe_predict_ai4i(input_df: pd.DataFrame):
    try:
        norm_df = normalize_ai4i_columns(input_df)
        encoded = encode_features(norm_df)
        scaled = scaler.transform(encoded)
        raw_probs = model.predict(scaled, verbose=0).ravel()
        thresh = float(metrics.get("threshold", 0.90))
        preds = (raw_probs >= thresh).astype(int)

        out = input_df.copy()
        out["Failure Probability (%)"] = np.round(raw_probs * 100.0, 2)
        out["Predicted Condition"] = np.where(preds == 1, "Failure", "Normal")
        return out, None
    except Exception as e:
        return None, str(e)


# ==============================================================================
# SECTION 1: MODE A - PRETRAINED MODEL PREDICTION
# ==============================================================================
if selected_nav == "⚡ Mode A: Pretrained Model":
    st.markdown("### ⚡ Mode A: Predict with Pretrained AI4I Neural Network")
    st.caption("Standard milling machine parameters: Air Temp [K], Process Temp [K], Speed [rpm], Torque [Nm], Tool Wear [min], Type.")

    if not baseline_loaded:
        show_friendly_notice(
            "Baseline Model Not Initialized",
            "The baseline neural network weights are currently being prepared or need to be trained.",
            "Run 'python industrial_fault_detection_dl/train_model.py' in PowerShell to generate the baseline model.",
            level="warning"
        )
        st.stop()

    tab_single, tab_batch = st.tabs(["🕹️ Single Machine Simulator", "📂 Batch CSV Prediction"])

    # TAB 1: Single Prediction
    with tab_single:
        st.markdown("<div class='aurel-panel'>", unsafe_allow_html=True)
        col_in, col_res = st.columns([1, 1], gap="large")

        with col_in:
            st.markdown("#### Machine Operating Sensors")
            machine_type = st.selectbox("Machine Type (Variant)", ["L", "M", "H"], index=1, help="L: Low Quality, M: Medium, H: Heavy Duty")
            air_temp = st.number_input("Air Temperature [K]", min_value=290.0, max_value=310.0, value=300.0, step=0.1)
            proc_temp = st.number_input("Process Temperature [K]", min_value=300.0, max_value=320.0, value=310.0, step=0.1)
            speed = st.number_input("Rotational Speed [rpm]", min_value=1000, max_value=3200, value=1500, step=10)
            torque = st.number_input("Torque [Nm]", min_value=3.0, max_value=90.0, value=40.0, step=0.5)
            wear = st.number_input("Tool Wear [min]", min_value=0, max_value=300, value=100, step=1)
            predict_btn = st.button("⚡ Run Fault Diagnostic", type="primary", use_container_width=True)

        with col_res:
            st.markdown("#### Diagnostic Assessment")
            if predict_btn:
                sample_data = pd.DataFrame([{
                    "Type": machine_type,
                    "Air temperature [K]": air_temp,
                    "Process temperature [K]": proc_temp,
                    "Rotational speed [rpm]": speed,
                    "Torque [Nm]": torque,
                    "Tool wear [min]": wear,
                }])
                res_df, err = safe_predict_ai4i(sample_data)
                st.session_state["single_diagnostic_result"] = (res_df, err)

            diag_result = st.session_state.get("single_diagnostic_result", None)
            if diag_result is not None:
                res_df, err = diag_result
                if err:
                    show_friendly_notice("Prediction Error", "Could not compute diagnostic for these inputs.", err, "warning")
                else:
                    prob = float(res_df["Failure Probability (%)"].iloc[0])
                    cond = res_df["Predicted Condition"].iloc[0]
                    bar_class = "meter-fill-fail" if cond == "Failure" else "meter-fill-normal"
                    val_color = "#EF4444" if cond == "Failure" else "#10B981"
                    glow_style = f"text-shadow: 0 0 28px rgba(239, 68, 68, 0.55);" if cond == "Failure" and is_dark else (f"text-shadow: 0 0 28px rgba(16, 185, 129, 0.55);" if is_dark else "text-shadow: 0 2px 10px rgba(16, 185, 129, 0.25);")

                    st.markdown(f"""
                    <div class="meter-wrapper">
                        <div style="font-size:0.85rem;color:{text_muted};font-weight:700;text-transform:uppercase;letter-spacing:0.06em;">Estimated Failure Probability</div>
                        <div class="meter-value" style="color:{val_color};{glow_style}">{prob:.1f}%</div>
                        <div class="meter-bar">
                            <div class="{bar_class}" style="width:{min(max(prob, 2.0), 100.0)}%;"></div>
                        </div>
                        <div style="font-size:1.15rem;font-weight:800;color:{val_color};margin-top:8px;">
                            {'🚨 CRITICAL: FAILURE RISK DETECTED' if cond == 'Failure' else '✅ NORMAL OPERATIONAL PARAMETERS'}
                        </div>
                        <div style="font-size:0.82rem;color:{text_muted};margin-top:6px;font-weight:500;">Decision Cutoff: &tau; = {metrics.get('threshold', 0.90):.2f} • Model: Deep MLP</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="meter-wrapper" style="padding:48px 24px;">
                    <div style="font-size:2.4rem;margin-bottom:12px;filter:drop-shadow(0 0 14px rgba(6, 182, 212, 0.5));">⚡</div>
                    <div style="color:{text_primary};font-size:1.05rem;font-weight:800;margin-bottom:6px;">Diagnostic Assessment Pending</div>
                    <div style="color:{text_muted};font-size:0.88rem;line-height:1.5;">Set sensor operating measurements on the left and click <b style="color:{'#818CF8' if is_dark else '#4F46E5'};">⚡ Run Fault Diagnostic</b> to generate live probability estimates.</div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # DIAGNOSTIC EVALUATION & PERFORMANCE CURVES (Rendered immediately below fault rate)
        if diag_result is not None and diag_result[1] is None:
            st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class='aurel-panel'>
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                    <h3 style="margin:0; font-size:1.25rem; color:{text_primary};">📊 Neural Network Evaluation & Diagnostic Curves</h3>
                    <div class="brand-badge"><span class="pulse-dot"></span> 2,000 TEST MACHINES</div>
                </div>
                <p style="color:{text_muted}; font-size:0.88rem; margin-bottom:16px;">
                    Quantitative benchmark metrics and validation curves from the trained TensorFlow model, evaluated on an untouched held-out test split of 2,000 machines.
                </p>
            """, unsafe_allow_html=True)

            # Metrics Cards Row
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Accuracy", f"{metrics.get('accuracy', 0.976)*100:.1f}%")
            c2.metric("Precision", f"{metrics.get('precision', 0.614)*100:.1f}%")
            c3.metric("Recall", f"{metrics.get('recall', 0.794)*100:.1f}%")
            c4.metric("F1-Score", f"{metrics.get('f1', 0.692)*100:.1f}%")
            c5.metric("ROC-AUC", f"{metrics.get('roc_auc', 0.973):.3f}")
            c6.metric("Threshold", f"{metrics.get('threshold', 0.90):.2f}")

            st.markdown("<div style='height:15px;'></div>", unsafe_allow_html=True)
            col_p1, col_p2 = st.columns([1.2, 1], gap="large")

            with col_p1:
                st.markdown("#### Training & Validation Convergence")
                if HISTORY_PLOT_PATH.exists():
                    st.image(str(HISTORY_PLOT_PATH), use_container_width=True)

                st.markdown("#### Precision-Recall Curve")
                if PR_PLOT_PATH.exists():
                    st.image(str(PR_PLOT_PATH), use_container_width=True)

            with col_p2:
                st.markdown("#### Test Confusion Matrix (2,000 Machines)")
                if CONFUSION_PLOT_PATH.exists():
                    st.image(str(CONFUSION_PLOT_PATH), use_container_width=True)

                st.markdown("#### ROC Curve (AUC = 0.973)")
                if ROC_PLOT_PATH.exists():
                    st.image(str(ROC_PLOT_PATH), use_container_width=True)

            # Permutation Feature Importance Ranking
            if IMPORTANCE_PATH.exists():
                st.markdown("#### Permutation Feature Importance Ranking")
                imp_df = pd.read_csv(IMPORTANCE_PATH).sort_values("f1_drop_when_shuffled")

                fig, ax = plt.subplots(figsize=(8, 3.5), facecolor=card_bg)
                ax.set_facecolor(card_bg)
                ax.barh(imp_df["feature"], imp_df["f1_drop_when_shuffled"], color="#E99A35", edgecolor="#D98622", height=0.55)
                ax.set_title("Drop in Test F1 When Feature is Shuffled", color=text_primary, fontsize=10, fontweight="bold", pad=8)
                ax.set_xlabel("F1 Impact", color=text_muted, fontsize=9)
                for s in ax.spines.values(): s.set_color(border_color)
                ax.tick_params(colors=text_muted, labelsize=8)
                ax.grid(True, linestyle="-", alpha=0.4, color=border_color, axis="x")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            st.markdown("</div>", unsafe_allow_html=True)

    # TAB 2: Batch Prediction
    with tab_batch:
        st.markdown("<div class='aurel-panel'>", unsafe_allow_html=True)
        st.markdown("#### Bulk Telemetry CSV Ingestion")
        st.write("Upload a CSV containing new machine records for batch prediction.")

        sample_csv = Path("industrial_fault_detection_dl/data/sample_test_records.csv")
        btn_c1, btn_c2 = st.columns([1, 2])
        with btn_c1:
            if sample_csv.exists():
                st.download_button(
                    "📥 Download Sample Template CSV",
                    data=sample_csv.read_bytes(),
                    file_name="sample_test_records.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        uploaded_csv = st.file_uploader("Drop machine telemetry CSV here", type=["csv"], key="mode_a_uploader")
        active_batch = None

        if uploaded_csv is not None:
            try:
                active_batch = pd.read_csv(uploaded_csv)
            except Exception as e:
                show_friendly_notice("CSV Read Error", "The uploaded file could not be parsed as a valid CSV.", str(e), "warning")
        elif st.button("📂 Load Pre-Packaged Test Records (10 Machines)", key="load_sample_a_btn"):
            if sample_csv.exists():
                active_batch = pd.read_csv(sample_csv)

        if active_batch is not None:
            st.markdown(f"**Loaded Telemetry Table ({len(active_batch)} rows):**")
            st.dataframe(active_batch.head(8), use_container_width=True)

            with st.spinner("Running neural network inference across batch..."):
                results, err = safe_predict_ai4i(active_batch)

            if err:
                show_friendly_notice("Batch Schema Mismatch", "The uploaded CSV does not match the 6 required AI4I feature columns.", err, "warning")
            else:
                total_n = len(results)
                fail_n = int((results["Predicted Condition"] == "Failure").sum())
                norm_n = total_n - fail_n

                st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
                sm1, sm2, sm3 = st.columns(3)
                sm1.metric("Total Records Processed", f"{total_n:,}")
                sm2.metric("Normal Machines", f"{norm_n:,} ({norm_n/total_n*100:.1f}%)")
                sm3.metric("Failure Alerts", f"{fail_n:,} ({fail_n/total_n*100:.1f}%)")

                st.dataframe(results, use_container_width=True)
                st.download_button(
                    "📥 Export Annotated Predictions CSV",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name="batch_predictions_ai4i.csv",
                    mime="text/csv",
                    type="primary"
                )
        st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# SECTION 3: MODE B - TRAIN NEW DATASET
# ==============================================================================
elif selected_nav == "🧪 Mode B: Train New Dataset":
    st.markdown("### 🧪 Mode B: Train a Neural Network for a Custom Machine Dataset")
    st.caption("Upload any labelled sensor dataset (e.g. pumps, compressors, turbines) to train a dedicated Deep Learning model in isolation.")

    st.markdown("<div class='aurel-panel'>", unsafe_allow_html=True)

    # Initialize training dataframe in session state
    if "custom_train_df" not in st.session_state:
        st.session_state.custom_train_df = None

    sample_pump_path = Path("industrial_fault_detection_dl/data/sample_custom_pump_dataset.csv")
    sample_comp_path = Path("industrial_fault_detection_dl/data/compressor_training_dataset.csv")
    sample_turb_path = Path("industrial_fault_detection_dl/data/wind_turbine_training_dataset.csv")
    sample_motor_path = Path("industrial_fault_detection_dl/data/electric_motor_training_dataset.csv")

    st.markdown("##### 📁 Step 1: Upload or Select Training Dataset")
    uploaded_train = st.file_uploader("Upload labelled training dataset (CSV)", type=["csv"], key="train_uploader")

    if uploaded_train is not None:
        try:
            st.session_state.custom_train_df = pd.read_csv(uploaded_train)
        except Exception as e:
            show_friendly_notice("CSV Parse Notice", "Could not read uploaded CSV file.", str(e), "warning")

    st.markdown("<div style='font-size:0.85rem; color:" + text_muted + "; margin-bottom:6px;'>Or quickly load one of our bundled industrial training datasets:</div>", unsafe_allow_html=True)
    c_s1, c_s2, c_s3, c_s4 = st.columns(4)
    with c_s1:
        if st.button("📂 Pump (145)", key="btn_load_pump", use_container_width=True):
            if sample_pump_path.exists():
                st.session_state.custom_train_df = pd.read_csv(sample_pump_path)
    with c_s2:
        if st.button("📂 Compressor (260)", key="btn_load_comp", use_container_width=True):
            if sample_comp_path.exists():
                st.session_state.custom_train_df = pd.read_csv(sample_comp_path)
    with c_s3:
        if st.button("📂 Turbine (260)", key="btn_load_turb", use_container_width=True):
            if sample_turb_path.exists():
                st.session_state.custom_train_df = pd.read_csv(sample_turb_path)
    with c_s4:
        if st.button("📂 Motor (280)", key="btn_load_motor", use_container_width=True):
            if sample_motor_path.exists():
                st.session_state.custom_train_df = pd.read_csv(sample_motor_path)

    active_train_df = st.session_state.custom_train_df

    if active_train_df is not None:
        st.markdown("<hr style='border:none; border-top:1px solid " + border_color + "; margin:18px 0;'>", unsafe_allow_html=True)
        st.markdown("##### 📊 Step 2: Dataset Preview & Column Structure")
        st.dataframe(active_train_df.head(6), use_container_width=True)

        cols = list(active_train_df.columns)
        default_target_idx = 0
        for i, c in enumerate(cols):
            if any(k in c.lower() for k in ["fault", "fail", "target", "label", "status", "class"]):
                default_target_idx = i
                break

        col_cfg1, col_cfg2 = st.columns([1, 2])
        with col_cfg1:
            target_col = st.selectbox("Target Column (Failure Label)", cols, index=default_target_idx)

        # Audit dataset
        is_valid, issues, meta = inspect_dataset_for_training(active_train_df, target_col)

        if not is_valid:
            for iss in issues:
                show_friendly_notice("Dataset Unsuitable for Supervised Training", iss, "Ensure the target has exactly 2 classes and enough samples.", "warning")
        else:
            show_friendly_notice("Dataset Validated Successfully", f"Detected binary target with classes: {meta['target_classes']}", level="success")

            with col_cfg2:
                candidate_feats = [c for c in cols if c != target_col]
                default_feats = [c for c in candidate_feats if not any(k in c.lower() for k in ["id", "udi", "index", "serial", "unnamed"])]
                selected_features = st.multiselect("Selected Input Features", candidate_feats, default=default_feats)

            if selected_features:
                st.markdown("##### ⚙️ Step 3: Model Identifier & Training")
                custom_model_name = st.text_input(
                    "Custom Model Name / Identifier",
                    value="custom_machinery_v1",
                    help="Name for saving the neural network into models/custom_models/"
                )
                epochs = 35
                batch_size = 32
                st.caption("⚡ *Neural network hyperparameters configured automatically: Batch Size = 32, Max Epochs = 35 (with Early Stopping & Dynamic Class Weighting).*")

                st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
                if st.button("🚀 Train Deep Neural Network on New Dataset", type="primary", use_container_width=True):
                    with st.spinner("Training neural network with stratified split and class weights..."):
                        try:
                            config, save_dir = train_custom_neural_network(
                                df=active_train_df,
                                target_col=target_col,
                                feature_cols=selected_features,
                                model_name=custom_model_name,
                                epochs=epochs,
                                batch_size=batch_size
                            )

                            st.session_state.selected_custom_model = config["model_name"]
                            show_friendly_notice("Training Successful", f"Model '{config['model_name']}' trained and saved safely! It is now ready for inference.", level="success")

                            if st.button(f"🔮 Open '{config['model_name']}' in Custom Model Inference Tab", type="primary", key="goto_custom_tab_btn"):
                                st.session_state.selected_nav = "🔮 Custom Model Inference"
                                st.rerun()

                            m = config["metrics"]
                            st.markdown("#### Held-Out Test Evaluation Results")
                            tm1, tm2, tm3, tm4 = st.columns(4)
                            tm1.metric("Test Accuracy", f"{m['accuracy']*100:.1f}%")
                            tm2.metric("Precision", f"{m['precision']*100:.1f}%")
                            tm3.metric("Recall", f"{m['recall']*100:.1f}%")
                            tm4.metric("F1-Score", f"{m['f1']*100:.1f}%")

                            # High-quality Matplotlib convergence curves adapted to theme
                            hist = config["history"]
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=card_bg)
                            for a in (ax1, ax2):
                                a.set_facecolor(card_bg)
                                for s in a.spines.values(): s.set_color(border_color)
                                a.tick_params(colors=text_muted, labelsize=8)
                                a.grid(True, linestyle="-", alpha=0.4, color=border_color)

                            ax1.plot(hist["epoch"], hist["loss"], color="#E99A35", lw=2.2, label="Train Loss")
                            if hist.get("val_loss"):
                                ax1.plot(hist["epoch"], hist["val_loss"], color=text_muted, lw=1.8, linestyle="--", label="Val Loss")
                            ax1.set_title("Cross-Entropy Loss Across Epochs", color=text_primary, fontsize=10, fontweight="bold", pad=8)
                            ax1.set_xlabel("Epoch", color=text_muted, fontsize=9)
                            ax1.legend(frameon=True, facecolor=card_bg, edgecolor=border_color, fontsize=8)

                            ax2.plot(hist["epoch"], hist["accuracy"], color="#E99A35", lw=2.2, label="Train Accuracy")
                            if hist.get("val_accuracy"):
                                ax2.plot(hist["epoch"], hist["val_accuracy"], color=text_muted, lw=1.8, linestyle="--", label="Val Accuracy")
                            ax2.set_title("Accuracy Convergence Across Epochs", color=text_primary, fontsize=10, fontweight="bold", pad=8)
                            ax2.set_xlabel("Epoch", color=text_muted, fontsize=9)
                            ax2.legend(frameon=True, facecolor=card_bg, edgecolor=border_color, fontsize=8)

                            fig.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)
                        except Exception as e:
                            show_friendly_notice("Training Interrupted", "Could not complete training with these parameters.", str(e), "danger")
    st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# SECTION 4: CUSTOM MODEL INFERENCE
# ==============================================================================
elif selected_nav == "🔮 Custom Model Inference":
    st.markdown("### 🔮 Predict with a Saved Custom Neural Network")
    st.caption("Select any previously trained custom model and run batch predictions on matching machine telemetry.")

    st.markdown("<div class='aurel-panel'>", unsafe_allow_html=True)
    custom_models = list_custom_models()

    if not custom_models:
        show_friendly_notice("No Custom Models Found", "You haven't trained any custom models yet.", "Navigate to '🧪 Mode B: Train New Dataset' to train your first model.", level="warning")
    else:
        names = [m["model_name"] for m in custom_models]
        default_idx = 0
        if "selected_custom_model" in st.session_state and st.session_state.selected_custom_model in names:
            default_idx = names.index(st.session_state.selected_custom_model)
        chosen_name = st.selectbox("Choose Saved Model", names, index=default_idx)
        cfg = next(m for m in custom_models if m["model_name"] == chosen_name)
        cm_metrics = cfg.get("metrics", {})

        # Display Top Performance Metrics Cards
        st.markdown("#### 📊 Model Evaluation Metrics (Held-Out Test Set)")
        c_m1, c_m2, c_m3, c_m4, c_m5 = st.columns(5)
        acc_val = cm_metrics.get("accuracy", 0) * 100
        prec_val = cm_metrics.get("precision", 0) * 100
        rec_val = cm_metrics.get("recall", 0) * 100
        f1_val = cm_metrics.get("f1", 0) * 100
        auc_val = cm_metrics.get("roc_auc")
        auc_str = f"{auc_val:.3f}" if auc_val is not None else "N/A"

        c_m1.metric("Accuracy", f"{acc_val:.1f}%")
        c_m2.metric("Precision", f"{prec_val:.1f}%")
        c_m3.metric("Recall", f"{rec_val:.1f}%")
        c_m4.metric("F1-Score", f"{f1_val:.1f}%")
        c_m5.metric("ROC-AUC", auc_str)

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown(f"**Target Field:** `{cfg['target_col']}`")
            st.markdown(f"**Required Features:** `{', '.join(cfg['feature_cols'])}`")
        with col_m2:
            st.markdown(f"**Decision Cutoff:** `τ = {cfg.get('threshold', 0.5):.2f}`")
            st.markdown(f"**Created At:** `{cfg.get('created_at', 'Saved')}`")

        # Visualizations Expander (Confusion Matrix + Training Curves)
        with st.expander("📈 View Model Visualizations & Diagnostic Curves", expanded=False):
            viz_col1, viz_col2 = st.columns(2)
            
            # Confusion Matrix
            raw_cm = cm_metrics.get("confusion_matrix")
            if raw_cm and len(raw_cm) == 2:
                with viz_col1:
                    fig_cm, ax_cm = plt.subplots(figsize=(4.5, 3.5), facecolor=card_bg)
                    ax_cm.set_facecolor(card_bg)
                    sns.heatmap(
                        raw_cm, annot=True, fmt="d", cmap="Blues" if not is_dark else "YlOrBr",
                        cbar=False, ax=ax_cm,
                        xticklabels=["Normal", "Fault"], yticklabels=["Normal", "Fault"],
                        annot_kws={"size": 11, "weight": "bold"}
                    )
                    ax_cm.set_title("Test Confusion Matrix", color=text_primary, fontsize=10, fontweight="bold", pad=8)
                    ax_cm.set_xlabel("Predicted Label", color=text_muted, fontsize=9)
                    ax_cm.set_ylabel("Actual Label", color=text_muted, fontsize=9)
                    ax_cm.tick_params(colors=text_muted, labelsize=8)
                    fig_cm.tight_layout()
                    st.pyplot(fig_cm)
                    plt.close(fig_cm)

            # Training Curves
            hist = cfg.get("history")
            if hist and hist.get("loss"):
                with viz_col2:
                    fig_tc, ax_tc = plt.subplots(figsize=(5.2, 3.5), facecolor=card_bg)
                    ax_tc.set_facecolor(card_bg)
                    for s in ax_tc.spines.values(): s.set_color(border_color)
                    ax_tc.tick_params(colors=text_muted, labelsize=8)
                    ax_tc.grid(True, linestyle="-", alpha=0.3, color=border_color)

                    ax_tc.plot(hist["epoch"], hist["loss"], color="#E99A35", lw=2, label="Train Loss")
                    if hist.get("val_loss"):
                        ax_tc.plot(hist["epoch"], hist["val_loss"], color=text_muted, lw=1.6, linestyle="--", label="Val Loss")
                    ax_tc.set_title("Training vs Validation Loss", color=text_primary, fontsize=10, fontweight="bold", pad=8)
                    ax_tc.set_xlabel("Epoch", color=text_muted, fontsize=8)
                    ax_tc.legend(frameon=True, facecolor=card_bg, edgecolor=border_color, fontsize=8)
                    fig_tc.tight_layout()
                    st.pyplot(fig_tc)
                    plt.close(fig_tc)

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        st.markdown(f"#### 🧪 Run Inference with '{chosen_name}'")

        if "custom_infer_df" not in st.session_state:
            st.session_state.custom_infer_df = None

        uploaded_c_batch = st.file_uploader(f"Upload CSV for '{chosen_name}'", type=["csv"], key="c_infer_uploader")

        if uploaded_c_batch is not None:
            try:
                st.session_state.custom_infer_df = pd.read_csv(uploaded_c_batch)
            except Exception as e:
                show_friendly_notice("CSV Read Error", "Could not parse file.", str(e), "warning")
        elif "pump" in chosen_name.lower():
            pump_p = Path("industrial_fault_detection_dl/data/sample_custom_pump_dataset.csv")
            if pump_p.exists():
                if st.button("📂 Load Sample Test Records for this Pump Model", key="load_custom_sample_btn"):
                    st.session_state.custom_infer_df = pd.read_csv(pump_p).drop(columns=["Fault"], errors="ignore").head(12)
        elif "compressor" in chosen_name.lower() or "compresser" in chosen_name.lower():
            comp_p = Path("industrial_fault_detection_dl/data/compressor_test_records.csv")
            if comp_p.exists():
                if st.button("📂 Load Compressor Test Records (15 Machines)", key="load_custom_comp_btn"):
                    st.session_state.custom_infer_df = pd.read_csv(comp_p).head(15)
        elif "turbine" in chosen_name.lower():
            turb_p = Path("industrial_fault_detection_dl/data/wind_turbine_test_records.csv")
            if turb_p.exists():
                if st.button("📂 Load Wind Turbine Test Records (15 Turbines)", key="load_custom_turb_btn"):
                    st.session_state.custom_infer_df = pd.read_csv(turb_p).head(15)
        elif "motor" in chosen_name.lower():
            mot_p = Path("industrial_fault_detection_dl/data/electric_motor_test_records.csv")
            if mot_p.exists():
                if st.button("📂 Load Electric Motor Test Records (25 Motors)", key="load_custom_motor_btn"):
                    st.session_state.custom_infer_df = pd.read_csv(mot_p).head(25)

        c_input_df = st.session_state.custom_infer_df

        if c_input_df is not None:
            st.markdown(f"**Loaded Telemetry Records ({len(c_input_df)} rows):**")
            st.dataframe(c_input_df.head(6), use_container_width=True)

            if st.button(f"⚡ Run Predictions using '{chosen_name}'", type="primary", use_container_width=True):
                try:
                    c_preds, _ = predict_with_custom_model(chosen_name, c_input_df)
                    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
                    
                    # Summary metrics
                    total_c = len(c_preds)
                    fault_c = int((c_preds["Predicted_Condition"] == "Failure").sum()) if "Predicted_Condition" in c_preds.columns else int((c_preds["Predicted_Class"].astype(str).isin(["1", "Failure", "fault"])).sum())
                    norm_c = total_c - fault_c
                    thresh_val = float(c_preds["Threshold_Applied"].iloc[0]) * 100 if "Threshold_Applied" in c_preds.columns else 50.0
                    
                    sm1, sm2, sm3 = st.columns(3)
                    sm1.metric("Total Records Processed", f"{total_c:,}")
                    sm2.metric("Normal Operating Machines", f"{norm_c:,} ({norm_c/total_c*100:.1f}%)")
                    sm3.metric("Fault Alerts Detected", f"{fault_c:,} ({fault_c/total_c*100:.1f}%)")

                    st.info(f"💡 **Decision Rule Applied:** Any machine with a **Failure Score ≥ {thresh_val:.1f}%** (the model's optimal F1 threshold) is diagnosed as **Failure** for early preventive maintenance.")

                    st.success(f"✓ Inference Complete! Evaluated {total_c} records using '{chosen_name}'.")
                    st.dataframe(c_preds, use_container_width=True)
                    st.download_button(
                        "📥 Download Custom Predictions CSV",
                        data=c_preds.to_csv(index=False).encode("utf-8"),
                        file_name=f"{chosen_name}_predictions.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    show_friendly_notice("Feature Compatibility Notice", "The uploaded CSV does not contain the features expected by this custom model.", str(e), "warning")
    st.markdown("</div>", unsafe_allow_html=True)



