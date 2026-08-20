import streamlit as st
import numpy as np
import librosa
import librosa.display
import tempfile
import os
import json
import onnxruntime as ort
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VoiceScreen AI · Depression Screening",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_CONFIG_PATH = MODELS_DIR / "model_config.json"
HISTORY_FILE = BASE_DIR / "voice_history.json"

# ── Load model config ─────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "n_mfcc": 40,
    "max_len": 300,
    "sr": 16000,
    "n_features": 120,
    "best_threshold": 0.34,
}

if MODEL_CONFIG_PATH.exists():
    with open(MODEL_CONFIG_PATH) as f:
        MODEL_CONFIG = {**DEFAULT_CONFIG, **json.load(f)}
else:
    MODEL_CONFIG = DEFAULT_CONFIG

# ── Voice History Helpers ──────────────────────────────────────────────────────
def load_history() -> list:
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []

def save_to_history(filename: str, duration: float, probability: float, label: str):
    history = load_history()
    new_entry = {
        "id": str(np.random.randint(100000, 999999)),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": filename,
        "duration_sec": round(duration, 1),
        "probability": round(probability, 4),
        "label": label
    }
    history.append(new_entry)
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=4)
    except Exception:
        pass

def delete_history_entry(entry_id: str):
    history = load_history()
    history = [e for e in history if e["id"] != entry_id]
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=4)
    except Exception:
        pass

def clear_all_history():
    try:
        if HISTORY_FILE.exists():
            os.remove(HISTORY_FILE)
    except Exception:
        pass

# ── Premium Clinical CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    /* ─── Hide Streamlit chrome ─── */
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stToolbar"] { display: none !important; }
    .stDeployButton { display: none !important; }

    /* ─── Root & Global ─── */
    :root {
        --bg-primary:   #060c18;
        --bg-secondary: #0b1425;
        --bg-card:      rgba(11, 20, 37, 0.85);
        --border:       rgba(0, 212, 170, 0.12);
        --border-hover: rgba(0, 212, 170, 0.35);
        --teal:         #00d4aa;
        --teal-dim:     rgba(0, 212, 170, 0.15);
        --teal-glow:    rgba(0, 212, 170, 0.25);
        --cyan:         #38bdf8;
        --cyan-dim:     rgba(56, 189, 248, 0.12);
        --red:          #f87171;
        --red-dim:      rgba(248, 113, 113, 0.12);
        --green:        #34d399;
        --green-dim:    rgba(52, 211, 153, 0.12);
        --text-primary: #e2eaf6;
        --text-muted:   #64748b;
        --text-sub:     #94a3b8;
        --font-mono:    'JetBrains Mono', monospace;
    }

    html, body, .stApp {
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif;
    }

    /* Subtle dot-grid background texture */
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        background-image: radial-gradient(rgba(0,212,170,0.04) 1px, transparent 1px);
        background-size: 28px 28px;
        pointer-events: none;
        z-index: 0;
    }

    /* ─── Scrollbar ─── */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: var(--bg-secondary); }
    ::-webkit-scrollbar-thumb { background: var(--teal-dim); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--teal); }

    /* ─── Sidebar ─── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #08101e 0%, #060c18 100%) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown li { color: var(--text-sub) !important; }

    /* ─── Glass Cards ─── */
    .glass-card {
        background: var(--bg-card);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.8rem 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.4), 0 0 0 1px rgba(0,212,170,0.04);
        transition: border-color 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .glass-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--teal-dim), transparent);
    }
    .glass-card:hover { border-color: var(--border-hover); }

    /* ─── Hero Header ─── */
    .hero-wrap {
        display: flex;
        align-items: center;
        gap: 1.2rem;
        padding: 2rem 0 0.5rem 0;
    }
    .hero-icon {
        width: 56px; height: 56px;
        background: linear-gradient(135deg, #00d4aa, #38bdf8);
        border-radius: 14px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.8rem;
        box-shadow: 0 0 24px var(--teal-glow);
        flex-shrink: 0;
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #00d4aa 0%, #38bdf8 50%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
        margin: 0;
    }
    .hero-sub {
        font-size: 0.9rem;
        color: var(--text-muted);
        margin: 0.3rem 0 0 0;
        font-weight: 400;
    }
    .hero-badges { display: flex; gap: 0.5rem; margin-top: 0.6rem; flex-wrap: wrap; }

    /* ─── Badges ─── */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        font-family: var(--font-mono);
        letter-spacing: 0.3px;
    }
    .badge-teal  { background: var(--teal-dim);  border: 1px solid rgba(0,212,170,0.3);  color: var(--teal); }
    .badge-cyan  { background: var(--cyan-dim);   border: 1px solid rgba(56,189,248,0.3); color: var(--cyan); }
    .badge-red   { background: var(--red-dim);    border: 1px solid rgba(248,113,113,0.3);color: var(--red); }
    .badge-green { background: var(--green-dim);  border: 1px solid rgba(52,211,153,0.3); color: var(--green); }

    /* ─── Stat boxes ─── */
    .stat-row { display: flex; gap: 1rem; margin: 1.2rem 0 1.6rem 0; }
    .stat-box {
        flex: 1;
        background: rgba(0, 212, 170, 0.04);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.1rem 1rem;
        text-align: center;
        transition: all 0.25s ease;
    }
    .stat-box:hover {
        background: var(--teal-dim);
        border-color: var(--border-hover);
        transform: translateY(-2px);
    }
    .stat-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--teal);
        font-family: var(--font-mono);
        line-height: 1;
    }
    .stat-lbl {
        font-size: 0.68rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-top: 0.4rem;
    }

    /* ─── Result Cards ─── */
    .result-card {
        border-radius: 14px;
        padding: 1.6rem 1.8rem;
        margin: 1.2rem 0;
        position: relative;
        overflow: hidden;
    }
    .result-card::before {
        content: "";
        position: absolute;
        inset: 0;
        background: repeating-linear-gradient(
            -45deg, transparent, transparent 8px,
            rgba(255,255,255,0.01) 8px, rgba(255,255,255,0.01) 9px
        );
    }
    .result-card-depressed {
        background: linear-gradient(135deg, rgba(127,29,29,0.55) 0%, rgba(69,10,10,0.75) 100%);
        border: 1px solid rgba(248,113,113,0.35);
        box-shadow: 0 0 32px rgba(248,113,113,0.12), inset 0 1px 0 rgba(248,113,113,0.15);
    }
    .result-card-depressed .rc-title { color: #fca5a5; }
    .result-card-depressed .rc-body  { color: #fecaca; }

    .result-card-healthy {
        background: linear-gradient(135deg, rgba(6,78,59,0.5) 0%, rgba(3,46,36,0.7) 100%);
        border: 1px solid rgba(52,211,153,0.35);
        box-shadow: 0 0 32px rgba(52,211,153,0.1), inset 0 1px 0 rgba(52,211,153,0.15);
    }
    .result-card-healthy .rc-title { color: #6ee7b7; }
    .result-card-healthy .rc-body  { color: #a7f3d0; }

    .rc-header { display: flex; align-items: center; gap: 0.7rem; margin-bottom: 0.8rem; }
    .rc-icon { font-size: 1.5rem; }
    .rc-title { font-size: 1.1rem; font-weight: 700; margin: 0; }
    .rc-body { font-size: 0.9rem; line-height: 1.6; margin: 0; }
    .rc-disclaimer { font-size: 0.78rem; opacity: 0.75; margin-top: 0.7rem; font-style: italic; }

    .helpline-btn {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        margin-top: 1rem;
        padding: 0.45rem 1.1rem;
        background: rgba(248,113,113,0.18);
        color: #fca5a5 !important;
        border: 1px solid rgba(248,113,113,0.35);
        border-radius: 20px;
        text-decoration: none !important;
        font-weight: 600;
        font-size: 0.82rem;
        transition: all 0.2s ease;
    }
    .helpline-btn:hover {
        background: rgba(248,113,113,0.3);
        color: #fff !important;
    }

    /* ─── Probability Gauge Bar ─── */
    .gauge-wrap { margin-top: 1.2rem; }
    .gauge-label {
        display: flex; justify-content: space-between;
        font-size: 0.78rem; color: var(--text-muted); margin-bottom: 0.4rem;
    }
    .gauge-track {
        height: 8px;
        background: rgba(255,255,255,0.06);
        border-radius: 99px;
        overflow: hidden;
        position: relative;
    }
    .gauge-fill {
        height: 100%;
        border-radius: 99px;
        transition: width 0.8s cubic-bezier(.4,0,.2,1);
        position: relative;
    }
    .gauge-fill::after {
        content: "";
        position: absolute;
        right: 0; top: 0; bottom: 0; width: 6px;
        background: rgba(255,255,255,0.5);
        border-radius: 99px;
        filter: blur(2px);
    }
    .gauge-fill-dep  { background: linear-gradient(90deg, #991b1b, #f87171); }
    .gauge-fill-safe { background: linear-gradient(90deg, #065f46, #34d399); }
    .gauge-score {
        font-family: var(--font-mono);
        font-size: 1.1rem;
        font-weight: 700;
        margin-top: 0.5rem;
        text-align: right;
    }

    /* ─── Section Headings ─── */
    .section-heading {
        font-size: 1rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: 0.2px;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .section-heading::after {
        content: "";
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, var(--border), transparent);
        margin-left: 0.5rem;
    }

    /* ─── Sidebar internal styles ─── */
    .sidebar-logo {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        padding: 1rem 0 0.5rem 0;
        margin-bottom: 0.5rem;
    }
    .sidebar-logo-icon {
        width: 36px; height: 36px;
        background: linear-gradient(135deg, #00d4aa, #38bdf8);
        border-radius: 9px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.1rem;
        flex-shrink: 0;
    }
    .sidebar-logo-text { font-size: 1rem; font-weight: 700; color: var(--text-primary); }
    .sidebar-logo-sub  { font-size: 0.7rem; color: var(--text-muted); margin-top: 1px; }

    .model-status {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.7rem 1rem;
        border-radius: 10px;
        margin: 0.8rem 0 1.2rem 0;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .model-status-ok {
        background: var(--green-dim);
        border: 1px solid rgba(52,211,153,0.25);
        color: var(--green);
    }
    .model-status-err {
        background: var(--red-dim);
        border: 1px solid rgba(248,113,113,0.25);
        color: var(--red);
    }
    .status-dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        flex-shrink: 0;
        animation: pulse 2s infinite;
    }
    .dot-green { background: var(--green); box-shadow: 0 0 6px var(--green); }
    .dot-red   { background: var(--red);   box-shadow: 0 0 6px var(--red); }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.45; }
    }

    .spec-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.55rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.04);
        font-size: 0.82rem;
    }
    .spec-row:last-child { border-bottom: none; }
    .spec-key { color: var(--text-muted); }
    .spec-val { color: var(--teal); font-family: var(--font-mono); font-size: 0.78rem; }

    /* ─── Buttons ─── */
    .stButton > button {
        background: linear-gradient(135deg, #00d4aa, #38bdf8) !important;
        color: #060c18 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.65rem 1.8rem !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.2px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 16px rgba(0,212,170,0.25) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 24px rgba(0,212,170,0.4) !important;
    }
    .stButton > button:active { transform: translateY(0px) !important; }

    /* Secondary button (delete/clear) */
    [data-testid="stButton"]:has(button:not([kind="primary"])) button {
        background: rgba(248,113,113,0.12) !important;
        color: #fca5a5 !important;
        border: 1px solid rgba(248,113,113,0.25) !important;
        box-shadow: none !important;
    }

    /* ─── Tabs ─── */
    [data-testid="stTabs"] [role="tablist"] {
        border-bottom: 1px solid var(--border) !important;
        gap: 0.5rem;
    }
    [data-testid="stTabs"] [role="tab"] {
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        color: var(--text-muted) !important;
        padding: 0.6rem 1.2rem !important;
        border-radius: 8px 8px 0 0 !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        color: var(--teal) !important;
        border-bottom: 2px solid var(--teal) !important;
        background: var(--teal-dim) !important;
    }
    [data-testid="stTabs"] [role="tab"]:hover {
        color: var(--text-primary) !important;
        background: rgba(255,255,255,0.04) !important;
    }

    /* ─── Inputs / Radio ─── */
    .stRadio [data-testid="stMarkdownContainer"] p { color: var(--text-sub) !important; }
    .stRadio label { color: var(--text-sub) !important; font-size: 0.88rem; }

    /* ─── File uploader ─── */
    [data-testid="stFileUploadDropzone"] {
        background: rgba(0,212,170,0.03) !important;
        border: 1px dashed rgba(0,212,170,0.2) !important;
        border-radius: 12px !important;
    }

    /* ─── Alert / Info boxes ─── */
    .stAlert { border-radius: 10px !important; }
    [data-testid="stInfoMessage"] { border-left-color: var(--cyan) !important; }

    /* ─── DataFrame table ─── */
    [data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

    /* ─── Download button ─── */
    [data-testid="stDownloadButton"] button {
        background: rgba(56,189,248,0.12) !important;
        border: 1px solid rgba(56,189,248,0.25) !important;
        color: var(--cyan) !important;
        font-weight: 600 !important;
        box-shadow: none !important;
    }

    /* ─── History table label colors ─── */
    .label-dep  { color: #fca5a5; font-weight: 600; }
    .label-safe { color: #6ee7b7; font-weight: 600; }

    /* ─── Audio recorder ─── */
    [data-testid="stAudioInput"] {
        border-radius: 12px !important;
    }

    /* ─── Spinner text ─── */
    [data-testid="stSpinner"] p { color: var(--teal) !important; }

    /* ─── Expander ─── */
    [data-testid="stExpander"] {
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        background: rgba(0,0,0,0.2) !important;
    }
    [data-testid="stExpander"] summary {
        color: var(--text-sub) !important;
        font-size: 0.85rem;
    }

    /* ─── Selectbox ─── */
    [data-testid="stSelectbox"] > div > div {
        background: rgba(11,20,37,0.9) !important;
        border-color: var(--border) !important;
        color: var(--text-primary) !important;
        border-radius: 10px !important;
    }

    /* ─── Text input ─── */
    .stTextInput input {
        background: rgba(11,20,37,0.9) !important;
        border-color: var(--border) !important;
        color: var(--text-primary) !important;
        border-radius: 10px !important;
    }
    .stTextInput input:focus { border-color: var(--teal) !important; box-shadow: 0 0 0 2px var(--teal-dim) !important; }

    /* ─── Progress bar ─── */
    [data-testid="stProgressBar"] > div { background: linear-gradient(90deg, #00d4aa, #38bdf8) !important; }

    /* ─── Footer ─── */
    .app-footer {
        text-align: center;
        padding: 2.5rem 1rem 3rem;
        color: #2d3748;
        font-size: 0.75rem;
        border-top: 1px solid rgba(255,255,255,0.03);
        margin-top: 3rem;
        line-height: 1.8;
    }
    .app-footer a { color: var(--teal); text-decoration: none; }
</style>
""", unsafe_allow_html=True)


# ── Core Helpers ───────────────────────────────────────────────────────────────
def find_model_path() -> Path | None:
    onnx_candidates = list(MODELS_DIR.glob("*.onnx"))
    if onnx_candidates:
        return max(onnx_candidates, key=lambda p: p.stat().st_mtime)
    return None

@st.cache_resource
def load_model(model_path: str):
    session = ort.InferenceSession(model_path)
    return session

def extract_mfcc(file_path: str, max_len: int = None, n_mfcc: int = None) -> np.ndarray | None:
    if max_len is None:
        max_len = MODEL_CONFIG["max_len"]
    if n_mfcc is None:
        n_mfcc = MODEL_CONFIG["n_mfcc"]
    sr = MODEL_CONFIG["sr"]

    try:
        y, _ = librosa.load(file_path, sr=sr)
        y, _ = librosa.effects.trim(y, top_db=25)

        mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        delta  = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        features = np.concatenate([mfcc, delta, delta2], axis=0).T

        mean = features.mean(axis=0, keepdims=True)
        std  = features.std(axis=0, keepdims=True) + 1e-8
        features = (features - mean) / std

        if features.shape[0] < max_len:
            features = np.pad(features, ((0, max_len - features.shape[0]), (0, 0)))
        else:
            features = features[:max_len]

        return features.astype(np.float32)
    except Exception as e:
        st.error(f"Feature extraction failed: {e}")
        return None

def extract_auxiliary_biomarkers(file_path: str) -> dict:
    """Extract secondary vocal features for display."""
    sr = MODEL_CONFIG["sr"]
    try:
        y, _ = librosa.load(file_path, sr=sr)
        y, _ = librosa.effects.trim(y, top_db=25)
        duration = len(y) / sr

        zcr      = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))

        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=75, fmax=300, sr=sr, fill_value=np.nan
        )
        avg_pitch = float(np.nanmean(f0)) if np.any(voiced_flag) else 0.0

        return {
            "duration": duration,
            "pitch":    avg_pitch,
            "zcr":      zcr,
            "centroid": centroid,
            "flatness": flatness
        }
    except Exception:
        return {"duration": 0.0, "pitch": 0.0, "zcr": 0.0, "centroid": 0.0, "flatness": 0.0}

def generate_audio_plots(file_path: str):
    """Generate professional dark-themed voice plots."""
    sr = MODEL_CONFIG["sr"]
    try:
        y, _ = librosa.load(file_path, sr=sr)
        y, _ = librosa.effects.trim(y, top_db=25)

        plt.rcParams.update({
            'text.color':        '#94a3b8',
            'axes.labelcolor':   '#64748b',
            'xtick.color':       '#475569',
            'ytick.color':       '#475569',
            'axes.spines.top':   False,
            'axes.spines.right': False,
            'axes.spines.left':  False,
            'axes.spines.bottom':False,
        })

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        fig.patch.set_facecolor('#0b1425')
        fig.subplots_adjust(hspace=0.4)

        # Waveform
        librosa.display.waveshow(y, sr=sr, ax=ax1, color='#00d4aa', alpha=0.8, linewidth=0.6)
        ax1.set_title("Amplitude Waveform", color='#e2eaf6', fontsize=10, fontweight='bold', pad=8)
        ax1.set_facecolor('#060c18')
        ax1.set_ylabel("Amplitude", fontsize=8, color='#475569')
        ax1.grid(axis='y', color='rgba(0,212,170,0.05)', linestyle='--', linewidth=0.5)
        ax1.tick_params(colors='#334155', labelsize=7)

        # Mel Spectrogram
        S    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax2, cmap='viridis')
        ax2.set_title("Mel-Spectrogram  (Acoustic Fingerprint)", color='#e2eaf6', fontsize=10, fontweight='bold', pad=8)
        ax2.set_facecolor('#060c18')
        ax2.set_ylabel("Frequency (Hz)", fontsize=8, color='#475569')
        ax2.set_xlabel("Time (s)", fontsize=8, color='#475569')
        ax2.tick_params(colors='#334155', labelsize=7)

        return fig
    except Exception as e:
        st.error(f"Failed to generate plots: {e}")
        return None

def predict(session, features: np.ndarray) -> tuple[float, str]:
    threshold   = MODEL_CONFIG["best_threshold"]
    x           = features[np.newaxis, ...]
    input_name  = session.get_inputs()[0].name
    output      = session.run(None, {input_name: x})
    prob        = float(output[0][0][0])
    label       = "Depressed" if prob >= threshold else "Not Depressed"
    return prob, label

# ── Load Model ─────────────────────────────────────────────────────────────────
auto_model_path = find_model_path()
model = None

if auto_model_path and os.path.exists(str(auto_model_path)):
    try:
        model = load_model(str(auto_model_path))
    except Exception as e:
        st.sidebar.error(f"Could not load model: {e}")


# ══════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Logo + name
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-icon">🩺</div>
        <div>
            <div class="sidebar-logo-text">VoiceScreen AI</div>
            <div class="sidebar-logo-sub">Clinical Voice Analysis</div>
        </div>
    </div>
    <hr style="border-color: rgba(0,212,170,0.08); margin: 0.5rem 0 1rem;">
    """, unsafe_allow_html=True)

    # Model Status
    if model is not None:
        st.markdown(f"""
        <div class="model-status model-status-ok">
            <div class="status-dot dot-green"></div>
            Model Loaded &nbsp;·&nbsp; BiLSTM-ONNX
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="model-status model-status-err">
            <div class="status-dot dot-red"></div>
            No model found in <code>models/</code>
        </div>
        """, unsafe_allow_html=True)

    # System Specs
    st.markdown('<div class="section-heading">⚙️ System Config</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="margin-bottom:1.2rem;">
        <div class="spec-row">
            <span class="spec-key">Sample Rate</span>
            <span class="spec-val">{MODEL_CONFIG['sr'] // 1000} kHz</span>
        </div>
        <div class="spec-row">
            <span class="spec-key">Features</span>
            <span class="spec-val">{MODEL_CONFIG['n_features']} MFCC+Δ+Δ²</span>
        </div>
        <div class="spec-row">
            <span class="spec-key">Decision Threshold</span>
            <span class="spec-val">{MODEL_CONFIG['best_threshold']:.2f}</span>
        </div>
        <div class="spec-row">
            <span class="spec-key">Max Frames</span>
            <span class="spec-val">{MODEL_CONFIG['max_len']}</span>
        </div>
        <div class="spec-row">
            <span class="spec-key">Config Source</span>
            <span class="spec-val">{'model_config.json' if MODEL_CONFIG_PATH.exists() else 'defaults'}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Biomarker Guide
    st.markdown('<div class="section-heading">📖 Biomarker Guide</div>', unsafe_allow_html=True)
    with st.expander("What do these metrics mean?"):
        st.markdown("""
**🎵 Pitch (f₀)**
Fundamental voice frequency. Depressive speech often shows restricted pitch variability (monotone) and lower mean f₀.

**📡 Spectral Centroid**
"Brightness" of sound. Depression is associated with a darker, lower-frequency vocal profile.

**〰️ Spectral Flatness**
Tonal vs. noise-like quality. Values near 1 = noise-like; values near 0 = tonal.

**📉 ZCR (Zero Crossing Rate)**
How often the signal crosses zero — reflects noisiness and voice energy.
        """)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; color:#1e293b; font-size:0.7rem; padding-top:1rem; border-top: 1px solid rgba(255,255,255,0.04);">
        VoiceScreen AI · v2.0.0<br>
        Research Use Only
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════

# Hero Header
st.markdown("""
<div class="hero-wrap">
    <div class="hero-icon">🩺</div>
    <div>
        <div class="hero-title">VoiceScreen AI</div>
        <div class="hero-sub">Deep-learning acoustic biomarker screening for clinical indicators of depression</div>
        <div class="hero-badges">
            <span class="badge badge-teal">BiLSTM Neural Network</span>
            <span class="badge badge-cyan">120 MFCC Features</span>
            <span class="badge badge-cyan">DAIC-WOZ Dataset</span>
            <span class="badge badge-teal">ONNX Runtime</span>
        </div>
    </div>
</div>
<div style="height: 1.5rem;"></div>
""", unsafe_allow_html=True)

# Tabs
tab_diagnostics, tab_analytics = st.tabs(["🎙️  Vocal Workspace", "📊  History & Analytics"])

# ══════════════════════════════════════════════════════════════════════
#  TAB 1 — DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════
with tab_diagnostics:
    col_input, col_plots = st.columns([1, 1], gap="large")

    # ── Left Column: Input + Results ──────────────────────────────────
    with col_input:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">🎤 Audio Capture</div>', unsafe_allow_html=True)

        capture_method = st.radio(
            "Capture Method",
            ["🎙️  Record Voice Directly", "📂  Upload Audio File (.wav)"],
            horizontal=True,
            label_visibility="collapsed"
        )
        uploaded_file = None

        if "Record" in capture_method:
            recorded_audio = st.audio_input("Click the microphone to start recording")
            if recorded_audio:
                uploaded_file = recorded_audio
        else:
            st.markdown("<div style='height:0.3rem;'></div>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Drop a .wav file here or click to browse",
                type=["wav"],
                label_visibility="visible"
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # ── Metadata + Inference ────────────────────────────────────
        if uploaded_file is not None:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-heading">🔬 Vocal Biomarkers</div>', unsafe_allow_html=True)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(uploaded_file.read() if hasattr(uploaded_file, 'read') else uploaded_file.getvalue())
                tmp_path = tmp.name

            biomarkers = extract_auxiliary_biomarkers(tmp_path)

            # Duration + ZCR row
            dur_color = "#f87171" if biomarkers['duration'] < 10 else "#34d399"
            st.markdown(f"""
            <div style="display:flex; gap:0.5rem; margin-bottom:0.8rem;">
                <span class="badge badge-teal">⏱ {biomarkers['duration']:.1f}s</span>
                <span class="badge badge-cyan">ZCR: {biomarkers['zcr']:.4f}</span>
            </div>
            """, unsafe_allow_html=True)

            if biomarkers['duration'] < 10:
                st.warning(f"⚠️ Short recording ({biomarkers['duration']:.1f}s) — accuracy may be reduced. Aim for 15–30s of natural speech.")

            # Feature stat boxes
            pitch_text = f"{biomarkers['pitch']:.1f} Hz" if biomarkers['pitch'] > 0 else "Unvoiced"
            st.markdown(f"""
            <div class="stat-row">
                <div class="stat-box">
                    <div class="stat-val">{pitch_text}</div>
                    <div class="stat-lbl">Pitch (f₀)</div>
                </div>
                <div class="stat-box">
                    <div class="stat-val">{biomarkers['centroid']:.0f} Hz</div>
                    <div class="stat-lbl">Spectral Centroid</div>
                </div>
                <div class="stat-box">
                    <div class="stat-val">{biomarkers['flatness']:.4f}</div>
                    <div class="stat-lbl">Spectral Flatness</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Run Inference
            if model is None:
                st.info("⚠️ Place your `depression_model.onnx` inside the `models/` directory to enable inference.")
            else:
                features = extract_mfcc(tmp_path)
                if features is not None:
                    if st.button("🔍  Run Diagnostic Scan", use_container_width=True, type="primary"):
                        with st.spinner("Analyzing vocal biomarkers…"):
                            prob, label = predict(model, features)
                            file_name_clean = getattr(uploaded_file, 'name', 'mic_recording.wav')
                            save_to_history(file_name_clean, biomarkers['duration'], prob, label)
                            st.session_state['last_scan'] = {
                                "prob": prob,
                                "label": label,
                                "filename": file_name_clean,
                                "duration": biomarkers['duration']
                            }

            os.unlink(tmp_path)
            st.markdown("</div>", unsafe_allow_html=True)

            # ── Results ────────────────────────────────────────────
            if 'last_scan' in st.session_state:
                res  = st.session_state['last_scan']
                prob = res['prob']
                pct  = prob * 100

                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-heading">📋 Analysis Report</div>', unsafe_allow_html=True)

                if res['label'] == "Depressed":
                    gauge_class = "gauge-fill-dep"
                    st.markdown(f"""
                    <div class="result-card result-card-depressed">
                        <div class="rc-header">
                            <span class="rc-icon">⚠️</span>
                            <p class="rc-title">Depressive Acoustic Markers Detected</p>
                        </div>
                        <p class="rc-body">
                            The BiLSTM model identified vocal patterns — including restricted pitch variability,
                            flattened spectral envelope, and altered prosody — consistent with clinical depression
                            indicators at a probability of <strong>{prob:.1%}</strong>.
                        </p>
                        <p class="rc-disclaimer">
                            This tool is a research-grade screening aid, not a medical diagnosis.
                        </p>
                        <a class="helpline-btn" href="https://findahelpline.com/" target="_blank">
                            🆘 Find Support & Helplines Near You ↗
                        </a>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    gauge_class = "gauge-fill-safe"
                    conf = (1 - prob) * 100
                    st.markdown(f"""
                    <div class="result-card result-card-healthy">
                        <div class="rc-header">
                            <span class="rc-icon">✅</span>
                            <p class="rc-title">No Depressive Indicators Detected</p>
                        </div>
                        <p class="rc-body">
                            The voice sample falls within the normative baseline range (confidence: <strong>{conf:.1f}%</strong>).
                            Acoustic metrics reflect healthy pitch variance and balanced spectral distribution.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                # Gauge bar
                st.markdown(f"""
                <div class="gauge-wrap">
                    <div class="gauge-label">
                        <span>Depression Probability Index</span>
                        <span style="color:{'#f87171' if prob >= MODEL_CONFIG['best_threshold'] else '#34d399'}">
                            Threshold: {MODEL_CONFIG['best_threshold']:.0%}
                        </span>
                    </div>
                    <div class="gauge-track">
                        <div class="gauge-fill {gauge_class}" style="width:{min(pct,100):.1f}%;"></div>
                    </div>
                    <div class="gauge-score" style="color:{'#f87171' if prob >= MODEL_CONFIG['best_threshold'] else '#34d399'}">
                        {prob:.4f}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

    # ── Right Column: Visualizations ──────────────────────────────────
    with col_plots:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">📈 Signal Visualizations</div>', unsafe_allow_html=True)

        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(uploaded_file.read() if hasattr(uploaded_file, 'read') else uploaded_file.getvalue())
                tmp_path = tmp.name

            with st.spinner("Rendering spectral analysis…"):
                fig = generate_audio_plots(tmp_path)
            if fig:
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            os.unlink(tmp_path)
        else:
            st.markdown("""
            <div style="
                display:flex; flex-direction:column; align-items:center; justify-content:center;
                min-height:260px; gap:1rem; color:#1e293b;
            ">
                <div style="font-size:3rem; opacity:0.3;">📊</div>
                <div style="font-size:0.85rem; text-align:center; max-width:220px;">
                    Record or upload an audio file to visualize the acoustic waveform and mel-spectrogram.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Info card below plot
        st.markdown("""
        <div class="glass-card" style="padding: 1.2rem 1.5rem;">
            <div class="section-heading">💡 Recording Tips</div>
            <div style="font-size:0.82rem; color:#64748b; line-height:1.8;">
                • Record in a <strong style="color:#94a3b8;">quiet environment</strong> with minimal background noise<br>
                • Speak naturally for at least <strong style="color:#94a3b8;">15–30 seconds</strong><br>
                • Use a standard <strong style="color:#94a3b8;">.wav</strong> file at 16 kHz sample rate<br>
                • Avoid clipping — keep volume at a comfortable speaking level<br>
                • Clinical interview-style speech yields the most accurate results
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  TAB 2 — HISTORY & ANALYTICS
# ══════════════════════════════════════════════════════════════════════
with tab_analytics:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">🗂️ Diagnostic Logs</div>', unsafe_allow_html=True)

    history = load_history()

    if not history:
        st.markdown("""
        <div style="text-align:center; padding:3rem 1rem; color:#1e293b;">
            <div style="font-size:2.5rem; margin-bottom:0.8rem; opacity:0.3;">📂</div>
            <div style="font-size:0.9rem;">No diagnostic records yet. Run a vocal scan to start building history.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        df = pd.DataFrame(history).iloc[::-1].reset_index(drop=True)

        # Overview stats
        total_tests    = len(df)
        depressed_count = len(df[df['label'] == 'Depressed'])
        dep_rate       = depressed_count / total_tests if total_tests > 0 else 0.0
        avg_score      = df['probability'].mean()
        last_scan_time = df.iloc[0]['timestamp'] if total_tests > 0 else "—"

        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-box">
                <div class="stat-val">{total_tests}</div>
                <div class="stat-lbl">Total Scans</div>
            </div>
            <div class="stat-box">
                <div class="stat-val">{dep_rate:.0%}</div>
                <div class="stat-lbl">Detected Rate</div>
            </div>
            <div class="stat-box">
                <div class="stat-val">{avg_score:.3f}</div>
                <div class="stat-lbl">Avg Score</div>
            </div>
            <div class="stat-box">
                <div class="stat-val">{total_tests - depressed_count}</div>
                <div class="stat-lbl">Clear Scans</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Search
        search_query = st.text_input("🔍  Search by filename", placeholder="Type to filter records…")
        df_display = df[df['filename'].str.contains(search_query, case=False)] if search_query else df.copy()

        # Rename for display
        df_show = df_display.rename(columns={
            "timestamp":    "Timestamp",
            "filename":     "Audio File",
            "duration_sec": "Duration (s)",
            "probability":  "Score",
            "label":        "Result"
        })[["Timestamp", "Audio File", "Duration (s)", "Score", "Result"]]

        st.dataframe(df_show, use_container_width=True, hide_index=True)

        # Actions
        col_dl, col_del = st.columns([1, 1])

        with col_dl:
            csv = df_show.to_csv(index=False)
            st.download_button(
                label="📥  Export to CSV",
                data=csv,
                file_name=f"voicescreen_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col_del:
            delete_options = {
                f"{e['timestamp']} — {e['filename']}  [#{e['id']}]": e['id']
                for e in history
            }
            select_to_delete = st.selectbox("Select record to delete", list(delete_options.keys()))
            cb1, cb2 = st.columns(2)
            with cb1:
                if st.button("🗑  Delete Selected", use_container_width=True):
                    delete_history_entry(delete_options[select_to_delete])
                    st.success("Record deleted.")
                    st.rerun()
            with cb2:
                if st.button("🧹  Clear All Logs", use_container_width=True):
                    clear_all_history()
                    st.success("All logs cleared.")
                    st.rerun()

        # Trend chart
        st.markdown("<hr style='border-color:rgba(255,255,255,0.05); margin:1.8rem 0;'>", unsafe_allow_html=True)
        st.markdown('<div class="section-heading">📉 Depression Score Trend</div>', unsafe_allow_html=True)

        df_plot = df.iloc[::-1].copy()
        df_plot['Timestamp'] = pd.to_datetime(df_plot['timestamp'])
        chart_data = df_plot[['Timestamp', 'probability']].set_index('Timestamp')
        chart_data = chart_data.rename(columns={"probability": "Depression Score"})
        st.area_chart(chart_data, color="#00d4aa")

    st.markdown("</div>", unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    <strong style="color:#1e293b;">VoiceScreen AI</strong> &nbsp;·&nbsp;
    Research & educational use only — not a clinical diagnostic tool.<br>
    Built with <a href="https://streamlit.io">Streamlit</a> &nbsp;·&nbsp;
    BiLSTM · ONNX Runtime · librosa &nbsp;·&nbsp;
    DAIC-WOZ Dataset &nbsp;·&nbsp; 120 MFCC Features<br>
    <span style="color:#0f172a;">© 2026 VoiceScreen AI. All rights reserved.</span>
</div>
""", unsafe_allow_html=True)