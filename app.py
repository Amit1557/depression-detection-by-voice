import streamlit as st
import numpy as np
import librosa
import tempfile
import os
import json
import onnxruntime as ort
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Depression Detection – Audio Analysis",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_CONFIG_PATH = MODELS_DIR / "model_config.json"

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

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Hero header */
    .hero {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem;
    }
    .hero h1 {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #8b5cf6, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .hero p {
        color: #94a3b8;
        font-size: 1.05rem;
        max-width: 520px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* Info cards row */
    .info-row {
        display: flex;
        gap: 1rem;
        margin: 1.5rem 0;
    }
    .info-card {
        flex: 1;
        background: linear-gradient(135deg, #1e1b4b10, #312e8110);
        border: 1px solid #312e8130;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .info-card .num {
        font-size: 1.5rem;
        font-weight: 700;
        color: #818cf8;
    }
    .info-card .label {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 0.25rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Upload section */
    .upload-section {
        background: linear-gradient(135deg, #0f172a, #1e1b4b);
        border: 1px dashed #4f46e580;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1.5rem 0;
    }
    .upload-section h3 {
        color: #e2e8f0;
        margin-bottom: 0.5rem;
    }
    .upload-section p {
        color: #64748b;
        font-size: 0.9rem;
    }

    /* Result cards */
    .result-depressed {
        background: linear-gradient(135deg, #450a0a, #7f1d1d);
        border: 1px solid #dc262640;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .result-depressed h2 {
        color: #fca5a5;
        margin: 0;
        font-size: 1.8rem;
    }
    .result-depressed .conf {
        color: #fecaca;
        font-size: 1rem;
        margin-top: 0.5rem;
    }

    .result-healthy {
        background: linear-gradient(135deg, #052e16, #14532d);
        border: 1px solid #16a34a40;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .result-healthy h2 {
        color: #86efac;
        margin: 0;
        font-size: 1.8rem;
    }
    .result-healthy .conf {
        color: #bbf7d0;
        font-size: 1rem;
        margin-top: 0.5rem;
    }

    /* Feature badge */
    .feature-badge {
        display: inline-block;
        background: #312e8130;
        color: #a5b4fc;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.2rem;
    }

    /* Divider */
    .custom-divider {
        border: none;
        height: 1px;
        background: linear-gradient(to right, transparent, #4f46e540, transparent);
        margin: 2rem 0;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: #64748b;
        font-size: 0.8rem;
        line-height: 1.6;
    }

    /* Hide default streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Progress bar color */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #6366f1, #a78bfa);
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #7c3aed) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def find_model_path() -> Path | None:
    """Find a usable ONNX or Keras model inside models/."""
    # Prefer ONNX
    onnx_candidates = list(MODELS_DIR.glob("*.onnx"))
    if onnx_candidates:
        return max(onnx_candidates, key=lambda p: p.stat().st_mtime)

    return None


@st.cache_resource
def load_model(model_path: str):
    """Load the ONNX model (cached across reruns)."""
    session = ort.InferenceSession(model_path)
    return session


def extract_mfcc(file_path: str, max_len: int = None, n_mfcc: int = None) -> np.ndarray | None:
    """Extract MFCC + delta + delta-delta features — mirrors the fixed notebook."""
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


def predict(session, features: np.ndarray) -> tuple[float, str]:
    """Run ONNX inference and return (probability, label)."""
    threshold = MODEL_CONFIG["best_threshold"]
    x = features[np.newaxis, ...]  # (1, max_len, n_features)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: x})
    prob = float(output[0][0][0])
    label = "Depressed" if prob >= threshold else "Not Depressed"
    return prob, label


# ── UI ─────────────────────────────────────────────────────────────────────────

# Hero
st.markdown("""
<div class="hero">
    <h1>Depression Detection from Audio</h1>
    <p>
        Upload a voice recording and our BiLSTM model will analyze
        vocal biomarkers to screen for signs of depression.
    </p>
</div>
""", unsafe_allow_html=True)

# Info cards
st.markdown(f"""
<div class="info-row">
    <div class="info-card">
        <div class="num">120</div>
        <div class="label">Audio Features</div>
    </div>
    <div class="info-card">
        <div class="num">BiLSTM</div>
        <div class="label">Architecture</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

# ── Load model silently ───────────────────────────────────────────────────────
auto_model_path = find_model_path()
model = None

if auto_model_path and os.path.exists(str(auto_model_path)):
    try:
        model = load_model(str(auto_model_path))
    except Exception as e:
        st.error(f"Could not load model: {e}")
else:
    st.warning("No ONNX model found in the `models/` folder. Please add your `depression_model.onnx` file there.")

# ── Upload section ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="upload-section">
    <h3>Upload Audio File</h3>
    <p>Supported format: WAV - Mono or stereo - Any sample rate (resampled to 16 kHz)</p>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Choose a WAV file",
    type=["wav"],
    label_visibility="collapsed",
)

if uploaded is not None:
    # Audio player
    st.audio(uploaded, format="audio/wav")

    # Save to temp file for librosa
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    # Check audio duration
    y_check, _ = librosa.load(tmp_path, sr=MODEL_CONFIG["sr"])
    duration_sec = len(y_check) / MODEL_CONFIG["sr"]

    if duration_sec < 10:
        st.warning(
            f"Audio is only **{duration_sec:.1f}s** long. "
            "This model was trained on clinical interview recordings (1-20 minutes). "
            "Short clips like laughs, sounds, or brief phrases may produce unreliable results. "
            "For best accuracy, upload a longer conversational speech recording."
        )

    # Extract features
    with st.spinner("Extracting audio features..."):
        features = extract_mfcc(tmp_path)

    os.unlink(tmp_path)

    if features is not None:
        # Show feature info
        st.markdown(f"""
        <div style="text-align: center; margin: 1rem 0;">
            <span class="feature-badge">File: {uploaded.name}</span>
            <span class="feature-badge">Shape: {features.shape[0]} x {features.shape[1]}</span>
            <span class="feature-badge">Ready for analysis</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        # Predict
        if model is None:
            st.info("Load a model first to run predictions.")
        else:
            if st.button("Analyse Audio", type="primary", use_container_width=True):
                with st.spinner("Running inference..."):
                    prob, label = predict(model, features)

                # Result card
                if label == "Depressed":
                    st.markdown(f"""
                    <div class="result-depressed">
                        <h2>Signs of Depression Detected</h2>
                        <p class="conf">Confidence: <strong>{prob:.1%}</strong> &nbsp;|&nbsp; Threshold: {MODEL_CONFIG['best_threshold']:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-healthy">
                        <h2>No Signs of Depression</h2>
                        <p class="conf">Confidence: <strong>{1 - prob:.1%}</strong> &nbsp;|&nbsp; Threshold: {MODEL_CONFIG['best_threshold']:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Probability bar
                st.markdown("")
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.caption("Depression Probability")
                    st.progress(min(prob, 1.0))
                with col2:
                    st.metric("Score", f"{prob:.4f}")

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    This tool is for <strong>research purposes only</strong> and is not a clinical diagnostic instrument.<br>
    If you or someone you know is struggling, please reach out to a mental health professional.<br><br>
    Built with DAIC-WOZ Dataset - BiLSTM - MFCC Features
</div>
""", unsafe_allow_html=True)