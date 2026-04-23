# Depression Detection from Audio (DAIC-WOZ)

An AI-powered screening tool that analyzes audio clips to detect markers associated with depression. This project uses a **Bidirectional Long Short-Term Memory (BiLSTM)** neural network trained on the [DAIC-WOZ dataset](https://dcapswoz.ict.usc.edu/).

## Key Features
- **High-Fidelity Audio Analysis**: Uses 40 MFCCs plus Delta and Delta-Delta features (120 total features per frame).
- **Proactive Pre-processing**: Automatic silence trimming and per-sample z-normalization.
- **Deep Learning Architecture**: Stacked Bidirectional LSTMs with Batch Normalization for temporal feature learning.
- **Interactive UI**: Clean Streamlit interface for uploading audio and viewing real-time predictions.

## Setup & Installation

1. **Clone the repository** (or navigate to the directory).
2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```
3. **Install dependencies**:
   ```bash
   pip install streamlit tensorflow librosa numpy pandas scikit-learn
   ```

## How to Run

1. **Train/Download the model**:
   - Ensure you have `best_depression_model.keras` and `model_config.json` inside the `models/` folder.
2. **Start the application**:
   ```bash
   streamlit run app.py
   ```
3. **Analyze**: Upload a `.wav` file or use the built-in test audio.

## Model Performance
The current model utilizes an optimized threshold of **0.34**, achieving:
- **Accuracy**: ~74%
- **Recall (Depressed)**: ~86%
- **F1-Score (Depressed)**: ~0.76

## Disclaimer
This tool is for **research purposes only** and is not a clinical diagnostic instrument. It is designed to demonstrate the potential of vocal biomarkers in mental health screening. Always consult a medical professional for clinical diagnosis.
