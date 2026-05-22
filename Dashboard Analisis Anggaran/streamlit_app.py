import csv
import io
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Paths
DATA_PATH = Path("data/02_realisasi_anggaran_klasifikasi.csv")
MODEL_PATH = Path("model/Best_model.pkcls")
FALLBACK_MODEL_PATH = Path("Best_model.pkcls")

st.set_page_config(
    page_title="Dashboard Realisasi Anggaran",
    page_icon="📊",
    layout="wide",
)

# -----------------------------
# Utility functions
# -----------------------------

def detect_csv_delimiter(sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;|\t")
        return dialect.delimiter
    except csv.Error:
        return ","


@st.cache_data
def load_dataset(path):
    if isinstance(path, (Path, str)):
        with open(path, "rb") as f:
            raw_data = f.read()
    else:
        path.seek(0)
        raw_data = path.read()

    raw_text = raw_data.decode("utf-8", errors="replace") if isinstance(raw_data, bytes) else raw_data

    if not raw_text.strip():
        raise ValueError("Dataset kosong.")

    delimiter = detect_csv_delimiter(raw_text[:4096])

    df = pd.read_csv(io.StringIO(raw_text), sep=delimiter, engine="python")

    required_columns = [
        "realisasi_tercapai_95persen",
        "provinsi",
        "jenis_belanja_utama",
        "tipe_satker",
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Kolom tidak ditemukan: {col}")

    df["target"] = df["realisasi_tercapai_95persen"].map({"Ya": 1, "Tidak": 0})

    return df


@st.cache_resource
def load_model(path):
    try:
        if isinstance(path, (Path, str)):
            with open(path, "rb") as f:
                return pickle.load(f)
        else:
            path.seek(0)
            return pickle.load(path)
    except Exception as e:
        st.error(
            "❌ Model tidak dapat dimuat. Kemungkinan model membutuhkan Orange.\n\n"
            "Solusi:\n"
            "- Gunakan model scikit-learn (.pkl)\n"
            "- Atau upload model yang kompatibel"
        )
        return None


def get_model_feature_names(model):
    if hasattr(model, "domain"):  # Orange
        return [attr.name for attr in model.domain.attributes]
    elif hasattr(model, "feature_names_in_"):  # sklearn
        return list(model.feature_names_in_)
    else:
        raise ValueError("Format model tidak dikenali.")


def parse_numeric_value(value):
    if pd.isna(value):
        return np.nan

    s = str(value).replace(" ", "")

    s = s.replace(",", ".") if "," in s and "." not in s else s

    try:
        return float(s)
    except:
        return np.nan


def build_feature_matrix(df, model):
    if isinstance(df, pd.Series):
        df = df.to_frame().T

    features = get_model_feature_names(model)
    X = np.zeros((len(df), len(features)))

    for i, name in enumerate(features):
        if name.startswith("tipe_satker="):
            tipe = name.split("=")[1]
            X[:, i] = (df["tipe_satker"] == tipe).astype(float)
        elif name in df.columns:
            X[:, i] = pd.to_numeric(df[name], errors="coerce").fillna(0)
        else:
            X[:, i] = 0

    return X


def predict_from_features(X, model):
    if hasattr(model, "skl_model"):
        predictor = model.skl_model
    else:
        predictor = model

    pred = predictor.predict(X).astype(int)
    prob = predictor.predict_proba(X)[:, 1]

    return pred, prob


# -----------------------------
# Main App
# -----------------------------

def main():
    st.title("📊 Dashboard Realisasi Anggaran")

    # Load dataset
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    dataset_source = uploaded_file if uploaded_file else DATA_PATH

    try:
        df = load_dataset(dataset_source)
    except Exception as e:
        st.error(f"Gagal load data: {e}")
        return

    # Load model
    uploaded_model = st.sidebar.file_uploader("Upload Model", type=["pkl", "pkcls"])

    if uploaded_model:
        model_source = uploaded_model
    elif MODEL_PATH.exists():
        model_source = MODEL_PATH
    elif FALLBACK_MODEL_PATH.exists():
        model_source = FALLBACK_MODEL_PATH
    else:
        st.error("Model tidak ditemukan.")
        return

    model = load_model(model_source)

    if model is None:
        st.stop()

    # Prediction
    X = build_feature_matrix(df, model)
    pred, prob = predict_from_features(X, model)

    df["prediksi"] = np.where(pred == 1, "Ya", "Tidak")
    df["prob"] = prob

    # Summary
    st.subheader("Ringkasan")
    st.write(df.head())

    # Metrics
    st.metric("Akurasi", f"{(pred == df['target']).mean()*100:.2f}%")

    # Chart
    st.subheader("Distribusi Prediksi")
    st.bar_chart(df.groupby("provinsi")["prob"].mean())

    # 🔮 Manual prediction
    st.subheader("Prediksi Manual")

    jumlah_spm = st.number_input("Jumlah SPM", 0, 1000, 30)
    revisi_dipa = st.number_input("Revisi DIPA", 0, 20, 1)
    deviasi = st.slider("Deviasi (%)", 0.0, 100.0, 20.0)
    skor = st.slider("Skor IKPA", 0.0, 100.0, 80.0)

    tipe = st.selectbox("Tipe Satker", df["tipe_satker"].unique())

    input_df = pd.DataFrame([{
        "jumlah_spm": jumlah_spm,
        "revisi_dipa": revisi_dipa,
        "deviasi_rpd_persen": deviasi,
        "skor_ikpa": skor,
        "tipe_satker": tipe
    }])

    X_new = build_feature_matrix(input_df, model)
    pred_new, prob_new = predict_from_features(X_new, model)

    st.metric("Hasil", "YA" if pred_new[0] == 1 else "TIDAK")
    st.progress(float(prob_new[0]))


if __name__ == "__main__":
    main()
