"""
automate_Wisnu-anugrah-pratama.py

Script otomatisasi preprocessing dataset Facies Classification
(Hugoton & Panoma Fields, Kansas, USA - SEG 2016 ML Contest).

Konversi dari notebook Eksperimen_Wisnu-anugrah-pratama.ipynb menjadi
fungsi yang dapat dijalankan ulang secara konsisten (reproducible), termasuk
tahapan download dataset dari sumber asli (bukan hanya membaca file lokal).

Dijalankan dari ROOT REPOSITORY:
    uv run python preprocessing/automate_Wisnu-anugrah-pratama.py
"""

import os
import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


import os
import argparse
import urllib.request
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


# ── Konfigurasi default ──────────────────────────────────────────────────────
DATASET_URL          = "https://raw.githubusercontent.com/seg/2016-ml-contest/master/training_data.csv"
DEFAULT_RAW_PATH     = "facies_raw.csv"
DEFAULT_OUTPUT_PATH  = "preprocessing/facies_preprocessing.csv"

CATEGORICAL_COLS = ["Formation", "Well Name"]
NUMERICAL_COLS   = ["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE"]
SCALE_COLS       = ["Depth", "GR", "ILD_log10", "DeltaPHI", "PHIND", "PE"]

FACIES_LABELS = ["SS", "CSiS", "FSiS", "SiSh", "MS", "WS", "D", "PS", "BS"]


def download_data(url: str = DATASET_URL, save_path: str = DEFAULT_RAW_PATH) -> str:
    """
    Mengunduh dataset mentah dari sumber asli (GitHub SEG 2016 ML Contest).
    Identik dengan tahapan download di preprocessing/Eksperimen_Wisnu-anugrah-pratama.ipynb.
    """
    print(f"[download_data] Mengunduh dataset dari: {url}")
    urllib.request.urlretrieve(url, save_path)
    print(f"[download_data] Selesai! File tersimpan di: {save_path}")
    return save_path


def load_data(path: str) -> pd.DataFrame:
    """Memuat dataset mentah dari path CSV. Download otomatis jika belum ada."""
    if not os.path.exists(path):
        print(f"[load_data] File tidak ditemukan di '{path}', mengunduh otomatis...")
        download_data(save_path=path)

    df = pd.read_csv(path)
    print(f"[load_data] Dataset dimuat: {df.shape}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputasi defensif menggunakan median untuk kolom numerik.
    Dataset asli sudah bersih, namun fungsi ini memastikan pipeline
    tetap robust apabila dijalankan pada data baru yang punya missing value.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns

    n_imputed = 0
    for col in numeric_cols:
        n_missing = df[col].isnull().sum()
        if n_missing > 0:
            df[col] = df[col].fillna(df[col].median())
            n_imputed += n_missing

    print(f"[handle_missing_values] Total nilai diimputasi: {n_imputed}")
    return df


def encode_categorical(df: pd.DataFrame) -> tuple:
    """Label encoding untuk kolom kategorikal. Kembalikan df dan dict encoders."""
    df = df.copy()
    encoders = {}

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

    print(f"[encode_categorical] Kolom di-encode: {list(encoders.keys())}")
    return df, encoders


def handle_outliers(df: pd.DataFrame, cols: list = None) -> pd.DataFrame:
    """Capping outlier pada kolom numerik menggunakan metode IQR (winsorizing)."""
    df = df.copy()
    cols = cols or NUMERICAL_COLS

    total_outliers = 0
    for col in cols:
        if col not in df.columns:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        n_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        total_outliers += n_outliers

    print(f"[handle_outliers] Total outlier di-cap: {total_outliers}")
    return df


def scale_features(df: pd.DataFrame) -> tuple:
    """Standard scaling untuk fitur numerik. Kembalikan df dan scaler object."""
    df = df.copy()
    cols_present = [c for c in SCALE_COLS if c in df.columns]

    scaler = StandardScaler()
    df[cols_present] = scaler.fit_transform(df[cols_present])

    print(f"[scale_features] Kolom di-scale: {cols_present}")
    return df, scaler


def preprocess_facies(input_path: str = DEFAULT_RAW_PATH, force_download: bool = False) -> pd.DataFrame:
    """
    Pipeline preprocessing lengkap yang menggabungkan seluruh fungsi di atas
    secara berurutan, identik dengan tahapan manual di notebook eksperimen.

    Args:
        input_path: path ke file CSV raw facies (relatif terhadap root repo)
        force_download: jika True, selalu unduh ulang dataset meskipun file
                         sudah ada (berguna untuk memastikan data selalu
                         terbaru dari sumber, sesuai kriteria Advance)

    Returns:
        DataFrame yang sudah siap dilatih (clean, encoded, scaled)
    """
    print("=" * 60)
    print("MEMULAI PIPELINE PREPROCESSING OTOMATIS")
    print("=" * 60)

    if force_download or not os.path.exists(input_path):
        download_data(save_path=input_path)

    df = load_data(input_path)
    df = handle_missing_values(df)
    df, _encoders = encode_categorical(df)
    df = handle_outliers(df, cols=NUMERICAL_COLS)
    df, _scaler   = scale_features(df)

    print("=" * 60)
    print(f"PIPELINE SELESAI -- Shape akhir: {df.shape}")
    print(f"Missing values  : {df.isnull().sum().sum()}")
    print(f"Duplikat        : {df.duplicated().sum()}")
    print(f"Distribusi kelas Facies:")
    for code, count in df["Facies"].value_counts().sort_index().items():
        label = FACIES_LABELS[int(code) - 1]
        print(f"  Facies {code} ({label:5s}): {count}")
    print("=" * 60)

    return df


def save_preprocessed_data(df: pd.DataFrame, output_path: str = DEFAULT_OUTPUT_PATH) -> None:
    """Simpan DataFrame hasil preprocessing ke CSV."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"[save_preprocessed_data] Dataset tersimpan di: {output_path}")


def main():
    """Entry point untuk menjalankan script via command line."""
    parser = argparse.ArgumentParser(
        description="Download + preprocessing otomatis dataset Facies Classification"
    )
    parser.add_argument(
        "--input", type=str, default=DEFAULT_RAW_PATH,
        help=f"Path ke file CSV raw (default: {DEFAULT_RAW_PATH})"
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT_PATH,
        help=f"Path output CSV hasil preprocessing (default: {DEFAULT_OUTPUT_PATH})"
    )
    parser.add_argument(
        "--force-download", action="store_true",
        help="Selalu unduh ulang dataset dari sumber asli, meskipun file sudah ada"
    )
    args = parser.parse_args()

    df_clean = preprocess_facies(args.input, force_download=args.force_download)
    save_preprocessed_data(df_clean, args.output)


if __name__ == "__main__":
    main()
