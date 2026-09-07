# ml/dataset_setup.py
import json
import os
import shutil
import zipfile
from pathlib import Path
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.model_selection import train_test_split

DATASET_SLUG = "guru001/hasyv2"
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# Diccionario canónico de 52 símbolos: (latex_token, category, display_name)
TARGET_VOCABULARY = {
    # Dígitos (10)
    "0": {"latex": "0", "category": "digit"},
    "1": {"latex": "1", "category": "digit"},
    "2": {"latex": "2", "category": "digit"},
    "3": {"latex": "3", "category": "digit"},
    "4": {"latex": "4", "category": "digit"},
    "5": {"latex": "5", "category": "digit"},
    "6": {"latex": "6", "category": "digit"},
    "7": {"latex": "7", "category": "digit"},
    "8": {"latex": "8", "category": "digit"},
    "9": {"latex": "9", "category": "digit"},
    # Variables (12)
    "a": {"latex": "a", "category": "variable"},
    "b": {"latex": "b", "category": "variable"},
    "c": {"latex": "c", "category": "variable"},
    "k": {"latex": "k", "category": "variable"},
    "m": {"latex": "m", "category": "variable"},
    "n": {"latex": "n", "category": "variable"},
    "x": {"latex": "x", "category": "variable"},
    "y": {"latex": "y", "category": "variable"},
    "z": {"latex": "z", "category": "variable"},
    "\\alpha": {"latex": "\\alpha", "category": "variable"},
    "\\beta": {"latex": "\\beta", "category": "variable"},
    "\\pi": {"latex": "\\pi", "category": "variable"},
    # Operadores básicos y comparación (14)
    "+": {"latex": "+", "category": "operator"},
    "-": {"latex": "-", "category": "operator"},
    "\\times": {"latex": "\\times", "category": "operator"},
    "\\div": {"latex": "\\div", "category": "operator"},
    "=": {"latex": "=", "category": "relation"},
    "\\neq": {"latex": "\\neq", "category": "relation"},
    "<": {"latex": "<", "category": "relation"},
    ">": {"latex": ">", "category": "relation"},
    "\\le": {"latex": "\\le", "category": "relation"},
    "\\ge": {"latex": "\\ge", "category": "relation"},
    "\\pm": {"latex": "\\pm", "category": "operator"},
    "\\in": {"latex": "\\in", "category": "relation"},
    "\\to": {"latex": "\\to", "category": "relation"},
    "\\partial": {"latex": "\\partial", "category": "operator"},
    # Cálculo y delimitadores (16)
    "\\int": {"latex": "\\int", "category": "calculus"},
    "\\sum": {"latex": "\\sum", "category": "calculus"},
    "\\prod": {"latex": "\\prod", "category": "calculus"},
    "\\infty": {"latex": "\\infty", "category": "symbol"},
    "\\sqrt": {"latex": "\\sqrt{}", "category": "operator"},
    "(": {"latex": "(", "category": "delimiter"},
    ")": {"latex": ")", "category": "delimiter"},
    "[": {"latex": "[", "category": "delimiter"},
    "]": {"latex": "]", "category": "delimiter"},
    "\\{": {"latex": "\\{", "category": "delimiter"},
    "\\}": {"latex": "\\}", "category": "delimiter"},
    "!": {"latex": "!", "category": "operator"},
    "/": {"latex": "/", "category": "operator"},
    "\\theta": {"latex": "\\theta", "category": "variable"},
    "\\lambda": {"latex": "\\lambda", "category": "variable"},
    "\\sigma": {"latex": "\\sigma", "category": "variable"}
}

def download_and_extract_dataset():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    
    zip_target = RAW_DIR / "hasy-data.zip"
    if not (RAW_DIR / "hasy-data-labels.csv").exists():
        print(f"Descargando {DATASET_SLUG} desde Kaggle...")
        api.dataset_download_files(DATASET_SLUG, path=RAW_DIR, unzip=False)
        
        # Descomprimir
        print("Descomprimiendo archivos...")
        downloaded_zip = list(RAW_DIR.glob("*.zip"))[0]
        with zipfile.ZipFile(downloaded_zip, 'r') as zip_ref:
            zip_ref.extractall(RAW_DIR)
        downloaded_zip.unlink()
        print("Extracción completada.")
    else:
        print("Dataset ya descargado y verificado en caché.")

def prepare_splits(val_size: float = 0.15, seed: int = 42):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RAW_DIR / "hasy-data-labels.csv"
    df = pd.read_csv(csv_path)

    # Filtrar únicamente los símbolos de interés definidos en el diccionario
    valid_latex = set(TARGET_VOCABULARY.keys())
    df_filtered = df[df["latex"].isin(valid_latex)].copy()

    # Mapeo id numérico -> latex
    unique_symbols = sorted(df_filtered["latex"].unique())
    label_to_id = {sym: idx for idx, sym in enumerate(unique_symbols)}
    id_to_metadata = {
        idx: {
            "index": idx,
            "symbol": sym,
            "latex": TARGET_VOCABULARY[sym]["latex"],
            "category": TARGET_VOCABULARY[sym]["category"]
        }
        for sym, idx in label_to_id.items()
    }

    df_filtered["label_id"] = df_filtered["latex"].map(label_to_id)

    # Split estratificado
    train_df, val_df = train_test_split(
        df_filtered,
        test_size=val_size,
        random_state=seed,
        stratify=df_filtered["label_id"]
    )

    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)

    # Exportar metadatos para uso de Next.js y el pipeline PyTorch
    classes_path = PROCESSED_DIR / "classes.json"
    with open(classes_path, "w", encoding="utf-8") as f:
        json.dump(id_to_metadata, f, indent=2, ensure_ascii=False)

    print(f"Procesamiento listo: {len(train_df)} muestras de train, {len(val_df)} de val.")
    print(f"Metadatos de clases exportados en: {classes_path}")

if __name__ == "__main__":
    download_and_extract_dataset()
    prepare_splits()