# ml/train.py
import json
import logging
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import MathSymbolCNN
from transforms import get_train_transforms, get_val_transforms

# Configuración de logging estructurado
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("TrainSymbols")

# Resolución de rutas relativas a la raíz del paquete
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def generate_confusion_matrix_artifact(y_true, y_pred, class_names, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(24, 20))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicción")
    plt.ylabel("Verdadero")
    plt.title("Matriz de Confusión - Símbolos Matemáticos")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


class HasyDataset(Dataset):
    def __init__(self, csv_path: Path, raw_dir: Path, transform=None):
        if not csv_path.exists():
            raise FileNotFoundError(f"Archivo de dataset no encontrado: {csv_path}")
        self.df = pd.read_csv(csv_path)
        self.raw_dir = raw_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_rel_path = row["path"]
        img_path = self.raw_dir / img_rel_path

        if not img_path.exists():
            img_path = Path(img_rel_path)

        if not img_path.exists():
            raise FileNotFoundError(f"Imagen no encontrada en disco: {img_path}")

        try:
            image = Image.open(img_path).convert("L")
        except Exception as e:
            logger.error(f"Error cargando imagen {img_path}: {e}")
            raise e

        label = int(row["label_id"])

        if self.transform:
            image = self.transform(image)

        return image, label


def train_pipeline(
    train_dataset: Dataset,
    val_dataset: Dataset,
    class_map: dict,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    device_name: str = "cpu"
) -> Path:
    seed_everything(42)
    device = torch.device(device_name)
    num_classes = len(class_map)

    # Garantizar existencia de directorios de destino
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    best_model_path = MODELS_DIR / "best_model.pt"

    logger.info(f"Cargando DataLoaders (workers=0, pin_memory=False para CPU)...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    model = MathSymbolCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    mlflow.set_experiment("SmartMathNotebook_Phase2_Symbols")

    with mlflow.start_run(run_name="CNN_52_Symbols_Run"):
        mlflow.log_params({
            "num_classes": num_classes,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "optimizer": "AdamW",
            "loss": "CrossEntropy_LabelSmoothing_0.05",
            "device": device_name
        })

        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            # Fase Entrenamiento
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            
            pbar = tqdm(train_loader, desc=f"Época {epoch:02d}/{epochs:02d} [Train]", leave=False)
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels).item()
                total += labels.size(0)
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            scheduler.step()
            train_loss = running_loss / total
            train_acc = correct / total

            # Fase Validación
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            all_preds, all_labels = [], []

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * images.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += torch.sum(preds == labels).item()
                    val_total += labels.size(0)

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            val_loss = val_loss / val_total
            val_acc = val_correct / val_total

            logger.info(
                f"Época {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {train_loss:.4f} - Acc: {train_acc*100:.2f}% | "
                f"Val Loss: {val_loss:.4f} - Acc: {val_acc*100:.2f}%"
            )

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "lr": scheduler.get_last_lr()[0]
            }, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                try:
                    torch.save(model.state_dict(), best_model_path)
                    logger.info(f"  └─ Nuevo mejor modelo guardado en {best_model_path} (Val Loss: {val_loss:.4f})")
                except Exception as e:
                    logger.error(f"Error al guardar checkpoint: {e}")
                    raise e

        # Evaluación Final y Artefactos
        logger.info("Generando artefactos de evaluación final...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        class_names = [class_map[i] for i in range(num_classes)]

        cm_path = ARTIFACTS_DIR / "confusion_matrix.png"
        generate_confusion_matrix_artifact(all_labels, all_preds, class_names, cm_path)
        mlflow.log_artifact(str(cm_path))

        # Reporte sin advertencias de división por cero
        report_path = ARTIFACTS_DIR / "classification_report.json"
        report = classification_report(
            all_labels, 
            all_preds, 
            target_names=class_names, 
            output_dict=True,
            zero_division=0  # <--- silencia los warnings de precisión indefinida
        )
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(str(report_path))

        # Guardar en MLflow compatible con PyTorch < 2.4 (evita 'pt2')
        try:
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="pytorch_model",
                serialization_format="cloudpickle"  # <--- compatible con PyTorch 2.2
            )
        except Exception as e:
            logger.warning(f"No se pudo registrar el modelo en MLflow, pero el archivo local está a salvo: {e}")

    return best_model_path


if __name__ == "__main__":
    try:
        # Selección segura de CPU
        device = "cpu"
        logger.info(f"Iniciando pipeline en dispositivo: {device}")

        # Verificación de archivos requeridos
        classes_file = DATA_DIR / "processed" / "classes.json"
        train_csv = DATA_DIR / "processed" / "train.csv"
        val_csv = DATA_DIR / "processed" / "val.csv"
        raw_dir = DATA_DIR / "raw"

        for f in [classes_file, train_csv, val_csv]:
            if not f.exists():
                logger.error(f"Archivo requerido no encontrado: {f}")
                logger.error("Asegúrate de haber ejecutado 'python ml/dataset_setup.py' primero.")
                sys.exit(1)

        with open(classes_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        class_map = {int(k): v["symbol"] for k, v in metadata.items()}

        train_ds = HasyDataset(train_csv, raw_dir=raw_dir, transform=get_train_transforms())
        val_ds = HasyDataset(val_csv, raw_dir=raw_dir, transform=get_val_transforms())

        logger.info(f"Dataset validado: {len(train_ds)} train, {len(val_ds)} val, {len(class_map)} clases.")

        best_path = train_pipeline(
            train_dataset=train_ds,
            val_dataset=val_ds,
            class_map=class_map,
            epochs=15,
            batch_size=64,
            lr=1e-3,
            device_name=device
        )
        logger.info(f"Entrenamiento completado exitosamente.")
        logger.info(f"Ruta de pesos listos para exportar: {best_path}")

    except KeyboardInterrupt:
        logger.warning("Entrenamiento interrumpido manualmente por el usuario.")
        sys.exit(0)
    except Exception as err:
        logger.exception(f"Fallo crítico durante el pipeline de entrenamiento: {err}")
        sys.exit(1)