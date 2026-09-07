# ml/export_onnx.py
import json
import logging
import sys
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import torch

from model import MathSymbolCNN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ExportONNX")

BASE_DIR = Path(__file__).resolve().parent.parent  # packages/v2-expr-parser
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

def export_to_onnx(
    weights_path: Path,
    output_onnx_path: Path,
    num_classes: int = 52
):
    if not weights_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de pesos: {weights_path}")

    output_onnx_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Cargando modelo PyTorch desde {weights_path} con {num_classes} clases...")
    model = MathSymbolCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 1, 32, 32, requires_grad=False)

    logger.info(f"Exportando a formato ONNX en: {output_onnx_path}")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_onnx_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )

    # 1. Validación estructural
    logger.info("Verificando integridad del grafo ONNX...")
    onnx_model = onnx.load(str(output_onnx_path))
    onnx.checker.check_model(onnx_model)

    # 2. Validación numérica (PyTorch vs ONNX Runtime)
    logger.info("Validando equivalencia numérica de inferencia...")
    ort_session = ort.InferenceSession(str(output_onnx_path))
    test_tensor = np.random.randn(4, 1, 32, 32).astype(np.float32)

    with torch.no_grad():
        torch_out = model(torch.from_numpy(test_tensor)).numpy()

    ort_inputs = {ort_session.get_inputs()[0].name: test_tensor}
    ort_out = ort_session.run(None, ort_inputs)[0]

    np.testing.assert_allclose(torch_out, ort_out, rtol=1e-03, atol=1e-05)
    logger.info("✅ Exportación validada exitosamente. PyTorch y ONNX producen salidas idénticas.")

if __name__ == "__main__":
    try:
        weights_file = MODELS_DIR / "best_model.pt"
        onnx_file = MODELS_DIR / "math_symbols.onnx"
        classes_file = DATA_DIR / "processed" / "classes.json"

        with open(classes_file, "r", encoding="utf-8") as f:
            classes = json.load(f)

        export_to_onnx(
            weights_path=weights_file,
            output_onnx_path=onnx_file,
            num_classes=len(classes)
        )
    except Exception as err:
        logger.exception(f"Error al exportar a ONNX: {err}")
        sys.exit(1)