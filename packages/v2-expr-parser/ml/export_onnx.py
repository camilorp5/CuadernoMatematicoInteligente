# ml/export_onnx.py
import json
import torch
import onnx
import onnxruntime as ort
import numpy as np
from model import MathSymbolCNN
from pathlib import Path
import pandas as pd
from PIL import Image

def export_to_onnx(
    weights_path: str,
    output_onnx_path: str,
    class_map_path: str,
    num_classes: int = 52
):
    model = MathSymbolCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    # Tensor dummy: [Batch_Size, Canales, Alto, Ancho]
    dummy_input = torch.randn(1, 1, 32, 32, requires_grad=False)

    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=14,  # Alta compatibilidad con onnxruntime-web WASM
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )

    # 1. Verificación estructural con ONNX
    onnx_model = onnx.load(output_onnx_path)
    onnx.checker.check_model(onnx_model)

    # 2. Verificación de equivalencia numérica PyTorch vs ONNX Runtime
    ort_session = ort.InferenceSession(output_onnx_path)
    test_tensor = np.random.randn(4, 1, 32, 32).astype(np.float32)

    with torch.no_grad():
        torch_out = model(torch.from_numpy(test_tensor)).numpy()

    ort_inputs = {ort_session.get_inputs()[0].name: test_tensor}
    ort_out = ort_session.run(None, ort_inputs)[0]

    np.testing.assert_allclose(torch_out, ort_out, rtol=1e-03, atol=1e-05)
    print(f"Exportación exitosa. Modelo verificado y validado en: {output_onnx_path}")

if __name__ == "__main__":
    export_to_onnx(
        weights_path="/packages/v2-expr-parser/models/best_model.pt",
        output_onnx_path="/packages/v2-expr-parser/models/math_symbols.onnx",
        class_map_path="/packages/v2-expr-parser/models/classes.json"
    )