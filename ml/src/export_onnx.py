import os
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np

# Reutilizamos la misma estructura del modelo
class MNISTConvNet(nn.Module):
    def __init__(self):
        super(MNISTConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def export_to_onnx():
    pth_path = "../models/best_mnist_cnn.pth"
    onnx_path = "../models/mnist_cnn.onnx"
    web_models_dir = "../../web/public/models"
    web_onnx_path = os.path.join(web_models_dir, "mnist_cnn.onnx")

    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"No se encontró el archivo de pesos {pth_path}. Ejecuta train.py primero.")

    # 1. Cargar el modelo en modo EVAL (desactiva Dropout y entrena BatchNorm)
    model = MNISTConvNet()
    model.load_state_dict(torch.load(pth_path, map_location=torch.device('cpu')))
    model.eval()

    # 2. Generar un Tensor "Dummy" que represente una imagen de entrada: Batch=1, Channel=1, Height=28, Width=28
    dummy_input = torch.randn(1, 1, 28, 28, requires_grad=True)

    # 3. Exportar el grafo a ONNX
    print("🔄 Exportando grafo de PyTorch a formato ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,            # Opset 14 es ampliamente compatible con ONNX Runtime Web
        do_constant_folding=True,    # Optimiza constantes en el grafo
        input_names=['input'],       # Nombre del nodo de entrada en el tensor
        output_names=['output'],     # Nombre del nodo de salida
        dynamic_axes={               # Permite lotes dinámicos (1 imagen o N imágenes a la vez)
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"✅ Modelo guardado en: {onnx_path}")

    # 4. Validar la integridad del grafo ONNX
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("🔍 Validación de estructura ONNX: CORRECTA")

    # 5. Probar Inferencia de Verificación con ONNX Runtime (Python)
    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.detach().numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    
    # Inferencia en PyTorch para comparar
    with torch.no_grad():
        torch_out = model(dummy_input)

    # Verificar que la diferencia entre PyTorch y ONNX sea despreciable (< 1e-5)
    np.testing.assert_allclose(torch_out.numpy(), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("🧪 Verificación numérica (PyTorch vs ONNX): MATCH PERFECTO")

    # 6. Copiar automáticamente a la carpeta de la app Web (/web/public/models)
    os.makedirs(web_models_dir, exist_ok=True)
    import shutil
    shutil.copy(onnx_path, web_onnx_path)
    print(f"🚀 Modelo copiado automáticamente a la App Frontend: {web_onnx_path}")

if __name__ == "__main__":
    export_to_onnx()