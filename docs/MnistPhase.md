# 🧠 Phase 01: MNIST Convolutional Neural Network (CNN) & MLOps Pipeline

## 📌 Executive Summary

The primary objective of this phase is to move from theoretical foundations to a robust, scalable, and standardized Machine Learning Pipeline.

Using the classic **MNIST Benchmark Dataset**, this phase covers the complete lifecycle of an engineering-grade ML project:

1. **Mathematical & Architectural Foundations:** Convolution, Pooling, and Dense Layers.
2. **Exploratory Data Analysis (EDA):** Statistical verification of dataset distribution and data normalization bounds.
3. **Model Development:** PyTorch-based CNN implementation.
4. **Experiment Tracking & MLOps:** Experiment logging and metric tracking using **MLflow**.
5. **Model Packaging & Export:** Standardizing trained artifacts into **ONNX** (Open Neural Network Exchange) format for production inference.

---

## 🏗️ 1. Technical Stack & Dependencies

### Core Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate

# Core Libraries
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn
pip install mlflow onnx onnxruntime
```

### Verified Version Matrix

| Component | Version |
|---|---|
| Python | 3.11.x |
| PyTorch | 2.2.2 |
| Torchvision | 0.17.2 |
| NumPy | 1.26.4 |
| MLflow | Latest stable |
| ONNX Runtime | Latest stable |

> **Note:** NumPy is constrained to `<2.0` for ecosystem compatibility.

---

## 📊 2. Dataset Pipeline & Exploratory Data Analysis (EDA)

The MNIST dataset consists of **70,000 grayscale images** of handwritten digits, each with dimensions of **28 × 28 pixels** and a single channel.

### 2.1 Data Splitting Strategy

To avoid data leakage and ensure unbiased evaluation, the standard 60,000 training images were deterministically split into:

- **48,000 training images (80%)**
- **12,000 validation images (20%)**
- **10,000 test images**

The final dataset distribution is:

$$
\text{Total Dataset}
=
\underbrace{\text{Train }(48,000) + \text{Validation }(12,000)}_{\text{Full Train }(60,000)}
+
\text{Test }(10,000)
$$

### Python Implementation

```python
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import random_split

data_dir = "../data"

raw_transform = transforms.Compose([
    transforms.ToTensor()
])

full_train_dataset = torchvision.datasets.MNIST(
    root=data_dir,
    train=True,
    download=True,
    transform=raw_transform
)

test_dataset = torchvision.datasets.MNIST(
    root=data_dir,
    train=False,
    download=True,
    transform=raw_transform
)

# Deterministic split
generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset = random_split(
    full_train_dataset,
    [48000, 12000],
    generator=generator
)
```

---

### 2.2 Stratification & Class Balance Verification

An empirical check was performed to verify the label distribution across the different splits.

| Digit | Train (%) | Validation (%) | Test (%) | Status |
|---:|---:|---:|---:|---|
| 0 | 9.86% | 9.90% | 9.80% | Balanced |
| 1 | 11.21% | 11.33% | 11.35% | Balanced |
| 2 | 9.92% | 9.97% | 10.32% | Balanced |
| 3 | 10.18% | 10.38% | 10.10% | Balanced |
| 4 | 9.75% | 9.68% | 9.82% | Balanced |
| 5 | 9.10% | 8.77% | 8.92% | Balanced |
| 6 | 9.90% | 9.72% | 9.58% | Balanced |
| 7 | 10.38% | 10.68% | 10.28% | Balanced |
| 8 | 9.74% | 9.81% | 9.74% | Balanced |
| 9 | 9.95% | 9.78% | 10.09% | Balanced |

The distribution is sufficiently uniform for the intended classification task.

---

### 2.3 Pixel Normalization Parameters

After applying `transforms.ToTensor()`, raw pixel values are converted to the range:

$$
[0.0, 1.0]
$$

The global channel-wise mean and standard deviation calculated over the training set are:

$$
\mu = 0.1307
$$

$$
\sigma = 0.3082
$$

These values are used to normalize the input images.

### PyTorch Transformation Pipeline

```python
train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3082,))
])
```

---

# 📐 3. Convolutional Neural Network Architecture

## 3.1 Mathematical Principles of CNNs

A Convolutional Neural Network (CNN) is particularly suitable for image classification because it can automatically learn spatial features such as:

- Edges
- Curves
- Textures
- Shapes
- Higher-level visual patterns

Given an input image tensor:

$$
X \in \mathbb{R}^{C \times H \times W}
$$

where:

- $C = 1$ → number of channels
- $H = 28$ → image height
- $W = 28$ → image width

---

## 3.2 2D Convolution Operator

A simplified representation of the 2D convolution operation is:

$$
(X * K)_{i,j}
=
\sum_m \sum_n
X_{i+m,j+n}K_{m,n}
+
b
$$

where:

- $X$ = input feature map
- $K$ = convolution kernel/filter
- $b$ = bias
- $(i,j)$ = spatial position

The convolution operation allows the network to detect local patterns in the image.

---

## 3.3 Feature Map Dimensionality

The output dimensions of a convolution operation are calculated using:

$$
H_{out}
=
\left\lfloor
\frac{H_{in}+2P-K}{S}
\right\rfloor
+1
$$

and similarly:

$$
W_{out}
=
\left\lfloor
\frac{W_{in}+2P-K}{S}
\right\rfloor
+1
$$

where:

- $P$ = Padding
- $K$ = Kernel Size
- $S$ = Stride

For the architecture used in this project:

- Kernel size = $3 \times 3$
- Stride = $1$
- Padding = $1$

Therefore, the spatial dimensions are preserved after each convolution.

---

## 3.4 CNN Architecture

The model follows the architecture:

```text
Input
(1 × 28 × 28)
       │
       ▼
Conv2D
1 → 32 channels
3×3 kernel
       │
       ▼
BatchNorm
       │
       ▼
ReLU
       │
       ▼
MaxPool 2×2
       │
       ▼
(32 × 14 × 14)
       │
       ▼
Conv2D
32 → 64 channels
3×3 kernel
       │
       ▼
BatchNorm
       │
       ▼
ReLU
       │
       ▼
MaxPool 2×2
       │
       ▼
(64 × 7 × 7)
       │
       ▼
Flatten
       │
       ▼
Linear
3136 → 128
       │
       ▼
ReLU
       │
       ▼
Dropout 0.25
       │
       ▼
Linear
128 → 10
       │
       ▼
Output
10 classes
```

---

## 3.5 Model Definition: `MNISTConvNet`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTConvNet(nn.Module):

    def __init__(self):
        super(MNISTConvNet, self).__init__()

        # Layer 1:
        # Input:  (1, 28, 28)
        # Output: (32, 28, 28)
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.bn1 = nn.BatchNorm2d(32)

        # Layer 2:
        # Input:  (32, 14, 14)
        # Output: (64, 14, 14)
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(64)

        # Pooling
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # Regularization
        self.dropout = nn.Dropout(0.25)

        # Classification Head
        self.fc1 = nn.Linear(
            64 * 7 * 7,
            128
        )

        self.fc2 = nn.Linear(
            128,
            10
        )

    def forward(self, x):

        # Conv Block 1
        x = self.pool(
            F.relu(
                self.bn1(
                    self.conv1(x)
                )
            )
        )

        # Shape: (Batch, 32, 14, 14)

        # Conv Block 2
        x = self.pool(
            F.relu(
                self.bn2(
                    self.conv2(x)
                )
            )
        )

        # Shape: (Batch, 64, 7, 7)

        # Flatten
        x = x.view(
            -1,
            64 * 7 * 7
        )

        # Dense Layers
        x = F.relu(
            self.fc1(x)
        )

        x = self.dropout(x)

        x = self.fc2(x)

        return x
```

---

# 📈 4. Experiment Tracking with MLflow

[MLflow](https://mlflow.org/) is integrated into the project to track:

- Hyperparameters
- Training loss
- Validation loss
- Validation accuracy
- Model artifacts
- Experiment runs
- Model lineage

This provides reproducibility and makes it possible to compare different experiments systematically.

---

## 4.1 MLflow Configuration

```python
import mlflow
import mlflow.pytorch


mlflow.set_experiment(
    "MNIST_CNN_Phase"
)


with mlflow.start_run(
    run_name="CNN_Baseline_v1"
):

    # Hyperparameters
    params = {
        "batch_size": 64,
        "learning_rate": 0.001,
        "epochs": 10,
        "optimizer": "Adam",
        "loss_function": "CrossEntropyLoss"
    }

    # Log hyperparameters
    mlflow.log_params(params)

    # Training Loop
    for epoch in range(params["epochs"]):

        # ... training code ...

        mlflow.log_metric(
            "train_loss",
            train_loss,
            step=epoch
        )

        mlflow.log_metric(
            "val_loss",
            val_loss,
            step=epoch
        )

        mlflow.log_metric(
            "val_accuracy",
            val_accuracy,
            step=epoch
        )

    # Log PyTorch model
    mlflow.pytorch.log_model(
        model,
        "model"
    )
```

---

## 4.2 Metrics Tracked

The main metrics tracked during training are:

| Metric | Description |
|---|---|
| `train_loss` | Training loss per epoch |
| `val_loss` | Validation loss per epoch |
| `val_accuracy` | Validation classification accuracy |

The MLflow experiment allows the evolution of these metrics to be visualized and compared between runs.

---

# 📦 5. Model Export & Interoperability with ONNX

The trained PyTorch model is exported to **ONNX (Open Neural Network Exchange)**.

The objective is to decouple the training framework from the production inference environment.

This allows the trained model to be deployed in environments where PyTorch is not necessarily available.

Potential environments include:

- C++
- Rust
- Node.js
- Python
- Embedded systems
- Cloud inference services

---

## 5.1 Exporting PyTorch to ONNX

```python
import torch
import torch.onnx


# Switch model to evaluation mode
model.eval()


# Dummy input with the same dimensions as MNIST
dummy_input = torch.randn(
    1,
    1,
    28,
    28,
    device="cpu"
)


onnx_path = "mnist_cnn.onnx"


torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {
            0: "batch_size"
        },
        "output": {
            0: "batch_size"
        }
    }
)


print(
    f"Model successfully exported to {onnx_path}"
)
```

---

## 5.2 ONNX Runtime Inference Verification

After exporting the model, the ONNX file should be tested using ONNX Runtime.

```python
import onnxruntime as ort
import numpy as np


# Initialize ONNX Runtime inference session
ort_session = ort.InferenceSession(
    "mnist_cnn.onnx"
)


# Prepare input
dummy_ort_input = np.random.randn(
    1,
    1,
    28,
    28
).astype(np.float32)


# Run inference
outputs = ort_session.run(
    None,
    {
        "input": dummy_ort_input
    }
)


print(
    "ONNX Output Shape:",
    outputs[0].shape
)
```

Expected output shape:

```text
ONNX Output Shape: (1, 10)
```

The `10` corresponds to the ten MNIST classes:

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9
```

---

# 🧪 6. Training Pipeline Overview

The complete training process can be summarized as follows:

```text
MNIST Dataset
      │
      ▼
Data Loading
      │
      ▼
Train / Validation / Test Split
      │
      ▼
Normalization
      │
      ▼
CNN Model
      │
      ├── Conv2D
      ├── BatchNorm
      ├── ReLU
      ├── MaxPooling
      ├── Conv2D
      ├── BatchNorm
      ├── ReLU
      ├── MaxPooling
      ├── Flatten
      ├── Dense
      ├── Dropout
      └── Dense
      │
      ▼
Training
      │
      ▼
Validation
      │
      ▼
MLflow Tracking
      │
      ▼
Trained Model
      │
      ▼
ONNX Export
      │
      ▼
ONNX Runtime Verification
      │
      ▼
Production-Ready Artifact
```

---

# 🎯 7. Key Takeaways & Milestones

## Deterministic Pipeline

A fixed random seed is used during the train/validation split:

```python
generator = torch.Generator().manual_seed(42)
```

This makes the split reproducible.

---

## Modern CNN Architecture

The model combines:

- Convolutional layers
- Batch Normalization
- ReLU activation
- Max Pooling
- Fully Connected layers
- Dropout regularization

This creates a compact architecture suitable for the MNIST classification task.

---

## MLOps Integration

MLflow provides experiment management and allows the project to track:

- Hyperparameters
- Training metrics
- Validation metrics
- Model artifacts
- Experiment runs

This improves reproducibility and facilitates model comparison.

---

## Production Readiness

The trained PyTorch model is exported to ONNX, allowing inference through ONNX Runtime and facilitating integration with environments outside the original PyTorch training stack.

---

# 📁 8. Suggested Repository Structure

A recommended project structure is:

```text
project/
│
├── data/
│   └── MNIST/
│
├── notebooks/
│   └── mnist_eda.ipynb
│
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── export_onnx.py
│
├── models/
│   ├── mnist_cnn.pth
│   └── mnist_cnn.onnx
│
├── mlruns/
│
├── requirements.txt
│
├── MnistPhase.md
│
└── README.md
```

---

# 🚀 9. Next Steps

The following extensions can be implemented after completing the baseline MNIST CNN:

### 9.1 Model Evaluation

Add:

- Test accuracy
- Precision
- Recall
- F1-score
- Classification report
- Confusion matrix

### 9.2 Visualization

Generate:

- Training loss curves
- Validation loss curves
- Accuracy curves
- Confusion matrix
- Incorrectly classified examples
- Feature map visualizations

### 9.3 Hyperparameter Experiments

Experiment with:

- Learning rate
- Batch size
- Number of convolutional filters
- Dropout rate
- Optimizers
- Number of epochs
- Kernel size

Each experiment can be tracked with MLflow.

### 9.4 Model Deployment

The ONNX model can later be exposed through:

- FastAPI
- Docker
- ONNX Runtime
- REST API
- Cloud inference

---

# 🏁 Conclusion

This phase establishes the transition from theoretical Machine Learning concepts toward an engineering-oriented Machine Learning workflow.

The project now includes:

- Mathematical foundations of CNNs
- MNIST dataset analysis
- Deterministic train/validation splitting
- Input normalization
- CNN architecture design
- PyTorch model implementation
- MLflow experiment tracking
- ONNX model export
- ONNX Runtime inference verification

The resulting pipeline provides a solid foundation for subsequent phases involving **model evaluation, deployment, containerization, APIs, monitoring, and more advanced MLOps practices**.
