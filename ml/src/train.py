import os
import random
import ssl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split


import torchvision
import torchvision.transforms as transforms
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# Bypass de SSL para macOS
ssl._create_default_https_context = ssl._create_unverified_context

# ==========================================
# 1. REPRODUCIBILIDAD (Seed Setup)
# ==========================================
SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# ==========================================
# 2. ARQUITECTURA CNN
# ==========================================
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

# ==========================================
# 3. PIPELINE DE DATOS CON AUGMENTATION
# ==========================================
def get_dataloaders(data_dir='../data', batch_size=64):
    mean_mnist, std_mnist = 0.1307, 0.3081

    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize((mean_mnist,), (std_mnist,))
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean_mnist,), (std_mnist,))
    ])

    full_train_dataset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=eval_transform)

    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    generator = torch.Generator().manual_seed(SEED)
    
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader

# ==========================================
# 4. BUCLE DE ENTRENAMIENTO
# ==========================================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / total, correct / total, all_preds, all_labels

# ==========================================
# 5. MLFLOW EXPERIMENT RUNNER
# ==========================================
def run_experiment(epochs=10, batch_size=64, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Entrenando en el dispositivo: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size)
    model = MNISTConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    mlflow.set_experiment("smart-math-v1-mnist")

    with mlflow.start_run(run_name="CNN_Augmented_Baseline"):
        mlflow.log_params({
            "architecture": "MNISTConvNet",
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "optimizer": "Adam",
            "device": str(device),
            "seed": SEED
        })

        best_val_loss = float('inf')
        os.makedirs("../models", exist_ok=True)
        best_model_path = "../models/best_mnist_cnn.pth"

        for epoch in range(epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            }, step=epoch)

            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)

        model.load_state_dict(torch.load(best_model_path))
        test_loss, test_acc, test_preds, test_targets = evaluate(model, test_loader, criterion, device)

        mlflow.log_metrics({"test_loss": test_loss, "test_acc": test_acc})
        print(f"\n🎯 [TEST EVALUATION] Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}")

        cm = confusion_matrix(test_targets, test_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
        plt.title('Matriz de Confusión en Test Set')
        plt.xlabel('Predicción')
        plt.ylabel('Valor Real')
        plt.tight_layout()
        
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()

        mlflow.log_artifact(cm_path)
        if os.path.exists(cm_path):
            os.remove(cm_path)

        mlflow.pytorch.log_model(
            pytorch_model=model, 
            artifact_path="mnist_cnn_pytorch_model", 
            serialization_format="pickle"
        )
        print(f"✅ Experimento guardado exitosamente en MLflow.")

if __name__ == "__main__":
    run_experiment(epochs=10, batch_size=64, lr=0.001)