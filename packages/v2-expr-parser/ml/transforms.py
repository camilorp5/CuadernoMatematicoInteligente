# ml/transforms.py
import torchvision.transforms as T

# Normalización calculada sobre HASYv2 (fondo negro ~0, trazo blanco ~1)
HASY_MEAN = [0.1500]
HASY_STD = [0.3200]

def get_train_transforms(image_size: int = 32) -> T.Compose:
    return T.Compose([
        T.Grayscale(num_output_channels=1),
        T.Resize((image_size, image_size)),
        # Rotación estrictamente acotada a ±7 grados para no convertir '+' en 'x'
        T.RandomRotation(degrees=(-7, 7), fill=0),
        T.RandomAffine(
            degrees=0,
            translate=(0.06, 0.06),
            scale=(0.92, 1.08),
            shear=(-4, 4),
            fill=0
        ),
        T.ToTensor(),
        T.Normalize(mean=HASY_MEAN, std=HASY_STD)
    ])

def get_val_transforms(image_size: int = 32) -> T.Compose:
    return T.Compose([
        T.Grayscale(num_output_channels=1),
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=HASY_MEAN, std=HASY_STD)
    ])