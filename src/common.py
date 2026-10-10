"""Configuracion y generadores compartidos por los dos modelos."""
import json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DESTINO = os.path.join(ROOT, "data", "split")
MODELS_DIR = os.path.join(ROOT, "models")
OUTPUTS_DIR = os.path.join(ROOT, "outputs")

IMG_WIDTH = 150
IMG_HEIGHT = 150
BATCH_SIZE = 32


def make_generators(destino=DESTINO, batch_size=BATCH_SIZE, augment=False):
    """train/val/test desde la estructura DESTINO, class_mode='categorical'."""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    if augment:
        train_datagen = ImageDataGenerator(
            rescale=1. / 255, rotation_range=20,
            width_shift_range=0.2, height_shift_range=0.2,
            shear_range=0.2, zoom_range=0.2,
            horizontal_flip=True, fill_mode="nearest",
        )
    else:
        train_datagen = ImageDataGenerator(rescale=1. / 255)
    eval_datagen = ImageDataGenerator(rescale=1. / 255)

    common = dict(target_size=(IMG_HEIGHT, IMG_WIDTH),
                  batch_size=batch_size, class_mode="categorical")
    train = train_datagen.flow_from_directory(
        os.path.join(destino, "train"), shuffle=True, **common)
    val = eval_datagen.flow_from_directory(
        os.path.join(destino, "val"), shuffle=False, **common)
    test = eval_datagen.flow_from_directory(
        os.path.join(destino, "test"), shuffle=False, **common)
    return train, val, test


def reset_history(nombre):
    """Borra el historial previo para que una corrida nueva no se acumule."""
    path = os.path.join(OUTPUTS_DIR, f"history_{nombre}.json")
    if os.path.exists(path):
        os.remove(path)


def save_history(history, nombre):
    """Anade al historial (las dos fases del VGG16 comparten archivo)."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    path = os.path.join(OUTPUTS_DIR, f"history_{nombre}.json")
    prev = {}
    if os.path.exists(path):
        with open(path) as f:
            prev = json.load(f)
    for k, v in history.history.items():
        prev.setdefault(k, []).extend([float(x) for x in v])
    with open(path, "w") as f:
        json.dump(prev, f, indent=2)
    return path


def plot_loss(nombre, titulo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with open(os.path.join(OUTPUTS_DIR, f"history_{nombre}.json")) as f:
        h = json.load(f)
    epochs = range(1, len(h["loss"]) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, h["loss"], marker="o", label="Perdida Entrenamiento")
    plt.plot(epochs, h["val_loss"], marker="o", label="Perdida validacion")
    plt.title(titulo); plt.xlabel("Epoch"); plt.ylabel("Perdida"); plt.legend()
    out = os.path.join(OUTPUTS_DIR, f"loss_{nombre}.png")
    plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close()
    return out
