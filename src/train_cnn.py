"""CNN propia (misma arquitectura del notebook), clasificacion multiclase."""
import argparse, os, time
from common import (IMG_HEIGHT, IMG_WIDTH, MODELS_DIR, make_generators,
                    plot_loss, reset_history, save_history)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--augment", action="store_true")
    args = ap.parse_args()

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
    from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                            ReduceLROnPlateau)

    train_gen, val_gen, _ = make_generators(batch_size=args.batch_size,
                                            augment=args.augment)
    os.makedirs(MODELS_DIR, exist_ok=True)
    reset_history("cnn")
    ckpt = os.path.join(MODELS_DIR, "best_model_scratch.keras")

    model = Sequential([
        Conv2D(32, (3, 3), activation="relu",
               input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation="relu"),
        Dense(train_gen.num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6),
    ]
    t0 = time.time()
    history = model.fit(train_gen, epochs=args.epochs,
                        validation_data=val_gen, callbacks=callbacks)
    print(f"\nEntrenamiento: {(time.time() - t0) / 60:.1f} min")
    save_history(history, "cnn")
    print("Grafica:", plot_loss("cnn", "Perdida en entrenamiento y validacion -- CNN Propia"))
    print("Modelo :", ckpt)


if __name__ == "__main__":
    main()
