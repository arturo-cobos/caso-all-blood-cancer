"""VGG16 preentrenado (ImageNet): fase 1 congelado + fase 2 fine-tuning."""
import argparse, os, time
from common import (IMG_HEIGHT, IMG_WIDTH, MODELS_DIR, make_generators,
                    plot_loss, reset_history, save_history)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs-initial", type=int, default=15)
    ap.add_argument("--epochs-finetune", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                            ReduceLROnPlateau)

    train_gen, val_gen, _ = make_generators(batch_size=args.batch_size)
    os.makedirs(MODELS_DIR, exist_ok=True)
    reset_history("vgg16")
    ckpt = os.path.join(MODELS_DIR, "best_model_vgg16.keras")

    base_model = VGG16(weights="imagenet", include_top=False,
                       input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    base_model.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(train_gen.num_classes, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6),
    ]

    t0 = time.time()
    print("\n=== Fase 1: cabeza nueva, base congelada ===")
    h1 = model.fit(train_gen, epochs=args.epochs_initial,
                   validation_data=val_gen, callbacks=callbacks)
    save_history(h1, "vgg16")

    if args.epochs_finetune > 0:
        print("\n=== Fase 2: fine-tuning del bloque block5 ===")
        base_model.trainable = True
        for layer in base_model.layers:
            layer.trainable = layer.name.startswith("block5")
        model.compile(optimizer=Adam(learning_rate=1e-5),
                      loss="categorical_crossentropy", metrics=["accuracy"])
        h2 = model.fit(train_gen, epochs=args.epochs_finetune,
                       validation_data=val_gen, callbacks=callbacks)
        save_history(h2, "vgg16")

    print(f"\nEntrenamiento total: {(time.time() - t0) / 60:.1f} min")
    print("Grafica:", plot_loss("vgg16", "Perdida en entrenamiento y validacion -- VGG16"))
    print("Modelo :", ckpt)


if __name__ == "__main__":
    main()
