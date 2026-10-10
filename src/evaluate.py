"""Evalua un modelo guardado sobre el conjunto de test: accuracy + matriz."""
import argparse, os
import numpy as np
from common import MODELS_DIR, OUTPUTS_DIR, make_generators


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="ruta .keras o 'cnn'/'vgg16'")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    atajos = {"cnn": "best_model_scratch.keras", "vgg16": "best_model_vgg16.keras"}
    path = os.path.join(MODELS_DIR, atajos[args.model]) if args.model in atajos else args.model
    nombre = os.path.splitext(os.path.basename(path))[0]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import tensorflow as tf
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix)

    _, _, test_gen = make_generators(batch_size=args.batch_size)
    class_names = list(test_gen.class_indices.keys())
    model = tf.keras.models.load_model(path)

    predictions = model.predict(test_gen)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_gen.classes

    acc = accuracy_score(true_classes, predicted_classes)
    print(f"\nAccuracy en test ({nombre}): {acc:.4f}\n")
    print(classification_report(true_classes, predicted_classes,
                                target_names=class_names, digits=4))

    mtx = confusion_matrix(true_classes, predicted_classes)
    plt.subplots(figsize=(14, 8))
    sns.heatmap(mtx, cmap="Blues", xticklabels=class_names,
                yticklabels=class_names, annot=True, annot_kws={"size": 10},
                fmt="d", linewidths=.1)
    plt.grid(False); plt.yticks(rotation=0)
    plt.xlabel("Predicted labels"); plt.ylabel("True labels")
    plt.title(f"Matriz de confusion -- {nombre} (acc={acc:.3f})")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    out = os.path.join(OUTPUTS_DIR, f"confusion_{nombre}.png")
    plt.savefig(out, dpi=120, bbox_inches="tight"); plt.close()
    print("Matriz de confusion:", out)


if __name__ == "__main__":
    main()
