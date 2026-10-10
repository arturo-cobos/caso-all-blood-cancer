#!/usr/bin/env bash
# Corre el caso completo: CNN propia + VGG16 con fine-tuning, y evalua ambos.
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
export TF_CPP_MIN_LOG_LEVEL=2

[ -d data/split ] || python src/prep_data.py

echo "=== CNN propia ==="
python src/train_cnn.py --epochs "${EPOCHS_CNN:-15}"
python src/evaluate.py --model cnn

echo "=== VGG16 preentrenado + fine-tuning ==="
python src/train_vgg16.py --epochs-initial "${EPOCHS_VGG:-15}" --epochs-finetune "${EPOCHS_FT:-5}"
python src/evaluate.py --model vgg16

echo "Listo. Modelos en models/, graficas en outputs/"
