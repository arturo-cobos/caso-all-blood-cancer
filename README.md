# Caso ALL — Clasificación de células de sangre (Javeriana, Módulo 5)

Ambiente local para correr el caso del notebook de Colab: dos clasificadores
sobre el dataset *Blood cell Cancer [ALL]* (4 clases) — una **CNN propia** y
**VGG16 preentrenado con fine-tuning**.

## Entorno

- Python 3.12 en `.venv` (creado con `uv`), TensorFlow 2.21 / Keras 3.15, **CPU**
  (esta máquina es aarch64 sin GPU NVIDIA).
- Dependencias exactas en `requirements.txt`.

```bash
source .venv/bin/activate      # siempre, antes de cualquier comando
```

Para recrearlo desde cero en otra máquina:

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

## Datos

> **Los datos no están en este repo.** Pesan 3,5 GB y `data/` está en
> `.gitignore`. El dataset es *Blood cell Cancer [ALL]* (4 clases), público en
> Kaggle; descárgalo y descomprímelo en `data/Blood cell Cancer [ALL]/`, luego
> corre `python src/prep_data.py` para regenerar la partición.

`data/Blood cell Cancer [ALL]/` — 3.266 imágenes (1,7 GB). Distribución por clase:

| Clase | Imágenes |
|---|---|
| Benign | 528 |
| [Malignant] Pre-B | 955 |
| [Malignant] Pro-B | 804 |
| [Malignant] early Pre-B | 979 |

`data/split/` — partición 70/15/15 por clase (semilla 42, igual que el notebook),
ya generada. Las imágenes se guardaron reescaladas a 180 px de lado mayor: los
originales son de 1024×768 a 2560×1920 y decodificarlos en cada época era el
cuello de botella en CPU. Los originales quedan intactos.

Para regenerarla:

```bash
python src/prep_data.py                # 180 px (por defecto)
python src/prep_data.py --full-res     # copia los originales sin reescalar
```

## Ejecución

### Opción A — notebook

```bash
jupyter lab notebooks/caso_all_local.ipynb
```

Mismo flujo CRISP-DM del original, sin `drive.mount` y con las 4 clases en
`categorical` de punta a punta.

### Opción B — scripts

```bash
python src/train_cnn.py --epochs 15                              # CNN propia
python src/train_vgg16.py --epochs-initial 15 --epochs-finetune 5 # VGG16 + fine-tuning
python src/evaluate.py --model cnn                               # accuracy + matriz
python src/evaluate.py --model vgg16
```

Salidas: modelos en `models/`, gráficas e historiales en `outputs/`.

## Tiempos medidos en esta máquina (8 cores, sin GPU)

| Modelo | Por época | 15 épocas |
|---|---|---|
| CNN propia | ~18 s | ~5 min |
| VGG16 fase 1 (base congelada) | ~88 s | ~22 min |
| VGG16 fase 2 (fine-tuning block5) | ~126 s | ~11 min (5 épocas) |

## Diferencias respecto al notebook de Colab

1. Sin `google.colab.drive` ni `gdown`: los datos son locales.
2. El original mezclaba rutas de dos estructuras distintas y dejaba restos de la
   configuración binaria (`class_mode='binary'`, `NUM_CLASSES = 2`,
   `Dense(1, sigmoid)`). Aquí todo el flujo es multiclase de 4 salidas.
3. Las celdas marcadas como redundantes en el original no se trasladaron.
4. `load_and_predict` devolvía `'ALL'`/`'Sano'` con umbral 0.5 sobre la primera
   probabilidad, lo cual es incorrecto para 4 clases softmax. La versión local
   devuelve la clase predicha con su probabilidad.
5. VGG16 hace fine-tuning real: fase 1 con la base congelada y fase 2
   descongelando `block5` con learning rate 1e-5.
