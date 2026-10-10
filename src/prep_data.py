"""Divide el dataset original en train/val/test (70/15/15) como en el notebook.

Por defecto redimensiona las imagenes a 180px de lado mayor, porque los
originales son de hasta 5 MP y decodificarlos en cada epoca domina el tiempo
de entrenamiento en CPU. Con --full-res se copian tal cual (igual que Colab).
"""
import argparse, os, random, shutil
from concurrent.futures import ProcessPoolExecutor
from PIL import Image
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIGEN = os.path.join(ROOT, "data", "Blood cell Cancer [ALL]")
DESTINO = os.path.join(ROOT, "data", "split")
SPLITS = ("train", "val", "test")


def procesar(tarea):
    src, dst, size = tarea
    if size is None:
        shutil.copy(src, dst)
        return
    with Image.open(src) as im:
        im = im.convert("RGB")
        im.thumbnail((size, size), Image.LANCZOS)
        im.save(dst, "JPEG", quality=92)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origen", default=ORIGEN)
    ap.add_argument("--destino", default=DESTINO)
    ap.add_argument("--size", type=int, default=180, help="lado mayor en px")
    ap.add_argument("--full-res", action="store_true", help="copiar sin redimensionar")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    size = None if args.full_res else args.size

    clases = sorted(d for d in os.listdir(args.origen)
                    if os.path.isdir(os.path.join(args.origen, d)))
    print(f"Clases detectadas: {clases}")
    random.seed(args.seed)

    if os.path.exists(args.destino):
        shutil.rmtree(args.destino)
    for split in SPLITS:
        for cls in clases:
            os.makedirs(os.path.join(args.destino, split, cls), exist_ok=True)

    tareas, resumen = [], {}
    for cls in clases:
        imgs = sorted(os.listdir(os.path.join(args.origen, cls)))
        random.shuffle(imgs)
        n = len(imgs)
        n_train, n_val = int(n * 0.70), int(n * 0.15)
        reparto = {
            "train": imgs[:n_train],
            "val": imgs[n_train:n_train + n_val],
            "test": imgs[n_train + n_val:],
        }
        resumen[cls] = {k: len(v) for k, v in reparto.items()}
        for split, lista in reparto.items():
            for img in lista:
                tareas.append((os.path.join(args.origen, cls, img),
                               os.path.join(args.destino, split, cls, img), size))

    with ProcessPoolExecutor() as ex:
        list(tqdm(ex.map(procesar, tareas, chunksize=16),
                  total=len(tareas), desc="procesando"))

    print("\nImagenes por clase y split:")
    for cls, r in resumen.items():
        print(f"  {cls:26s} train={r['train']:4d} val={r['val']:4d} test={r['test']:4d}")
    print(f"\nListo -> {args.destino}")


if __name__ == "__main__":
    main()
