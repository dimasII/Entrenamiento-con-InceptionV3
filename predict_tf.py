import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

DEFAULT_MODEL = "results_tf/best_model.keras"
DEFAULT_CLASSES = "class_names.json"
DEFAULT_THRESHOLD = 0.5


def parse_args():
    ap = argparse.ArgumentParser(description="Clasifica una imagen de residuo.")
    ap.add_argument("image", help="ruta de la imagen a clasificar")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--class-names", default=DEFAULT_CLASSES)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help="umbral de confianza minima (default 0.5). Ajustable: "
                         "subelo (0.7-0.8) si prefieres menos falsos positivos.")
    ap.add_argument("--top-k", type=int, default=3,
                    help="muestra las K clases con mayor puntuacion")
    return ap.parse_args()


def load_and_preprocess(path, img_size=299):
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.applications.inception_v3 import preprocess_input

    raw = tf.io.read_file(str(path))
    # decode_image acepta jpeg/png/etc. y normaliza a 3 canales
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [img_size, img_size])
    img = preprocess_input(tf.cast(img, tf.float32))
    return tf.expand_dims(img, 0), np  # lote de 1


def main():
    args = parse_args()
    assert Path(args.image).is_file(), f"No existe la imagen: {args.image}"
    assert Path(args.model).is_file(), \
        f"No existe el modelo: {args.model} (entrena primero con train_tf.py)"
    assert Path(args.class_names).is_file(), \
        f"No existe {args.class_names} (lo genera train_tf.py)"

    import tensorflow as tf

    with open(args.class_names) as f:
        classes = json.load(f)

    model = tf.keras.models.load_model(args.model)
    batch, _ = load_and_preprocess(args.image)
    scores = [float(v) for v in model.predict(batch, verbose=0)[0]]

    order = sorted(range(len(classes)), key=lambda i: scores[i], reverse=True)
    pred = order[0]
    print(f"Imagen: {args.image}")
    print(f"Prediccion: {classes[pred]}  (confianza {scores[pred]:.3f})")
    print(f"Top-{args.top_k}:")
    for i in order[:args.top_k]:
        print(f"  {classes[i]:12s} {scores[i]:.4f}")

    if scores[pred] < args.threshold:
        print(f"\nADVERTENCIA: confianza {scores[pred]:.3f} < umbral "
              f"{args.threshold}. Resultado NO confiable: revisa manualmente. "
              f"El softmax no es garantia de acierto.")
    else:
        print(f"\nConfianza sobre el umbral ({args.threshold}). Aun asi, el "
              f"softmax no equivale a probabilidad clinica ni garantiza acierto.")


if __name__ == "__main__":
    main()
