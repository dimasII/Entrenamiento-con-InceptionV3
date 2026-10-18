"""
Clasificador de residuos con InceptionV3 (TensorFlow/Keras).

Etapas:
  1) Base InceptionV3 congelada + cabeza de 12 clases.
  2) Fine-tuning descongelando las ultimas capas (BN siempre congeladas) con LR baja.
La eleccion entre etapas se hace con F1 macro en VALIDACION.
El conjunto de PRUEBA se evalua una sola vez al final.

Uso por pasos (recomendado: cada paso se ejecuta por separado y lo
generado queda guardado en results_tf/):
  conda activate torchgpu
  python train_tf.py split                 # paso 1: inspeccion + split (~2 min)
  python train_tf.py train1                # paso 2: cabeza, base congelada (~1.5 h)
  python train_tf.py train2                # paso 3: fine-tuning (~2-3 h)
  python train_tf.py train2 --batch-size 16  # reintento con menos memoria
  python train_tf.py evaluate              # paso 4: elige mejor etapa y evalua test UNA vez
  python train_tf.py                       # todo de corrido (como antes)
  python train_tf.py split --check-only    # verifica pipeline sin entrenar

NOTA: en este equipo TensorFlow detecta solo CPU (sin GPU), el entrenamiento
completo es lento (horas). Cada paso guarda sus resultados y el siguiente
paso los reutiliza (no se repite trabajo).
"""

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
from pathlib import Path

# ---------------------------------------------------------------------
# SECCION DE CONFIGURACION (edita aqui)
# ---------------------------------------------------------------------
CONFIG = {
    "data_dir": "garbage_classification",
    "results_dir": "results_tf",
    "class_names_file": "class_names.json",
    "img_size": 299,          # InceptionV3 exige 299x299. NO cambiar.
    "batch_size": 16,         # Adecuado como punto de partida para GPU de 4 GB.
    "stage1_epochs": 10,      # cabeza clasificadora (base congelada)
    "stage2_epochs": 15,      # fine-tuning (capas finales descongeladas)
    "stage1_lr": 1e-3,        # LR cabeza. Rango util: 1e-4 a 1e-3.
    "stage2_lr": 1e-5,        # LR fine-tuning. Debe ser baja: 1e-5 a 1e-6.
    "fine_tune_layers": 40,   # ultimas N capas de la base a descongelar.
    "dropout": 0.3,           # regularizacion cabeza. 0.2-0.5 tipico.
    "train_ratio": 0.70,
    "val_ratio": 0.15,        # test = resto (~15%)
    "seed": 42,
    "patience_es": 5,         # early stopping (epocas sin mejora en val_loss)
    "patience_lr": 3,         # ReduceLROnPlateau (epocas sin mejora en val_loss)
}

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def parse_args():
    # Opciones comunes: valen ANTES o DESPUES del paso
    # (p. ej. "train_tf.py train2 --batch-size 16" y
    #  "train_tf.py --batch-size 16 train2" son equivalentes).
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--batch-size", type=int, default=None)
    common.add_argument("--epochs1", type=int, default=None,
                        help="epocas etapa 1")
    common.add_argument("--epochs2", type=int, default=None,
                        help="epocas etapa 2")
    common.add_argument("--data-dir", type=str, default=None)
    common.add_argument("--results-dir", type=str, default=None)
    common.add_argument("--seed", type=int, default=None)
    common.add_argument("--check-only", action="store_true",
                        help="solo verifica datos + pipeline + modelo, sin entrenar")
    common.add_argument("--re-split", action="store_true",
                        help="recalcula el split aunque exista splits.json")
    common.add_argument("--model", type=str, default=None,
                        help="modelo a evaluar (solo paso evaluate)")
    common.add_argument("--weights", type=str, default=None,
                        help="pesos etapa 1 para fine-tuning (solo paso train2)")
    common.add_argument("--fine-tune-layers", type=int, default=None,
                        help="ultimas N capas de la base a descongelar en etapa 2")
    common.add_argument("--dropout", type=float, default=None,
                        help="dropout de la cabeza clasificadora (etapas 1 y 2)")
    ap = argparse.ArgumentParser(
        description="Entrena InceptionV3 en garbage_classification, por pasos o completo.",
        parents=[common])
    sub = ap.add_subparsers(dest="paso", help="paso a ejecutar (default: full)")
    sub.add_parser("split", parents=[common],
                   help="paso 1: inspeccion + split (~2 min)")
    sub.add_parser("train1", parents=[common],
                   help="paso 2: cabeza con base congelada (~1.5 h)")
    sub.add_parser("train2", parents=[common],
                   help="paso 3: fine-tuning ultimas capas (~2-3 h)")
    sub.add_parser("evaluate", parents=[common],
                   help="paso 4: elige mejor etapa y evalua test UNA vez")
    sub.add_parser("full", parents=[common], help="todo de corrido")
    return ap.parse_args()


# ---------------------------------------------------------------------
# 1. Inspeccion + split estratificado con control de duplicados
# ---------------------------------------------------------------------
def md5_of(path, chunk=65536):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def scan_dataset(data_dir):
    """Devuelve {clase: [rutas]} con clases ordenadas. Solo .jpg/.jpeg."""
    data_dir = Path(data_dir)
    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    per_class = {}
    for c in classes:
        files = sorted([p for p in (data_dir / c).iterdir()
                        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg")])
        per_class[c] = files
    return classes, per_class


def stratified_split_no_leak(per_class, train_ratio, val_ratio, seed):
    """Split 70/15/15 estratificado.

    Los duplicados exactos (mismo md5) van JUNTOS a la misma particion
    para evitar fuga de datos. Devuelve dicts {clase: [rutas]} y reporte.
    """
    rng = random.Random(seed)
    # Hash MD5 de cada archivo UNA sola vez (evita releer los 15k ficheros).
    file_hash, file_class = {}, {}
    for c, files in per_class.items():
        for f in files:
            h = md5_of(f)
            file_hash[str(f)] = h
            file_class[str(f)] = c
    from collections import defaultdict
    by_hash = defaultdict(list)
    for fstr, h in file_hash.items():
        by_hash[h].append(fstr)
    dup_groups = [v for v in by_hash.values() if len(v) > 1]
    # Si un duplicado cruzara clases seria un error de etiquetado: avisar.
    cross = [v for v in dup_groups
             if len({file_class[f] for f in v}) > 1]
    if cross:
        print(f"AVISO: {len(cross)} grupos duplicados aparecen en >1 clase (posible "
              f"etiquetado cruzado). Ejemplo: {cross[0]}")

    train, val, test = {}, {}, {}
    for c, files in per_class.items():
        # unidades = un representante por hash (los dups van juntos)
        seen_h = {}
        for f in files:
            seen_h.setdefault(file_hash[str(f)], f)
        units = list(seen_h.values())
        rng.shuffle(units)
        n = len(units)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # garantiza >=1 en cada particion si la clase tiene >=3 imagenes
        if n >= 3:
            n_train = max(1, min(n_train, n - 2))
            n_val = max(1, min(n_val, n - n_train - 1))
        hash_split = {}
        for u in units[:n_train]:
            hash_split[file_hash[str(u)]] = "train"
        for u in units[n_train:n_train + n_val]:
            hash_split[file_hash[str(u)]] = "val"
        for u in units[n_train + n_val:]:
            hash_split[file_hash[str(u)]] = "test"
        dests = {"train": train, "val": val, "test": test}
        for f in files:
            dests[hash_split[file_hash[str(f)]]].setdefault(c, []).append(f)
    report = {
        "md5_unicos": len(by_hash),
        "grupos_duplicados": len(dup_groups),
        "archivos_extra_por_duplicado": sum(len(g) - 1 for g in dup_groups),
        "duplicados_multiclase": len(cross),
    }
    return train, val, test, report


def compute_class_weights(train_split, num_classes, class_to_idx):
    from collections import Counter
    cnt = Counter()
    for c, files in train_split.items():
        cnt[class_to_idx[c]] = len(files)
    total = sum(cnt.values())
    return {i: total / (num_classes * cnt[i]) for i in range(num_classes)}, \
           {c: cnt[class_to_idx[c]] for c in class_to_idx}


# ---------------------------------------------------------------------
# 2. Pipelines tf.data
# ---------------------------------------------------------------------
def make_dataset(file_label_list, class_to_idx, img_size, batch_size,
                 training, seed):
    import tensorflow as tf
    from tensorflow.keras.applications.inception_v3 import preprocess_input

    paths = [str(p) for p, _ in file_label_list]
    labels = [lbl for _, lbl in file_label_list]

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        aug = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal", seed=seed),
            tf.keras.layers.RandomRotation(0.08, seed=seed),
            tf.keras.layers.RandomZoom(0.15, seed=seed),
            tf.keras.layers.RandomContrast(0.15, seed=seed),
        ])

    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [img_size, img_size])
        return img, label

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(min(len(paths), 2048), seed=seed,
                        reshuffle_each_iteration=True)

        def _aug(img, label):
            return aug(img, training=True), label
        ds = ds.map(_aug, num_parallel_calls=tf.data.AUTOTUNE)

    def _prep(img, label):
        return preprocess_input(tf.cast(img, tf.float32)), label

    ds = ds.map(_prep, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------
# 3. Modelo
# ---------------------------------------------------------------------
def build_model(num_classes, dropout, fine_tune_layers=0, stage1_weights=None):
    import tensorflow as tf
    from tensorflow.keras.applications import InceptionV3

    base = InceptionV3(include_top=False, weights="imagenet",
                       input_shape=(299, 299, 3))
    base.trainable = False
    inputs = tf.keras.Input(shape=(299, 299, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    if fine_tune_layers > 0:
        if stage1_weights is not None:
            model.load_weights(stage1_weights)
        base.trainable = True
        # congela todo menos las ultimas N capas...
        for layer in base.layers[:-fine_tune_layers]:
            layer.trainable = False
        # ...y mantiene BatchNormalization SIEMPRE congeladas
        for layer in base.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
    return model, base


# ---------------------------------------------------------------------
# 4. Metricas con numpy (sin sklearn)
# ---------------------------------------------------------------------
def compute_metrics(y_true, y_pred, class_names):
    import numpy as np
    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
    per_class = []
    f1s = []
    for i, c in enumerate(class_names):
        tp = cm[i, i]
        prec = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        rec = tp / cm[i, :].sum() if cm[i, :].sum() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        f1s.append(f1)
        per_class.append({"clase": c, "precision": round(float(prec), 4),
                          "recall": round(float(rec), 4),
                          "f1": round(float(f1), 4),
                          "soporte": int(cm[i, :].sum())})
    return {"accuracy": round(acc, 4), "macro_f1": round(float(np.mean(f1s)), 4),
            "por_clase": per_class, "confusion_matrix": cm.tolist()}


def save_history_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def history_to_rows(hist):
    keys = list(hist.history.keys())
    return [dict(epoch=i + 1, **{k: round(float(v[i]), 4) for k, v in hist.history.items()})
            for i in range(len(hist.history[keys[0]]))]


def plot_confusion_matrix(cm, class_names, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib no disponible: se omite confusion_matrix.png "
              "(ver confusion_matrix.csv).")
        return
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicha")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusion (test)")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------
# Pasos ejecutables por separado (cada uno guarda en results_tf/ y el
# siguiente paso reutiliza lo guardado: no se repite trabajo)
# ---------------------------------------------------------------------
def init_tf(cfg):
    import tensorflow as tf
    print("TensorFlow:", tf.__version__)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU con memory growth:", [g.name for g in gpus])
        except RuntimeError as e:
            print("Error configurando GPU:", e)
    print("GPU detectadas:", [g.name for g in gpus] or "ninguna -> se usa CPU")
    random.seed(cfg["seed"])
    tf.random.set_seed(cfg["seed"])
    return tf


def make_context(classes, train_sp, val_sp, test_sp):
    class_to_idx = {c: i for i, c in enumerate(classes)}
    class_weights, _ = compute_class_weights(train_sp, len(classes),
                                             class_to_idx)
    print("Pesos de clase (solo train):",
          {c: round(class_weights[class_to_idx[c]], 3) for c in classes})
    return {"classes": classes, "train_sp": train_sp, "val_sp": val_sp,
            "test_sp": test_sp, "class_to_idx": class_to_idx,
            "class_weights": class_weights}


def paso_split(cfg):
    """Paso 1 (~2 min): inspeccion + split + jsons. No entrena nada."""
    classes, per_class = scan_dataset(cfg["data_dir"])
    print(f"Clases detectadas ({len(classes)}): {classes}")
    total = sum(len(v) for v in per_class.values())
    for c in classes:
        print(f"  {c}: {len(per_class[c])}")
    print(f"  TOTAL: {total}")

    train_sp, val_sp, test_sp, dup_report = stratified_split_no_leak(
        per_class, cfg["train_ratio"], cfg["val_ratio"], cfg["seed"])
    print(f"Duplicados exactos: {dup_report['grupos_duplicados']} grupos, "
          f"{dup_report['archivos_extra_por_duplicado']} archivos extra "
          f"(misma particion, sin fuga).")
    for name, sp in [("train", train_sp), ("val", val_sp), ("test", test_sp)]:
        n = sum(len(v) for v in sp.values())
        print(f"  {name}: {n} ({n / total:.1%})")

    results = Path(cfg["results_dir"])
    results.mkdir(parents=True, exist_ok=True)
    with open(cfg["class_names_file"], "w") as f:
        json.dump(classes, f, indent=2)
    shutil.copy(cfg["class_names_file"], results / "class_names.json")
    with open(results / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    with open(results / "duplicates_report.json", "w") as f:
        json.dump(dup_report, f, indent=2)
    with open(results / "splits.json", "w") as f:
        json.dump({s: {c: [str(p) for p in sp.get(c, [])]
                       for c in classes}
                   for s, sp in [("train", train_sp), ("val", val_sp),
                                 ("test", test_sp)]}, f)
    print(f"Split guardado en {results}/splits.json")
    return make_context(classes, train_sp, val_sp, test_sp)


def load_context(cfg):
    """Reutiliza splits.json + class_names.json sin re-escanear MD5."""
    results = Path(cfg["results_dir"])
    if not (results / "splits.json").exists():
        raise FileNotFoundError(
            "No existe results_tf/splits.json. Ejecuta primero: "
            "python train_tf.py split")
    with open(results / "class_names.json") as f:
        classes = json.load(f)
    with open(results / "splits.json") as f:
        raw = json.load(f)
    conv = {s: {c: [Path(p) for p in raw[s].get(c, [])] for c in classes}
            for s in ("train", "val", "test")}
    print("Splits reutilizados de splits.json "
          f"(train={sum(len(v) for v in conv['train'].values())}, "
          f"val={sum(len(v) for v in conv['val'].values())}, "
          f"test={sum(len(v) for v in conv['test'].values())}).")
    return make_context(classes, conv["train"], conv["val"], conv["test"])


def get_context(cfg, args):
    results = Path(cfg["results_dir"])
    if (results / "splits.json").exists() and not args.re_split:
        return load_context(cfg)
    return paso_split(cfg)


def flat_list(sp, classes, class_to_idx):
    return [(p, class_to_idx[c]) for c in classes for p in sp.get(c, [])]


def build_datasets(cfg, ctx, need=("train", "val", "test")):
    out = {}
    if "train" in need:
        out["train"] = make_dataset(
            flat_list(ctx["train_sp"], ctx["classes"], ctx["class_to_idx"]),
            ctx["class_to_idx"], cfg["img_size"], cfg["batch_size"],
            True, cfg["seed"])
    if "val" in need:
        out["val"] = make_dataset(
            flat_list(ctx["val_sp"], ctx["classes"], ctx["class_to_idx"]),
            ctx["class_to_idx"], cfg["img_size"], cfg["batch_size"],
            False, cfg["seed"])
    if "test" in need:
        out["test"] = make_dataset(
            flat_list(ctx["test_sp"], ctx["classes"], ctx["class_to_idx"]),
            ctx["class_to_idx"], cfg["img_size"], cfg["batch_size"],
            False, cfg["seed"])
    return out


def check_pipeline(cfg, ctx, train_ds):
    model, _ = build_model(len(ctx["classes"]), cfg["dropout"])
    for xb, yb in train_ds.take(1):
        preds = model(xb, training=False)
        print(f"CHECK OK: batch {tuple(xb.shape)} -> logits {tuple(preds.shape)}; "
              f"rango softmax suma={float(preds[0].numpy().sum()):.3f}")
    print("Verificacion superada.")


def paso_train1(cfg, ctx, tf, train_ds, val_ds):
    """Paso 2 (~1.5 h en CPU): cabeza con base congelada."""
    results = Path(cfg["results_dir"])
    classes = ctx["classes"]
    cbs1 = [
        tf.keras.callbacks.ModelCheckpoint(
            str(results / "best_stage1.keras"), monitor="val_accuracy",
            save_best_only=True, mode="max"),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=cfg["patience_lr"]),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=cfg["patience_es"],
            restore_best_weights=True),
    ]
    model, _ = build_model(len(classes), cfg["dropout"])
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg["stage1_lr"]),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    print("\n=== ETAPA 1: base congelada ===")
    h1 = model.fit(train_ds, validation_data=val_ds,
                   epochs=cfg["stage1_epochs"],
                   class_weight=ctx["class_weights"], callbacks=cbs1)
    save_history_csv(history_to_rows(h1), results / "historial_etapa1.csv")
    m1 = compute_metrics(*gather_truth(model, val_ds), classes)
    with open(results / "val_etapa1.json", "w") as f:
        json.dump(m1, f, indent=2)
    print(f"Etapa 1 val: acc={m1['accuracy']} macroF1={m1['macro_f1']} "
          f"(guardado best_stage1.keras + historial_etapa1.csv)")
    return model, m1


def paso_train2(cfg, ctx, tf, train_ds, val_ds, weights_path):
    """Paso 3 (~2-3 h en CPU): fine-tuning. Requiere best_stage1.keras."""
    results = Path(cfg["results_dir"])
    classes = ctx["classes"]
    model2, _ = build_model(len(classes), cfg["dropout"],
                            fine_tune_layers=cfg["fine_tune_layers"],
                            stage1_weights=weights_path)
    model2.compile(optimizer=tf.keras.optimizers.Adam(cfg["stage2_lr"]),
                   loss="sparse_categorical_crossentropy",
                   metrics=["accuracy"])
    n_trainable = sum(int(tf.size(w)) for w in model2.trainable_weights)
    print(f"\n=== ETAPA 2: fine-tuning ultimas {cfg['fine_tune_layers']} capas "
          f"(BN congeladas, params entrenables={n_trainable}) ===")
    cbs2 = [
        tf.keras.callbacks.ModelCheckpoint(
            str(results / "best_stage2.keras"), monitor="val_accuracy",
            save_best_only=True, mode="max"),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=cfg["patience_lr"]),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=cfg["patience_es"],
            restore_best_weights=True),
    ]
    h2 = model2.fit(train_ds, validation_data=val_ds,
                    epochs=cfg["stage2_epochs"],
                    class_weight=ctx["class_weights"], callbacks=cbs2)
    save_history_csv(history_to_rows(h2), results / "historial_etapa2.csv")
    m2 = compute_metrics(*gather_truth(model2, val_ds), classes)
    with open(results / "val_etapa2.json", "w") as f:
        json.dump(m2, f, indent=2)
    print(f"Etapa 2 val: acc={m2['accuracy']} macroF1={m2['macro_f1']} "
          f"(guardado best_stage2.keras + historial_etapa2.csv)")
    return model2, m2


def load_val_metrics(results, val_ds, classes, tf):
    """Lee val_etapaN.json; si falta pero existe el .keras, lo recalcula."""
    out = {}
    for n in (1, 2):
        pj, kj = results / f"val_etapa{n}.json", results / f"best_stage{n}.keras"
        if pj.exists():
            with open(pj) as f:
                out[n] = json.load(f)
        elif kj.exists():
            print(f"val_etapa{n}.json no existe: recalculando metricas de "
                  f"best_stage{n}.keras en val...")
            m = tf.keras.models.load_model(str(kj))
            out[n] = compute_metrics(*gather_truth(m, val_ds), classes)
            with open(pj, "w") as f:
                json.dump(out[n], f, indent=2)
    return out


def paso_evaluate(cfg, ctx, tf, model_path=None):
    """Paso 4: elige mejor etapa por F1 macro val y evalua test UNA vez."""
    results = Path(cfg["results_dir"])
    classes = ctx["classes"]
    ds = build_datasets(cfg, ctx, need=("val", "test"))

    if model_path:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"No existe el modelo: {model_path}")
        winner = tf.keras.models.load_model(model_path)
        wname = Path(model_path).stem
        src_path = Path(model_path)
        vm = load_val_metrics(results, ds["val"], classes, tf)
        m1, m2 = vm.get(1), vm.get(2)
    else:
        vm = load_val_metrics(results, ds["val"], classes, tf)
        m1, m2 = vm.get(1), vm.get(2)
        if m1 is None and m2 is None:
            raise FileNotFoundError(
                "Sin metricas de validacion. Ejecuta train1 y/o train2 primero.")
        if m2 is not None and (m1 is None or m2["macro_f1"] >= m1["macro_f1"]):
            cand = results / "best_stage2.keras"
            if not cand.exists():
                raise FileNotFoundError(f"No existe {cand}. Ejecuta train2.")
            winner, wname, src_path = tf.keras.models.load_model(str(cand)), "etapa2", cand
        else:
            cand = results / "best_stage1.keras"
            if not cand.exists():
                raise FileNotFoundError(f"No existe {cand}. Ejecuta train1.")
            winner, wname, src_path = tf.keras.models.load_model(str(cand)), "etapa1", cand
        print(f"Modelo elegido por F1 macro val: {wname}")

    dest = results / "best_model.keras"
    if src_path.resolve() != dest.resolve():
        shutil.copy(src_path, dest)

    yt, yp = gather_truth(winner, ds["test"])
    mt = compute_metrics(yt, yp, classes)
    with open(results / "metricas_test.json", "w") as f:
        json.dump({"modelo_elegido": wname,
                   "val_etapa1": m1, "val_etapa2": m2, "test": mt},
                  f, indent=2)
    with open(results / "reporte_por_clase.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clase", "precision", "recall",
                                          "f1", "soporte"])
        w.writeheader()
        w.writerows(mt["por_clase"])
    with open(results / "confusion_matrix.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["real/predicha"] + classes)
        for c, row in zip(classes, mt["confusion_matrix"]):
            w.writerow([c] + row)
    plot_confusion_matrix(mt["confusion_matrix"], classes,
                          results / "confusion_matrix.png")
    print(f"TEST (una sola vez): acc={mt['accuracy']} macroF1={mt['macro_f1']}")
    print(f"Resultados en {results}/: best_model.keras, metricas_test.json, "
          f"reporte_por_clase.csv, confusion_matrix.csv/.png")


def main():
    args = parse_args()
    cfg = dict(CONFIG)
    if args.batch_size: cfg["batch_size"] = args.batch_size
    if args.epochs1: cfg["stage1_epochs"] = args.epochs1
    if args.epochs2: cfg["stage2_epochs"] = args.epochs2
    if args.data_dir: cfg["data_dir"] = args.data_dir
    if args.results_dir: cfg["results_dir"] = args.results_dir
    if args.seed is not None: cfg["seed"] = args.seed
    if args.fine_tune_layers is not None:
        cfg["fine_tune_layers"] = args.fine_tune_layers
    if args.dropout is not None: cfg["dropout"] = args.dropout
    paso = args.paso or "full"

    if paso == "split":
        ctx = get_context(cfg, args)
        if args.check_only:
            tf = init_tf(cfg)
            ds = build_datasets(cfg, ctx, need=("train",))
            check_pipeline(cfg, ctx, ds["train"])
            print("Entrenamiento completo PENDIENTE "
                  "(sigue con: python train_tf.py train1).")
        return

    tf = init_tf(cfg)

    if paso == "full":
        ctx = paso_split(cfg)
        ds = build_datasets(cfg, ctx)
        if args.check_only:
            check_pipeline(cfg, ctx, ds["train"])
            print("Entrenamiento completo PENDIENTE (ejecuta sin --check-only).")
            return
        _, m1 = paso_train1(cfg, ctx, tf, ds["train"], ds["val"])
        _, m2 = paso_train2(cfg, ctx, tf, ds["train"], ds["val"],
                            str(Path(cfg["results_dir"]) / "best_stage1.keras"))
        print(f"\nResumen val: etapa1 F1={m1['macro_f1']} | etapa2 F1={m2['macro_f1']}")
        paso_evaluate(cfg, ctx, tf)
        return

    if paso == "train1":
        ctx = get_context(cfg, args)
        ds = build_datasets(cfg, ctx, need=("train", "val"))
        paso_train1(cfg, ctx, tf, ds["train"], ds["val"])
        print("Siguiente paso: python train_tf.py train2")
        return

    if paso == "train2":
        ctx = get_context(cfg, args)
        ds = build_datasets(cfg, ctx, need=("train", "val"))
        w = args.weights or str(Path(cfg["results_dir"]) / "best_stage1.keras")
        if not Path(w).exists():
            raise FileNotFoundError(
                f"No existe {w}. Ejecuta primero: python train_tf.py train1")
        paso_train2(cfg, ctx, tf, ds["train"], ds["val"], w)
        print("Siguiente paso: python train_tf.py evaluate")
        return

    if paso == "evaluate":
        ctx = get_context(cfg, args)
        paso_evaluate(cfg, ctx, tf, model_path=args.model)
        return


def gather_truth(model, ds):
    import numpy as np
    yt, yp = [], []
    for xb, yb in ds:
        p = model.predict(xb, verbose=0).argmax(1)
        yt += [int(v) for v in yb.numpy()]
        yp += [int(v) for v in p]
    return np.array(yt), np.array(yp)


if __name__ == "__main__":
    main()
