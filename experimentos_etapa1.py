r"""
Experimentos de Etapa 1: ¿cuantas epocas entrenar la cabeza (base congelada)?

Entrena la etapa 1 con distintas epocas y luego la etapa 2 (config por
defecto) encima de cada una, comparando F1 macro en VALIDACION. El caso
de 10 epocas ya existe en results_tf/ (best_stage1.keras + best_stage2.keras).

Uso (terminal Ubuntu, entorno tf-gpu312 activo):
  python experimentos_etapa1.py

Cada variante guarda en results_tf/<nombre>/:
  best_stage1.keras, historial_etapa1.csv, val_etapa1.json,
  best_stage2.keras, historial_etapa2.csv, val_etapa2.json
"""
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import train_tf

BASE = Path(__file__).resolve().parent
BASELINE_EP10 = BASE / "results_tf" / "val_etapa2.json"  # etapa1=10 epocas + etapa2(15)

# Epocas de ETAPA 1 a comparar (la etapa 2 se entrena igual, 15 epocas).
VARIANTES = [
    {"nombre": "exp_e1_ep15", "stage1_epochs": 15},
    {"nombre": "exp_e1_ep20", "stage1_epochs": 20},
]


def main():
    cfg_base = dict(train_tf.CONFIG)
    tf = train_tf.init_tf(cfg_base)
    args = SimpleNamespace(re_split=False)
    ctx = train_tf.get_context(cfg_base, args)

    resultados = []
    for v in VARIANTES:
        print(f"\n{'='*70}\nVARIANTE: {v['nombre']} "
              f"(etapa1={v['stage1_epochs']} epocas)\n{'='*70}")
        cfg = dict(cfg_base)
        cfg["results_dir"] = f"results_tf/{v['nombre']}"
        cfg["stage1_epochs"] = v["stage1_epochs"]
        cfg["stage2_epochs"] = 15
        Path(cfg["results_dir"]).mkdir(parents=True, exist_ok=True)

        ds = train_tf.build_datasets(cfg, ctx, need=("train", "val"))
        try:
            _, m1 = train_tf.paso_train1(cfg, ctx, tf, ds["train"], ds["val"])
            _, m2 = train_tf.paso_train2(
                cfg, ctx, tf, ds["train"], ds["val"],
                str(Path(cfg["results_dir"]) / "best_stage1.keras"))
            resultados.append({
                "nombre": v["nombre"],
                "stage1_epochs": v["stage1_epochs"],
                "val1_acc": m1["accuracy"],
                "val1_f1": m1["macro_f1"],
                "val2_acc": m2["accuracy"],
                "val2_f1": m2["macro_f1"],
            })
            print(f" >> {v['nombre']}: etapa1 F1={m1['macro_f1']:.4f} | "
                  f"etapa2 F1={m2['macro_f1']:.4f}")
        except Exception as e:
            print(f" >> {v['nombre']}: ERROR -> {e}")
        finally:
            # Liberar memoria de la GPU entre variantes (evita OOM con 4 GB).
            tf.keras.backend.clear_session()
            gc.collect()

    # Referencia: el caso ya existente (etapa1=10 epocas)
    ref = None
    if BASELINE_EP10.exists():
        with open(BASELINE_EP10) as f:
            vm = json.load(f)
        ref = {"nombre": "base_ep10 (ya existente)", "stage1_epochs": 10,
               "val1_acc": None, "val1_f1": None,
               "val2_acc": vm["accuracy"], "val2_f1": vm["macro_f1"]}

    print("\n\n===== COMPARATIVA ETAPA 1 (validacion) =====")
    print(f"{'nombre':26s} {'ep1':>4s} {'val1_acc':>9s} {'val1_F1':>8s} "
          f"{'val2_acc':>9s} {'val2_F1':>8s}")
    filas = ([ref] if ref else []) + resultados
    for r in filas:
        v1a = f"{r['val1_acc']:.4f}" if r['val1_acc'] is not None else "-"
        v1f = f"{r['val1_f1']:.4f}" if r['val1_f1'] is not None else "-"
        print(f"{r['nombre']:26s} {r['stage1_epochs']:>4d} {v1a:>9s} {v1f:>8s} "
              f"{r['val2_acc']:>9.4f} {r['val2_f1']:>8.4f}")

    if filas:
        mejor = max(filas, key=lambda x: x["val2_f1"])
        print(f"\nMejor F1 de validacion (etapa 2): {mejor['nombre']} "
              f"(F1={mejor['val2_f1']:.4f})")
        print("\nSiguiente: evaluar ese modelo en test con:")
        print(f"  python train_tf.py evaluate --model results_tf/{mejor['nombre']}/best_stage2.keras")
        print("Nota: si el mejor es 'base_ep10', usa results_tf/best_stage2.keras")


if __name__ == "__main__":
    main()
