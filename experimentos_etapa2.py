r"""
Experimentos de Etapa 2 (fine-tuning) para buscar la mejor configuración.

Comienza desde results_tf/best_stage1.keras y prueba varias combinaciones
en VALIDACION con pocas epocas (cribada rapida). Luego se recomienda
entrenar a fondo la ganadora con 15 epocas y evaluar en test.

Uso (PowerShell, entorno ia2):
  $py = "C:\Users\dimas\miniconda3\envs\ia2\python.exe"
  & $py experimentos_etapa2.py
"""
import json
from pathlib import Path
from types import SimpleNamespace

import train_tf

BASE = Path(__file__).resolve().parent
PESOS_ETAPA1 = BASE / "results_tf" / "best_stage1.keras"

# Cada variante es un experimento independiente de etapa 2.
# 'stage2_epochs' pequeno (5) para cribada rapida en CPU.
VARIANTES = [
    {"nombre": "exp_a_cap20_lr1e-5",  "fine_tune_layers": 20,  "stage2_lr": 1e-5,  "stage2_epochs": 5, "batch_size": 16, "dropout": 0.3},
    {"nombre": "exp_b_cap40_lr1e-5",  "fine_tune_layers": 40,  "stage2_lr": 1e-5,  "stage2_epochs": 5, "batch_size": 16, "dropout": 0.3},
    {"nombre": "exp_c_cap60_lr1e-5",  "fine_tune_layers": 60,  "stage2_lr": 1e-5,  "stage2_epochs": 5, "batch_size": 16, "dropout": 0.3},
    {"nombre": "exp_d_cap40_lr5e-6",  "fine_tune_layers": 40,  "stage2_lr": 5e-6,  "stage2_epochs": 5, "batch_size": 16, "dropout": 0.3},
    {"nombre": "exp_e_cap40_lr1e-6",  "fine_tune_layers": 40,  "stage2_lr": 1e-6,  "stage2_epochs": 5, "batch_size": 16, "dropout": 0.3},
    {"nombre": "exp_f_cap80_lr1e-5",  "fine_tune_layers": 80,  "stage2_lr": 1e-5,  "stage2_epochs": 5, "batch_size": 16, "dropout": 0.3},
    {"nombre": "exp_g_cap40_lr1e-5_do0.2", "fine_tune_layers": 40, "stage2_lr": 1e-5, "stage2_epochs": 5, "batch_size": 16, "dropout": 0.2},
    {"nombre": "exp_h_cap40_lr1e-5_do0.5", "fine_tune_layers": 40, "stage2_lr": 1e-5, "stage2_epochs": 5, "batch_size": 16, "dropout": 0.5},
]


def main():
    cfg_base = dict(train_tf.CONFIG)
    tf = train_tf.init_tf(cfg_base)
    args = SimpleNamespace(re_split=False)
    ctx = train_tf.get_context(cfg_base, args)

    resultados = []

    for v in VARIANTES:
        print(f"\n{'='*70}\nVARIANTE: {v['nombre']}\n{'='*70}")
        cfg = dict(cfg_base)
        cfg["results_dir"] = f"results_tf/{v['nombre']}"
        cfg["fine_tune_layers"] = v["fine_tune_layers"]
        cfg["stage2_lr"] = v["stage2_lr"]
        cfg["stage2_epochs"] = v["stage2_epochs"]
        cfg["batch_size"] = v["batch_size"]
        cfg["dropout"] = v["dropout"]
        cfg["seed"] = cfg_base["seed"]

        Path(cfg["results_dir"]).mkdir(parents=True, exist_ok=True)

        ds = train_tf.build_datasets(cfg, ctx, need=("train", "val"))
        try:
            modelo, m2 = train_tf.paso_train2(
                cfg, ctx, tf, ds["train"], ds["val"],
                weights_path=str(PESOS_ETAPA1)
            )
            resumen = {
                "nombre": v["nombre"],
                "fine_tune_layers": v["fine_tune_layers"],
                "stage2_lr": v["stage2_lr"],
                "stage2_epochs": v["stage2_epochs"],
                "batch_size": v["batch_size"],
                "dropout": v["dropout"],
                "val_accuracy": m2["accuracy"],
                "val_macro_f1": m2["macro_f1"],
                "modelo": str(Path(cfg["results_dir"]) / "best_stage2.keras"),
            }
            resultados.append(resumen)
            print(f" >> {v['nombre']}: val_acc={m2['accuracy']:.4f}  val_macroF1={m2['macro_f1']:.4f}")
        except Exception as e:
            print(f" >> {v['nombre']}: ERROR -> {e}")

    print("\n\n===== TABLA COMPARATIVA (validacion, cribada rapida) =====")
    print(f"{'nombre':26s} {'capas':>5s} {'lr':>9s} {'ep':>3s} {'bs':>3s} {'do':>4s} {'val_acc':>9s} {'val_F1':>9s}")
    for r in sorted(resultados, key=lambda x: -x["val_macro_f1"]):
        print(f"{r['nombre']:26s} {r['fine_tune_layers']:>5d} {r['stage2_lr']:>9g} {r['stage2_epochs']:>3d} {r['batch_size']:>3d} {r['dropout']:>4.1f} {r['val_accuracy']:>9.4f} {r['val_macro_f1']:>9.4f}")

    if resultados:
        mejor = max(resultados, key=lambda x: x["val_macro_f1"])
        print(f"\nMejor en validacion: {mejor['nombre']}  (val_F1={mejor['val_macro_f1']:.4f})")
        print(f"\nSiguiente: entrenar esa variante a fondo (15 epocas) y evaluar en test:")
        print(f'  & $py train_tf.py train2 --results-dir results_tf/{mejor["nombre"]} --epochs2 15 --batch-size {mejor["batch_size"]} --weights results_tf/best_stage1.keras')
        print(f'  & $py train_tf.py evaluate --model results_tf/{mejor["nombre"]}/best_stage2.keras')


if __name__ == "__main__":
    main()
