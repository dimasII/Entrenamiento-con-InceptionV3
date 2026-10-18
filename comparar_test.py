r"""
Evalua varios modelos .keras en el conjunto de TEST y compara resultados.

Uso (PowerShell, entorno ia2):
  $py = "C:\Users\dimas\miniconda3\envs\ia2\python.exe"
  & $py comparar_test.py `
    results_tf/best_stage2.keras `
    results_tf/exp_b_cap40_lr1e-5/best_stage2.keras `
    results_tf/exp_c_cap60_lr1e-5/best_stage2.keras
"""
import json
import sys
from pathlib import Path

import train_tf


def main():
    if len(sys.argv) < 2:
        print("Uso: python comparar_test.py modelo1.keras [modelo2.keras ...]")
        sys.exit(1)

    cfg = dict(train_tf.CONFIG)
    tf = train_tf.init_tf(cfg)
    args = type("A", (), {"re_split": False})()
    ctx = train_tf.get_context(cfg, args)
    ds = train_tf.build_datasets(cfg, ctx, need=("val", "test"))

    print(f"{'modelo':38s} {'test_acc':>9s} {'test_F1':>9s} {'val_acc':>9s} {'val_F1':>9s}")
    filas = []
    for modelo_path in sys.argv[1:]:
        p = Path(modelo_path)
        if not p.exists():
            print(f"NO EXISTE: {modelo_path}")
            continue
        modelo = tf.keras.models.load_model(str(p))
        yt, yp = train_tf.gather_truth(modelo, ds["test"])
        mt = train_tf.compute_metrics(yt, yp, ctx["classes"])

        vp = p.parent / "val_etapa2.json"
        if vp.exists():
            with open(vp) as f:
                vm = json.load(f)
        else:
            vm = {"accuracy": None, "macro_f1": None}

        filas.append((p.name, mt["accuracy"], mt["macro_f1"], vm["accuracy"], vm["macro_f1"]))

    for nombre, ta, tf1, va, vf in sorted(filas, key=lambda x: -(x[2] or 0)):
        va_s = f"{va:.4f}" if va is not None else "-"
        vf_s = f"{vf:.4f}" if vf is not None else "-"
        print(f"{nombre:38s} {ta:>9.4f} {tf1:>9.4f} {va_s:>9s} {vf_s:>9s}")


if __name__ == "__main__":
    main()
