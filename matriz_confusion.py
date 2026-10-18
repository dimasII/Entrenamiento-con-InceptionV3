r"""
Genera results_tf/confusion_matrix.png a partir de las metricas ya calculadas
en results_tf/metricas_test.json (util si matplotlib no estaba instalado
cuando se ejecuto el paso evaluate).

Uso:
  python matriz_confusion.py                          # usa results_tf/metricas_test.json
  python matriz_confusion.py ruta/metricas_test.json  # otro archivo
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    json_path = (Path(sys.argv[1]) if len(sys.argv) > 1
                 else Path("results_tf/metricas_test.json"))
    classes_path = Path("results_tf/class_names.json")
    if not json_path.exists():
        raise SystemExit(f"No existe {json_path}")
    if not classes_path.exists():
        raise SystemExit(f"No existe {classes_path}")

    classes = json.loads(classes_path.read_text(encoding="utf-8"))
    data = json.loads(json_path.read_text(encoding="utf-8"))
    bloque = data.get("test", data)
    cm = np.array(bloque["confusion_matrix"])

    out_path = json_path.parent / "confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicha")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusion (test) - {bloque.get('accuracy', '?')} acc")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Guardado: {out_path}")
    print(f"accuracy={bloque.get('accuracy')}  macro_f1={bloque.get('macro_f1')}")


if __name__ == "__main__":
    main()
