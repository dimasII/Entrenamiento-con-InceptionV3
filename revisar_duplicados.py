r"""
Busca duplicados exactos (mismo MD5) que aparecen en MAS de una clase
(posible etiquetado cruzado) dentro de garbage_classification/.

Recalcula los MD5 del dataset (tarda ~1-2 min) y lista los grupos.

Uso:
  python revisar_duplicados.py
"""
import hashlib
from collections import defaultdict
from pathlib import Path

DATA = Path("garbage_classification")


def md5_of(path, chunk=1 << 16):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    by_hash = defaultdict(list)
    total = 0
    for cls_dir in sorted(p for p in DATA.iterdir() if p.is_dir()):
        for f in sorted(cls_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg"):
                by_hash[md5_of(f)].append(f)
                total += 1

    print(f"Archivos escaneados: {total}")
    grupos = [v for v in by_hash.values() if len(v) > 1]
    multiclase = [g for g in grupos if len({p.parent.name for p in g}) > 1]
    print(f"Grupos duplicados exactos: {len(grupos)}")
    print(f"Grupos que cruzan clases (posible etiquetado cruzado): {len(multiclase)}\n")
    for g in multiclase:
        clases = sorted({p.parent.name for p in g})
        print(f"- clases implicadas: {clases}")
        for p in g:
            print(f"    {p.parent.name}/{p.name}")


if __name__ == "__main__":
    main()
