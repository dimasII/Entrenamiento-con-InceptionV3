import argparse
import csv
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

# =====================================================================
# 1. CONFIGURACION PRINCIPAL  <-- EDITA AQUI PARA MEJORES RESULTADOS
# =====================================================================
CONFIG = {
    "data_dir": "garbage_classification",
    "checkpoint_dir": "checkpoints",
    "img_size": 299,          # InceptionV3 exige 299x299. NO cambiar.
    "epochs": 20,             # <-- NUMERO DE EPOCAS. Empieza con 20, sube a 30-50 si val_acc sigue mejorando.
    "batch_size": 32,         # 32 ok para RTX 3050 4GB. Si falta memoria (OOM): baja a 16.
    "lr": 3e-4,               # learning rate. Rango util: 1e-4 a 1e-3. Si overfitting rapido, baja a 1e-4.
    "weight_decay": 1e-4,     # regularizacion L2. Sube a 1e-3 si hay overfitting.
    "patience": 7,            # early stopping: epocas sin mejorar val_acc antes de parar. 5-10 tipico.
    "train_ratio": 0.70,      # 70% train
    "val_ratio": 0.15,        # 15% val
    # test = resto (15%)
    "seed": 42,
    "num_workers": 0,         # En Windows deja 0. En Linux puedes subir a 4.
    "freeze_backbone": True,  # True = entrena solo la ultima capa (rapido, recomendado inicio).
                              # False = fine-tuning completo (mejor acc final, mas lento).
    "unfreeze_after": 5,      # Si freeze_backbone=True, descongela todo despues de N epocas.
    "label_smoothing": 0.1,   # 0.0-0.2 ayuda con desbalance. 0 = desactivado.
}

CLASSES = ["battery", "biological", "brown-glass", "cardboard", "clothes",
           "green-glass", "metal", "paper", "plastic", "shoes", "trash", "white-glass"]


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(img_size):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


def stratified_split(dataset, train_ratio, val_ratio, seed):
    rng = random.Random(seed)
    targets = [s[1] for s in dataset.samples]
    by_class = defaultdict(list)
    for idx, t in enumerate(targets):
        by_class[t].append(idx)
    train_idx, val_idx, test_idx = [], [], []
    for c, idxs in by_class.items():
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_idx += idxs[:n_train]
        val_idx += idxs[n_train:n_train + n_val]
        test_idx += idxs[n_train + n_val:]
    rng.shuffle(train_idx); rng.shuffle(val_idx); rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def build_model(num_classes, freeze_backbone):
    model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    model.aux_logits = True
    # Clasificador principal + auxiliar adaptados a 12 clases
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
    if freeze_backbone:
        for name, p in model.named_parameters():
            if "fc" not in name:
                p.requires_grad = False
    return model


def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True


def accuracy(outputs, targets):
    return (outputs.argmax(1) == targets).float().mean().item()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    tot_loss, tot_acc, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out, aux = model(x)  # Inception devuelve (principal, auxiliar) en train
        loss = criterion(out, y) + 0.4 * criterion(aux, y)
        loss.backward()
        optimizer.step()
        bs = y.size(0)
        tot_loss += loss.item() * bs
        tot_acc += accuracy(out.detach(), y) * bs
        n += bs
    return tot_loss / n, tot_acc / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    tot_loss, tot_acc, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)  # en eval solo devuelve principal
        loss = criterion(out, y)
        bs = y.size(0)
        tot_loss += loss.item() * bs
        tot_acc += accuracy(out, y) * bs
        n += bs
    return tot_loss / n, tot_acc / n


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_acc, history, cfg):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "best_val_acc": best_val_acc,
        "history": history,
        "config": cfg,
    }, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--resume", type=str, default=None, help="ruta a checkpoints/last.pth para continuar")
    ap.add_argument("--no-freeze", action="store_true", help="fine-tuning completo desde epoca 0")
    args = ap.parse_args()

    cfg = dict(CONFIG)
    if args.epochs: cfg["epochs"] = args.epochs
    if args.batch_size: cfg["batch_size"] = args.batch_size
    if args.lr: cfg["lr"] = args.lr
    if args.no_freeze: cfg["freeze_backbone"] = False

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    ckpt_dir = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(exist_ok=True)
    # Guarda la config usada (no se pierde que hiperparametros dieron cada resultado)
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    train_tf, eval_tf = build_transforms(cfg["img_size"])
    base_ds = datasets.ImageFolder(cfg["data_dir"], transform=train_tf)
    print(f"Clases encontradas: {base_ds.classes}")
    assert base_ds.classes == CLASSES, f"Se esperaban {CLASSES}, hay {base_ds.classes}"

    train_idx, val_idx, test_idx = stratified_split(
        base_ds, cfg["train_ratio"], cfg["val_ratio"], cfg["seed"])
    print(f"Split -> train: {len(train_idx)}  val: {len(val_idx)}  test: {len(test_idx)}")

    # Dataset de val/test con transform SIN aumento
    eval_ds = datasets.ImageFolder(cfg["data_dir"], transform=eval_tf)
    train_ds = Subset(base_ds, train_idx)
    val_ds = Subset(eval_ds, val_idx)
    test_ds = Subset(eval_ds, test_idx)

    # Pesos por clase (clothes/shoes dominan -> se penalizan menos sus aciertos)
    from collections import Counter
    cnt = Counter([base_ds.samples[i][1] for i in train_idx])
    n_train = len(train_idx)
    weights = torch.tensor([n_train / (len(CLASSES) * cnt[c]) for c in range(len(CLASSES))],
                           dtype=torch.float)
    print("Conteo train por clase:", {CLASSES[k]: v for k, v in sorted(cnt.items())})

    g = torch.Generator().manual_seed(cfg["seed"])
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg["num_workers"], generator=g, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=cfg["num_workers"], pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False,
                             num_workers=cfg["num_workers"], pin_memory=True)

    model = build_model(len(CLASSES), cfg["freeze_backbone"]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device),
                                    label_smoothing=cfg["label_smoothing"])
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3)

    start_epoch, best_val_acc, history = 0, 0.0, []
    unfrozen = not cfg["freeze_backbone"]

    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        if ck.get("scheduler_state"):
            scheduler.load_state_dict(ck["scheduler_state"])
        start_epoch = ck["epoch"] + 1
        best_val_acc = ck["best_val_acc"]
        history = ck["history"]
        print(f"Reanudado desde {args.resume} en epoca {start_epoch}, best_val_acc={best_val_acc:.4f}")

    csv_path = ckpt_dir / "historial_entrenamiento.csv"
    write_header = not csv_path.exists() and start_epoch == 0

    patience_left = cfg["patience"]
    for epoch in range(start_epoch, cfg["epochs"]):
        # Descongelar backbone a mitad del entrenamiento (truco estandar transfer learning)
        if cfg["freeze_backbone"] and not unfrozen and epoch >= cfg["unfreeze_after"]:
            print(f"[Epoca {epoch+1}] Descongelando backbone para fine-tuning...")
            unfreeze_all(model)
            # LR menor para no destruir pesos preentrenados
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"] / 10,
                                          weight_decay=cfg["weight_decay"])
            unfrozen = True

        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(va_acc)
        dt = time.time() - t0

        row = {"epoch": epoch + 1, "train_loss": round(tr_loss, 4), "train_acc": round(tr_acc, 4),
               "val_loss": round(va_loss, 4), "val_acc": round(va_acc, 4),
               "lr": optimizer.param_groups[0]["lr"], "time_s": round(dt, 1)}
        history.append(row)
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader(); write_header = False
            w.writerow(row)

        print(f"Epoca {epoch+1}/{cfg['epochs']} [{dt:.0f}s] "
              f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | val loss={va_loss:.4f} acc={va_acc:.4f} "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        # === GUARDADO: nada se pierde ===
        save_checkpoint(ckpt_dir / f"checkpoint_epoch_{epoch+1}.pth",
                        model, optimizer, scheduler, epoch, best_val_acc, history, cfg)
        save_checkpoint(ckpt_dir / "last.pth",
                        model, optimizer, scheduler, epoch, best_val_acc, history, cfg)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            save_checkpoint(ckpt_dir / "best_model.pth",
                            model, optimizer, scheduler, epoch, best_val_acc, history, cfg)
            print(f"  -> Nuevo mejor modelo guardado (val_acc={va_acc:.4f})")
            patience_left = cfg["patience"]
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping en epoca {epoch+1}. Mejor val_acc={best_val_acc:.4f}")
                break

    # Evaluacion final en test con el mejor modelo
    if (ckpt_dir / "best_model.pth").exists():
        ck = torch.load(ckpt_dir / "best_model.pth", map_location=device)
        model.load_state_dict(ck["model_state"])
    te_loss, te_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTEST -> loss={te_loss:.4f} acc={te_acc:.4f}  (mejor val_acc={best_val_acc:.4f})")
    print(f"Archivos en {ckpt_dir}/ : best_model.pth, last.pth, checkpoint_epoch_N.pth, "
          f"historial_entrenamiento.csv, config.json")


if __name__ == "__main__":
    main()
