"""
training/train.py
===================
Entrena la cabeza de clasificación de PC sobre las features REALES
extraídas con los modelos ONNX de IntellEvent (Opción A: extractor fijo).

Maneja el desbalance de clases verificado (65/297/246/531) con pesos de
clase inversos a la frecuencia, tal como recomienda la documentación de
Ferrari.
"""

import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import CFG
from data.dataset import load_cached_dataset, FerrariFeatureDataset, split_dataset
from models.pc_classifier import PCClassifierFromIntellEventFeatures


def get_device():
    if CFG.train.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_class_weights(y, num_classes):
    """Pesos inversos a la frecuencia, normalizados a media 1."""
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def compute_macro_f1(logits, y, num_classes):
    preds = logits.argmax(dim=1)
    f1_scores = []
    for c in range(num_classes):
        tp = ((preds == c) & (y == c)).sum().item()
        fp = ((preds == c) & (y != c)).sum().item()
        fn = ((preds != c) & (y == c)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
    return sum(f1_scores) / len(f1_scores), f1_scores


@torch.no_grad()
def confusion_matrix(logits, y, num_classes):
    preds = logits.argmax(dim=1)
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(y, preds):
        cm[t, p] += 1
    return cm


def run_epoch(model, loader, device, class_weights, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_correct, total_n = 0.0, 0, 0
    all_logits, all_y = [], []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            mask = batch["mask"].to(device)

            logits = model(x, mask)
            loss = F.cross_entropy(logits, y, weight=class_weights.to(device))

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss += loss.item() * x.shape[0]
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_n += x.shape[0]
            all_logits.append(logits.detach().cpu())
            all_y.append(y.cpu())

    all_logits = torch.cat(all_logits)
    all_y = torch.cat(all_y)
    macro_f1, per_class_f1 = compute_macro_f1(all_logits, all_y, CFG.data.num_pc_classes)

    return {
        "loss": total_loss / total_n,
        "acc": total_correct / total_n,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "logits": all_logits,
        "y": all_y,
    }


def train_model():
    device = get_device()
    print(f"Dispositivo: {device}")
    torch.manual_seed(CFG.train.seed)

    print("\n[1/4] Cargando features cacheadas (extraídas con ONNX reales de IntellEvent)...")
    X, y, mask, filepaths = load_cached_dataset(CFG.paths.features_cache_dir, CFG.data.max_seq_len_features)
    print(f"  X={X.shape}  y distribución={np.bincount(y)}")

    (X_tr, y_tr, m_tr), (X_val, y_val, m_val), (X_te, y_te, m_te) = split_dataset(
        X, y, mask, CFG.train.val_split, CFG.train.test_split, CFG.train.seed
    )
    print(f"  train={len(y_tr)}  val={len(y_val)}  test={len(y_te)}")
    print(f"  train distribución: {np.bincount(y_tr, minlength=4)}")

    train_loader = DataLoader(FerrariFeatureDataset(X_tr, y_tr, m_tr), batch_size=CFG.train.batch_size, shuffle=True)
    val_loader = DataLoader(FerrariFeatureDataset(X_val, y_val, m_val), batch_size=CFG.train.batch_size, shuffle=False)
    test_loader = DataLoader(FerrariFeatureDataset(X_te, y_te, m_te), batch_size=CFG.train.batch_size, shuffle=False)

    class_weights = compute_class_weights(y_tr, CFG.data.num_pc_classes)
    print(f"  pesos de clase (inversos a frecuencia): {class_weights.tolist()}")

    print("\n[2/4] Construyendo modelo...")
    model = PCClassifierFromIntellEventFeatures(CFG).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parámetros entrenables: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.train.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)

    print(f"\n[3/4] Entrenando ({CFG.train.num_epochs} épocas)...")
    best_val_f1 = 0.0
    os.makedirs(CFG.paths.checkpoints_dir, exist_ok=True)

    for epoch in range(1, CFG.train.num_epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, class_weights, optimizer)
        val_metrics = run_epoch(model, val_loader, device, class_weights, optimizer=None)
        scheduler.step(val_metrics["macro_f1"])

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), os.path.join(CFG.paths.checkpoints_dir, "best_pc_classifier.pt"))

        if epoch % 2 == 0 or epoch == 1:
            print(f"  [{epoch:>3}/{CFG.train.num_epochs}] "
                  f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.3f} | "
                  f"val_acc={val_metrics['acc']:.3f} val_macro_f1={val_metrics['macro_f1']:.3f}")

    print(f"\n[4/4] Evaluación final en test set (mejor checkpoint, val_macro_f1={best_val_f1:.3f})...")
    model.load_state_dict(torch.load(os.path.join(CFG.paths.checkpoints_dir, "best_pc_classifier.pt")))
    test_metrics = run_epoch(model, test_loader, device, class_weights, optimizer=None)

    print(f"  Test accuracy:   {test_metrics['acc']:.3f}")
    print(f"  Test macro-F1:   {test_metrics['macro_f1']:.3f}")
    print(f"  F1 por clase:")
    for i, name in enumerate(CFG.data.pc_class_names):
        print(f"    {name}: {test_metrics['per_class_f1'][i]:.3f}")

    cm = confusion_matrix(test_metrics["logits"], test_metrics["y"], CFG.data.num_pc_classes)
    print(f"\n  Matriz de confusión (filas=real, columnas=predicho):")
    print(f"  {'':>30}" + "".join(f"{i:>6}" for i in range(CFG.data.num_pc_classes)))
    for i, name in enumerate(CFG.data.pc_class_names):
        print(f"  {name:>30}" + "".join(f"{cm[i,j].item():>6}" for j in range(CFG.data.num_pc_classes)))

    return model, test_metrics


def train_batch_resumable(num_epochs_this_call=5):
    """
    Entrena por lotes de épocas, guardando el estado completo (modelo,
    optimizer, scheduler, mejor F1, época actual) para poder reanudar
    en la siguiente llamada sin perder progreso. Pensado para entornos
    lentos (CPU) donde una corrida completa no cabe en una sola ejecución.

    Uso:
        python3 training/train.py --mode batch --epochs 5
        (repetir el comando las veces necesarias hasta completar CFG.train.num_epochs)
    """
    device = get_device()
    torch.manual_seed(CFG.train.seed)

    state_path = os.path.join(CFG.paths.checkpoints_dir, "training_state.pt")
    os.makedirs(CFG.paths.checkpoints_dir, exist_ok=True)

    print("[1/3] Cargando datos...")
    X, y, mask, filepaths = load_cached_dataset(CFG.paths.features_cache_dir, CFG.data.max_seq_len_features)
    (X_tr, y_tr, m_tr), (X_val, y_val, m_val), (X_te, y_te, m_te) = split_dataset(
        X, y, mask, CFG.train.val_split, CFG.train.test_split, CFG.train.seed
    )
    train_loader = DataLoader(FerrariFeatureDataset(X_tr, y_tr, m_tr), batch_size=CFG.train.batch_size, shuffle=True)
    val_loader = DataLoader(FerrariFeatureDataset(X_val, y_val, m_val), batch_size=CFG.train.batch_size, shuffle=False)
    class_weights = compute_class_weights(y_tr, CFG.data.num_pc_classes)

    model = PCClassifierFromIntellEventFeatures(CFG).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.train.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)

    start_epoch = 1
    best_val_f1 = 0.0
    if os.path.exists(state_path):
        state = torch.load(state_path, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        best_val_f1 = state["best_val_f1"]
        print(f"  Reanudando desde época {start_epoch} (mejor val_macro_f1 hasta ahora: {best_val_f1:.3f})")
    else:
        print("  No hay checkpoint previo, empezando desde cero.")

    end_epoch = min(start_epoch + num_epochs_this_call - 1, CFG.train.num_epochs)

    print(f"\n[2/3] Entrenando épocas {start_epoch} a {end_epoch} (de {CFG.train.num_epochs} totales)...")
    for epoch in range(start_epoch, end_epoch + 1):
        t0 = time.time()
        train_metrics = run_epoch(model, train_loader, device, class_weights, optimizer)
        val_metrics = run_epoch(model, val_loader, device, class_weights, optimizer=None)
        scheduler.step(val_metrics["macro_f1"])

        is_best = val_metrics["macro_f1"] > best_val_f1
        if is_best:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), os.path.join(CFG.paths.checkpoints_dir, "best_pc_classifier.pt"))

        dt = time.time() - t0
        print(f"  [{epoch:>3}/{CFG.train.num_epochs}] ({dt:.0f}s) "
              f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.3f} | "
              f"val_acc={val_metrics['acc']:.3f} val_macro_f1={val_metrics['macro_f1']:.3f}"
              f"{'  <- mejor hasta ahora' if is_best else ''}")

    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": end_epoch,
        "best_val_f1": best_val_f1,
    }, state_path)

    print(f"\n[3/3] Lote completado. Progreso: época {end_epoch}/{CFG.train.num_epochs}.")
    if end_epoch < CFG.train.num_epochs:
        print("Quedan épocas por entrenar. Vuelve a correr este script (--mode batch) para continuar.")
    else:
        print("Entrenamiento completo. Corre con --mode evaluate para ver resultados en test set.")

    return end_epoch, CFG.train.num_epochs


def evaluate_best_model():
    """Carga el mejor checkpoint guardado y evalúa en el test set."""
    device = get_device()
    X, y, mask, filepaths = load_cached_dataset(CFG.paths.features_cache_dir, CFG.data.max_seq_len_features)
    (X_tr, y_tr, m_tr), (X_val, y_val, m_val), (X_te, y_te, m_te) = split_dataset(
        X, y, mask, CFG.train.val_split, CFG.train.test_split, CFG.train.seed
    )
    test_loader = DataLoader(FerrariFeatureDataset(X_te, y_te, m_te), batch_size=CFG.train.batch_size, shuffle=False)
    class_weights = compute_class_weights(y_tr, CFG.data.num_pc_classes)

    model = PCClassifierFromIntellEventFeatures(CFG).to(device)
    ckpt_path = os.path.join(CFG.paths.checkpoints_dir, "best_pc_classifier.pt")
    if not os.path.exists(ckpt_path):
        print(f"No se encontró checkpoint en {ckpt_path}. Entrena primero con --mode batch.")
        return

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_metrics = run_epoch(model, test_loader, device, class_weights, optimizer=None)

    print(f"Test accuracy: {test_metrics['acc']:.3f}")
    print(f"Test macro-F1: {test_metrics['macro_f1']:.3f}")
    print("F1 por clase:")
    for i, name in enumerate(CFG.data.pc_class_names):
        print(f"  {name}: {test_metrics['per_class_f1'][i]:.3f}")

    cm = confusion_matrix(test_metrics["logits"], test_metrics["y"], CFG.data.num_pc_classes)
    print("\nMatriz de confusión (filas=real, columnas=predicho):")
    print(f"{'':>32}" + "".join(f"{i:>6}" for i in range(CFG.data.num_pc_classes)))
    for i, name in enumerate(CFG.data.pc_class_names):
        print(f"{name:>32}" + "".join(f"{cm[i,j].item():>6}" for j in range(CFG.data.num_pc_classes)))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "batch", "evaluate"], default="batch")
    parser.add_argument("--epochs", type=int, default=5, help="Épocas a entrenar en esta llamada (modo batch)")
    args = parser.parse_args()

    if args.mode == "full":
        train_model()
    elif args.mode == "batch":
        train_batch_resumable(num_epochs_this_call=args.epochs)
    else:
        evaluate_best_model()
