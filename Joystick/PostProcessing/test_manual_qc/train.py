"""
train.py

Training loop for ROI QC CNN.

Uses:
- session-safe dataset split
- BCEWithLogitsLoss
- AdamW optimizer
- ROC-AUC, F1, precision, recall
- early stopping
- model checkpointing
"""

from __future__ import annotations

import os
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset.dataset import load_datasets
from model.cnn import ROICNN

from tqdm import tqdm
import time


# -------------------------------------------------------------------------
# Metrics helper
# -------------------------------------------------------------------------

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "cm": confusion_matrix(y_true, y_pred),
    }

    return metrics


# -------------------------------------------------------------------------
# Evaluation loop
# -------------------------------------------------------------------------

def evaluate(model, loader, device):
    print("in evaluate")

    model.eval()

    print("eval done")

    all_probs = []
    all_labels = []
    

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x).squeeze(1)

            print("logits stats:",
            logits.min().item(),
            logits.max().item(),
            logits.mean().item())
            
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    print("Probability range:")
    print("min :", all_probs.min())
    print("max :", all_probs.max())
    print("mean:", all_probs.mean())
    
    print(np.percentile(all_probs,[0, 5, 25, 50, 75, 95, 100]))

    return compute_metrics(all_labels, all_probs)


# -------------------------------------------------------------------------
# Training loop
# -------------------------------------------------------------------------

def train(
    data_dir = "./dataset/dataset_out",
    epochs=25,
    batch_size=8,
    lr=1e-3,
    patience=3,
    save_path="./checkpoints/best_model.pt",
):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------
    train_ds, val_ds = load_datasets(
        data_dir,
        val_ratio=0.2,
        augment=False,
    )

    x, y = train_ds[0]
    print("Single sample load time OK - take 3")
    print(x.shape)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # -------------------------------------------------------------
    # Model
    # -------------------------------------------------------------
    model = ROICNN(pretrained=True, freeze_backbone=True).to(device)

    x = torch.randn(8, 2, 128, 128).to(device)

    print("starting")
    model.eval()
    with torch.no_grad():
        t0 = time.time()
        for _ in range(2):
            model(x)
        print("Average forward:", (time.time() - t0) / 2)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(f"Trainable: {trainable:,}")
    print(f"Total:     {total:,}")

    # print(self.patches.shape)

    # -------------------------------------------------------------
    # Loss + optimizer
    # -------------------------------------------------------------
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.4]).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # -------------------------------------------------------------
    # Early stopping
    # -------------------------------------------------------------
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    checkpoint_path = "./checkpoints/last.pt"
    
    start_epoch = 1
    best_auc = -1
    patience_counter = 0
    
    if os.path.exists(checkpoint_path):
        print(f"Resuming from {checkpoint_path}")
    
        checkpoint = torch.load(checkpoint_path, map_location=device)
    
        model.load_state_dict(checkpoint["model_state"], strict=False)

        if "optimizer_state" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
            except ValueError as exc:
                print(f"Could not restore optimizer state: {exc}")
                print("Reinitializing optimizer from scratch.")
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        else:
            print("No optimizer state found; starting optimizer from scratch.")
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
        start_epoch = checkpoint["epoch"] + 1
        best_auc = checkpoint["best_auc"]
        patience_counter = checkpoint["patience_counter"]
    
        print(f"Starting at epoch {start_epoch}")
        print(f"Best ROC-AUC so far: {best_auc:.4f}")
    else:
        print("No checkpoint found. Starting fresh.")

    # -------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------
    for epoch in range(start_epoch, epochs + 1):

        print(f"\n🧠 Epoch {epoch}/{epochs}")
        model.train()
    
        epoch_loss = 0.0
        start_time = time.time()

    
        pbar = tqdm(train_loader, desc=f"Training", leave=False)
    
        for x, y in pbar:
            # print("x inside loop:", x.shape)
            # print("y inside loop:", y.shape)

            # print("batch start")
    
            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            # print("x after x.to(device):", x.shape)
            # print("y after y.to(device).unsqueeze(1):", y.shape)
    
            optimizer.zero_grad()

            # print("forward start")
            logits = model(x)

            # print("loss start")
            loss = criterion(logits, y)

            # print("backward start")
            loss.backward()

            # print("step start")
            optimizer.step()

            # print("done batch")
    
            epoch_loss += loss.item()
    
            pbar.set_postfix(loss=loss.item())
    
        metrics = evaluate(model, val_loader, device)
    
        epoch_time = time.time() - start_time
    
        print(f"\nEpoch {epoch}")
        print(f"Loss: {epoch_loss / len(train_loader):.4f}")
        print(f"Time: {epoch_time:.1f}s")
        print(f"Val Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Val Precision: {metrics['precision']:.4f}")
        print(f"Val Recall:    {metrics['recall']:.4f}")
        print(f"Val F1:        {metrics['f1']:.4f}")
        print(f"Val ROC-AUC:   {metrics['roc_auc']:.4f}")
        print("Confusion Matrix:")
        print(metrics["cm"])



        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_auc": best_auc,
            "patience_counter": patience_counter,
        }
        torch.save(checkpoint, "./checkpoints/last.pt")

        # ---------------------------------------------------------
        # Checkpointing
        # ---------------------------------------------------------
        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            patience_counter = 0

            torch.save(model.state_dict(), save_path)
            print(f"Saved best model → {save_path}")

        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{patience})")

        # ---------------------------------------------------------
        # Early stopping
        # ---------------------------------------------------------
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

def eval_only(
    data_dir="./dataset/dataset_out",
    model_path="./checkpoints/best_model.pt",
):
    print("start")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, val_ds = load_datasets(
        data_dir,
        val_ratio=0.2,
        augment=False,
    )
    print("loaded val_ds")

    val_loader = DataLoader(
        val_ds,
        batch_size=32,
        shuffle=False,
    )

    print("done val_loader")

    model = ROICNN(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    print("loaded model")

    metrics = evaluate(model, val_loader, device)

    print(metrics)

# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

if __name__ == "__main__":

    train()
    # eval_only()


