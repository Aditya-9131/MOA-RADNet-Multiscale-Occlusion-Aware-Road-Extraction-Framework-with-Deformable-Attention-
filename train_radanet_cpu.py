"""
RADANet — CPU Optimized Training Pipeline
=========================================
• Image Size: 256x256 (Reduced for CPU efficiency)
• Backbone: Frozen ResNet-50 (Only RAM + DAM + Decoder trained)
• Batch Size: 1 + Gradient Accumulation (4 steps)
• Epochs: 60-80, Early stopping patience = 10
• Optimization: AdamW + CosineAnnealingLR
• Loss: 0.3 BCE + 0.5 Dice + 0.2 Focal
• Target Device: CPU
"""

import os
import sys
import gc
import time
import argparse

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── project imports ──────────────────────────────────────────────────
from models.model import RADANet          # models/ package (DO NOT modify)
from OARENet.dataset import RoadDataset   # OARENet dataset  (DO NOT modify)
from utils.losses import ImprovedRADALoss
from utils.plots import plot_training_results, plot_pr_curve


# =====================================================================
# Metric helpers
# =====================================================================
def calculate_metrics(pred, target, threshold=0.5):
    """Pixel-level TP / FP / FN → Precision, Recall, F1, IoU."""
    pred_bin = (pred > threshold).float()
    target_bin = target.float()

    tp = (pred_bin * target_bin).sum().item()
    fp = (pred_bin * (1.0 - target_bin)).sum().item()
    fn = ((1.0 - pred_bin) * target_bin).sum().item()

    precision = tp / (tp + fp + 1e-7)
    recall    = tp / (tp + fn + 1e-7)
    f1        = 2.0 * precision * recall / (precision + recall + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)

    return precision, recall, f1, iou


# =====================================================================
# Backbone freeze / unfreeze
# =====================================================================
def set_backbone_trainable(model, trainable=True):
    """Freeze or unfreeze the ResNet-50 encoder layers."""
    # Layers: enc0, enc1, enc2, enc3, enc4
    backbone = [model.enc0, model.enc1, model.enc2, model.enc3, model.enc4]
    for layer in backbone:
        for p in layer.parameters():
            p.requires_grad = trainable
    tag = "Unfrozen" if trainable else "Frozen"
    print(f"  [Backbone] {tag}")


# =====================================================================
# Main training loop
# =====================================================================
def train(args):
    # ── device ───────────────────────────────────────────────────────
    device = torch.device("cpu") # Forced to CPU as per requirements
    print("=" * 60)
    print(f"  RADANet CPU-Optimized Training — device={device}")
    print(f"  size={args.image_size}  batch={args.batch_size}  "
          f"accum={args.accum_steps}  epochs={args.epochs}")
    print("=" * 60)

    # ── directories ──────────────────────────────────────────────────
    os.makedirs("graphs", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # ── model ────────────────────────────────────────────────────────
    # CRITICAL: use_deform=False is highly recommended for CPU training 
    # to avoid OOM and extremely slow execution.
    use_deform = not args.no_deform
    model = RADANet(num_classes=1, use_deform=use_deform).to(device)

    # CRITICAL: Freeze Backbone by default
    set_backbone_trainable(model, trainable=False)
    
    # ── data ─────────────────────────────────────────────────────────
    train_dir = os.path.join(args.dataroot, "train")
    val_dir   = os.path.join(args.dataroot, "val")
    if not os.path.exists(val_dir):
        val_dir = os.path.join(args.dataroot, "valid")
    if not os.path.exists(val_dir):
        val_dir = os.path.join(args.dataroot, "test")

    train_ds = RoadDataset(train_dir, img_size=args.image_size, mode="train")
    val_ds   = RoadDataset(val_dir,   img_size=args.image_size, mode="test")

    # CPU optimization: num_workers=0
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)

    print(f"  Train samples: {len(train_ds)}   Val samples: {len(val_ds)}")

    # ── loss / optimizer / scheduler ─────────────────────────────────
    criterion = ImprovedRADALoss(w_bce=0.3, w_dice=0.5, w_focal=0.2)

    # Only optimize parameters that require_grad (RAM, DAM, Decoder)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── tracking ─────────────────────────────────────────────────────
    history       = []
    best_iou      = 0.0
    start_epoch   = 1
    patience_cnt  = 0
    accum_steps   = args.accum_steps

    # ── resume logic ─────────────────────────────────────────────────
    checkpoint_path = "best_model.pth"
    log_path = "training_logs_cpu.csv"

    if args.resume:
        if os.path.exists(checkpoint_path):
            print(f"  [Resume] Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            start_epoch = checkpoint.get("epoch", 0) + 1
            best_iou = checkpoint.get("best_iou", 0.0)
            
            # Restore history from CSV
            if os.path.exists(log_path):
                print(f"  [Resume] Restoring history from: {log_path}")
                try:
                    df_old = pd.read_csv(log_path)
                    history = df_old.to_dict("records")
                    print(f"  * History restored ({len(history)} epochs)")
                except Exception as e:
                    print(f"  ! Could not restore history: {e}")
            
            # Sync scheduler
            for _ in range(start_epoch - 1):
                scheduler.step()
            
            print(f"  [Resume] Continuing from Epoch {start_epoch} (Best IoU so far: {best_iou:.4f})")
        else:
            print(f"  ! No checkpoint found at {checkpoint_path}, starting from scratch.")

    t_start = time.time()

    # =================================================================
    # EPOCH LOOP
    # =================================================================
    for epoch in range(start_epoch, args.epochs + 1):

        # Optional: Unfreeze encoder for the last 15 epochs for fine-tuning
        if args.unfreeze_at > 0 and epoch == (args.epochs - args.unfreeze_at + 1):
            print(f"\n  [Phase 2] Unfreezing backbone for last {args.unfreeze_at} epochs...")
            set_backbone_trainable(model, trainable=True)
            # Re-initialize optimizer to include all parameters
            optimizer = optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.unfreeze_at)

        # ─────────────────────────────────────────────────────────────
        # TRAIN
        # ─────────────────────────────────────────────────────────────
        model.train()
        train_loss_sum = 0.0
        train_metrics  = {"prec": [], "rec": [], "f1": [], "iou": []}

        optimizer.zero_grad()
        if epoch % 5 == 0:
            gc.collect()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]", leave=False)

        for step, (images, masks) in enumerate(pbar):
            images = images.to(device)
            masks  = masks.to(device)

            # Forward + Backward
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Gradient Accumulation logic
            (loss / accum_steps).backward()

            train_loss_sum += loss.item()

            with torch.no_grad():
                p, r, f, iou = calculate_metrics(outputs, masks)
            train_metrics["prec"].append(p)
            train_metrics["rec"].append(r)
            train_metrics["f1"].append(f)
            train_metrics["iou"].append(iou)

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                # Manual memory cleanup at the end of the step
                del outputs, loss
                gc.collect()
            elif (step + 1) % 10 == 0:
                # More frequent cleanup during accumulation to prevent creep
                gc.collect()

            # Reduced logging inside loop for speed
            if step % 50 == 0:
                pbar.set_postfix({"loss": f"{train_loss_sum/(step+1):.4f}", "iou": f"{iou:.4f}"})

        # ─────────────────────────────────────────────────────────────
        # VALIDATE
        # ─────────────────────────────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        val_metrics  = {"prec": [], "rec": [], "f1": [], "iou": []}

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]", leave=False):
                images = images.to(device)
                masks  = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)

                val_loss_sum += loss.item()
                p, r, f, iou = calculate_metrics(outputs, masks)
                val_metrics["prec"].append(p)
                val_metrics["rec"].append(r)
                val_metrics["f1"].append(f)
                val_metrics["iou"].append(iou)

        # ── epoch summary ────────────────────────────────────────────
        n_train = max(len(train_loader), 1)
        n_val   = max(len(val_loader), 1)

        row = {
            "epoch":      epoch,
            "train_loss": train_loss_sum / n_train,
            "val_loss":   val_loss_sum / n_val,
            "train_iou":  np.mean(train_metrics["iou"]),
            "train_f1":   np.mean(train_metrics["f1"]),
            "val_prec":   np.mean(val_metrics["prec"]),
            "val_rec":    np.mean(val_metrics["rec"]),
            "val_f1":     np.mean(val_metrics["f1"]),
            "val_iou":    np.mean(val_metrics["iou"]),
            "lr":         optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        print(f"  Epoch {epoch:>3d} | Loss: {row['train_loss']:.4f}/{row['val_loss']:.4f} | "
              f"IoU: {row['val_iou']:.4f} | F1: {row['val_f1']:.4f} | LR: {row['lr']:.2e}")

        scheduler.step()

        # ── checkpoint & early stopping ──────────────────────────────
        if row["val_iou"] > best_iou:
            best_iou     = row["val_iou"]
            patience_cnt = 0
            torch.save({
                "epoch":      epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_iou":  best_iou,
            }, "best_model.pth")
            print(f"  ✓ New best model (IoU={best_iou:.4f})")
        else:
            patience_cnt += 1

        # ── RESEARCH-GRADE FAILURE CHECKS (Task Objective) ───────────
        if epoch > 2: # Give it a few epochs to start learning
            v_prec = row["val_prec"]
            v_rec  = row["val_rec"]
            v_iou  = row["val_iou"]

            if v_rec > 0.99:
                print(f"\n❌ FAILURE CONDITION: Recall={v_rec:.4f} (Model predicting all road). Stopping.")
                break
            if v_prec > 0.99:
                print(f"\n❌ FAILURE CONDITION: Precision={v_prec:.4f} (Model predicting no road). Stopping.")
                break
            if epoch > 5 and v_iou < 0.01:
                print(f"\n❌ FAILURE CONDITION: IoU constant at {v_iou:.4f} (Not learning). Stopping.")
                break

        if patience_cnt >= args.patience:
            print(f"\n  ⏹ Early stopping at epoch {epoch}")
            break

    # =================================================================
    # POST-TRAINING
    # =================================================================
    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  Training complete! Best IoU = {best_iou:.4f}")
    print(f"  Total time: {elapsed / 3600:.2f} hours")
    print(f"{'=' * 60}")

    # Save logs and plots
    df = pd.DataFrame(history)
    df.to_csv("training_logs_cpu.csv", index=False)
    plot_training_results(df, save_dir="graphs/")
    print("  Results saved to training_logs_cpu.csv and graphs/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RADANet CPU Optimized Training")

    parser.add_argument("--dataroot", type=str, default="data", help="Root with train/ and val/")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--accum_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--unfreeze_at", type=int, default=20, help="Epochs to unfreeze at end (0 to disable)")
    parser.add_argument("--resume", action="store_true", help="Resume training from best_radanet_cpu.pth")
    parser.add_argument("--no_deform", action="store_true", help="Disable Deformable Convolutions for CPU efficiency")

    args = parser.parse_args()
    train(args)
