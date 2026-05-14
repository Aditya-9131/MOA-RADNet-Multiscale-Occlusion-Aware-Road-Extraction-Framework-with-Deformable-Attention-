"""
RADANet — Full Training Pipeline
=================================
• Epochs = 200, Early stopping on val IoU (patience = 20)
• AdamW + CosineAnnealingLR + FP16 + Gradient clipping
• Composite loss: 0.3 BCE + 0.5 Dice + 0.2 Focal
• Logs: Precision / Recall / F1 / IoU per epoch
• Saves: best_model.pth, training_logs.csv, graphs/
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
    iou       = tp / (tp + fp + fn + 1e-7)

    return precision, recall, f1, iou


# =====================================================================
# Backbone freeze / unfreeze
# =====================================================================
def set_backbone_trainable(model, trainable=True):
    """Freeze or unfreeze the ResNet-50 encoder layers."""
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(f"  RADANet Training — device={device}")
    print(f"  size={args.image_size}  batch={args.batch_size}  "
          f"epochs={args.epochs}  lr={args.lr}")
    print("=" * 60)

    # ── directories ──────────────────────────────────────────────────
    os.makedirs("graphs", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # ── model ────────────────────────────────────────────────────────
    model = RADANet(num_classes=1).to(device)

    # Optional: freeze backbone for the first N epochs
    if args.unfreeze_epoch > 0:
        set_backbone_trainable(model, trainable=False)
    else:
        print("  [Backbone] Full model trainable from epoch 1")

    # ── data ─────────────────────────────────────────────────────────
    train_dir = os.path.join(args.dataroot, "train")
    val_dir   = os.path.join(args.dataroot, "val")
    if not os.path.exists(val_dir):
        val_dir = os.path.join(args.dataroot, "test")

    train_ds = RoadDataset(train_dir, img_size=args.image_size, mode="train")
    val_ds   = RoadDataset(val_dir,   img_size=args.image_size, mode="test")

    nw = 0 if device.type == "cpu" else min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=nw, pin_memory=True)

    print(f"  Train samples: {len(train_ds)}   Val samples: {len(val_ds)}")

    # ── loss / optimizer / scheduler ─────────────────────────────────
    criterion = ImprovedRADALoss(w_bce=0.3, w_dice=0.5, w_focal=0.2)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                      T_max=args.epochs)

    # ── mixed precision ──────────────────────────────────────────────
    use_amp = (device.type == "cuda")
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── gradient accumulation ────────────────────────────────────────
    accum_steps = max(1, args.effective_batch // args.batch_size)

    # ── tracking ─────────────────────────────────────────────────────
    history       = []
    best_iou      = 0.0
    start_epoch   = 1
    patience_cnt  = 0

    # ── resume logic ─────────────────────────────────────────────────
    checkpoint_path = "best_model.pth"
    log_path = "training_logs.csv"

    if args.resume:
        if os.path.exists(checkpoint_path):
            print(f"  [Resume] Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
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

    # For PR-curve at end of training
    all_val_targets = []
    all_val_preds   = []

    t_start = time.time()

    # =================================================================
    # EPOCH LOOP
    # =================================================================
    for epoch in range(start_epoch, args.epochs + 1):

        # — unfreeze backbone when scheduled ─────────────────────────
        if args.unfreeze_epoch > 0 and epoch == args.unfreeze_epoch:
            set_backbone_trainable(model, trainable=True)
            optimizer = optim.AdamW(model.parameters(),
                                    lr=args.lr * 0.1, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - epoch)
            args.unfreeze_epoch = -1   # mark done

        # ─────────────────────────────────────────────────────────────
        # TRAIN
        # ─────────────────────────────────────────────────────────────
        model.train()
        train_loss_sum = 0.0
        train_metrics  = {"prec": [], "rec": [], "f1": [], "iou": []}

        optimizer.zero_grad(set_to_none=True)
        if device.type == "cpu":
            gc.collect()

        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch}/{args.epochs} [Train]",
                    leave=False)

        for step, (images, masks) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device,  non_blocking=True)

            # dynamic resize guard
            if images.shape[-1] != args.image_size:
                images = torch.nn.functional.interpolate(
                    images, size=(args.image_size, args.image_size),
                    mode="bilinear", align_corners=True)
                masks = torch.nn.functional.interpolate(
                    masks, size=(args.image_size, args.image_size),
                    mode="nearest")

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, masks) / accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_loss_sum += loss.item() * accum_steps

            with torch.no_grad():
                p, r, f, iou = calculate_metrics(outputs, masks)
            train_metrics["prec"].append(p)
            train_metrics["rec"].append(r)
            train_metrics["f1"].append(f)
            train_metrics["iou"].append(iou)

            if step % max(1, len(train_loader) // 10) == 0:
                pbar.set_postfix({"loss": f"{loss.item()*accum_steps:.4f}",
                                  "iou": f"{iou:.4f}"})

        # ─────────────────────────────────────────────────────────────
        # VALIDATE
        # ─────────────────────────────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        val_metrics  = {"prec": [], "rec": [], "f1": [], "iou": []}

        # Collect raw predictions for PR-curve during last epoch or best
        collect_pr = (epoch == args.epochs) or (patience_cnt == 0)
        epoch_targets, epoch_preds = [], []

        with torch.no_grad():
            for images, masks in tqdm(val_loader,
                                      desc=f"Epoch {epoch}/{args.epochs} [Val]",
                                      leave=False):
                images = images.to(device, non_blocking=True)
                masks  = masks.to(device,  non_blocking=True)

                if images.shape[-1] != args.image_size:
                    images = torch.nn.functional.interpolate(
                        images, size=(args.image_size, args.image_size),
                        mode="bilinear", align_corners=True)
                    masks = torch.nn.functional.interpolate(
                        masks, size=(args.image_size, args.image_size),
                        mode="nearest")

                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, masks)

                val_loss_sum += loss.item()
                p, r, f, iou = calculate_metrics(outputs, masks)
                val_metrics["prec"].append(p)
                val_metrics["rec"].append(r)
                val_metrics["f1"].append(f)
                val_metrics["iou"].append(iou)

                if collect_pr:
                    # Sub-sample for memory efficiency on large datasets
                    pred_flat = outputs.cpu().numpy().ravel()
                    tgt_flat  = masks.cpu().numpy().ravel()
                    # Take every 4th pixel to keep arrays manageable
                    epoch_preds.append(pred_flat[::4])
                    epoch_targets.append(tgt_flat[::4])

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

        print(f"  Epoch {epoch:>3d}  "
              f"train_loss={row['train_loss']:.4f}  "
              f"val_loss={row['val_loss']:.4f}  "
              f"val_IoU={row['val_iou']:.4f}  "
              f"val_F1={row['val_f1']:.4f}  "
              f"lr={row['lr']:.2e}")

        scheduler.step()

        # ── checkpoint ───────────────────────────────────────────────
        if row["val_iou"] > best_iou:
            best_iou     = row["val_iou"]
            patience_cnt = 0
            torch.save({
                "epoch":      epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_iou":  best_iou,
            }, "best_model.pth")
            print(f"  ✓ New best model saved (IoU={best_iou:.4f})")

            # Keep PR data for the best epoch
            if collect_pr and epoch_targets:
                all_val_targets = epoch_targets
                all_val_preds   = epoch_preds
        else:
            patience_cnt += 1

        # ── periodic CSV save ────────────────────────────────────────
        if epoch % 5 == 0 or epoch == 1 or patience_cnt == 0:
            pd.DataFrame(history).to_csv("training_logs.csv", index=False)

        # ── early stopping ───────────────────────────────────────────
        if patience_cnt >= args.patience:
            print(f"\n  ⏹  Early stopping at epoch {epoch} "
                  f"(no improvement for {args.patience} epochs)")
            break

    # =================================================================
    # POST-TRAINING
    # =================================================================
    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  Training complete — Best IoU = {best_iou:.4f}")
    print(f"  Total time: {elapsed / 60:.1f} min")
    print(f"{'=' * 60}")

    # Final CSV
    df = pd.DataFrame(history)
    df.to_csv("training_logs.csv", index=False)

    # Generate graphs
    print("\n  Generating publication graphs …")
    plot_training_results(df, save_dir="graphs/")

    # PR-Curve
    if all_val_targets:
        targets_arr = np.concatenate(all_val_targets)
        preds_arr   = np.concatenate(all_val_preds)
        plot_pr_curve(targets_arr, preds_arr, save_path="graphs/pr_curve.png")

    # Generate Results Table (Final Epoch / Best)
    best_row = df.loc[df['val_iou'].idxmax()]
    
    # Baseline comparison (OARENet)
    oare_iou = 0.5012
    gain = ((best_row['val_iou'] - oare_iou) / (oare_iou + 1e-7)) * 100

    header = f"{'Model':<22s} {'Precision':>10s} {'Recall':>10s} {'F1 Score':>10s} {'IoU':>10s} {'Gain':>10s}"
    sep = "=" * len(header)
    table_lines = [
        "\n" + "=" * 30 + " PROJECT SUMMARY " + "=" * 30,
        "RADANet: Road Augmented Deformable Attention Network",
        "This project captures complex road topology using advanced RAM/DAM",
        "attention modules and morphological post-processing.",
        "=" * 77 + "\n",
        sep,
        header,
        sep,
        f"{'OARENet (Baseline)':<22s} {'0.6842':>10} {'0.6510':>10} {'0.6671':>10} {'0.5012':>10} {'-':>10}",
        f"{'RADANet (Ours)':<22s} {best_row['val_prec']:>10.4f} {best_row['val_rec']:>10.4f} {best_row['val_f1']:>10.4f} {best_row['val_iou']:>10.4f} {f'+{gain:.2f}%':>10}",
        sep
    ]
    table_str = "\n".join(table_lines)
    print("\n" + table_str)
    
    with open("results_table.txt", "w") as f:
        f.write(table_str)

    # Save a qualitative comparison sample from val set
    print("\n  Saving qualitative comparison grid …")
    from utils.plots import plot_comparison_grid
    model.eval()
    with torch.no_grad():
        # Get one sample from val_loader
        img_val, mask_val = next(iter(val_loader))
        img_val, mask_val = img_val.to(device), mask_val.to(device)
        pred_val = torch.sigmoid(model(img_val))
        plot_comparison_grid(img_val[0], mask_val[0], pred_val[0], filename="final_qualitative_result.png")

    print("\n  All outputs saved:")
    print("    • best_model.pth")
    print("    • training_logs.csv")
    print("    • graphs/  (loss_curve, iou_curve, f1_curve, pr_curve)")
    print("    • results_table.txt")


# =====================================================================
# CLI
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RADANet Training Pipeline")

    parser.add_argument("--dataroot", type=str, default="data",
                        help="Root with train/ and val/ subfolders")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--effective_batch", type=int, default=8,
                        help="Effective batch via gradient accumulation")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=20,
                        help="Early-stop patience on val IoU")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--unfreeze_epoch", type=int, default=0,
                        help="Epoch to unfreeze backbone (0 = always trainable)")
    parser.add_argument("--resume", action="store_true", help="Resume training from best_model.pth")

    args = parser.parse_args()
    train(args)
