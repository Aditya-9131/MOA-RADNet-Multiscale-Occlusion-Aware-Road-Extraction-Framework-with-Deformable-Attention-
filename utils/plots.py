"""
RADANet — Publication-Quality Plotting Utilities
=================================================
Generates clean, labeled, 300-dpi graphs for:
  (A) Training / Validation Loss curve
  (B) IoU curve
  (C) F1 Score curve
  (D) Precision-Recall curve (sklearn-based)
  (E) Qualitative comparison grids
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import precision_recall_curve, average_precision_score

def set_publication_style():
    """Configure matplotlib for high-quality figures."""
    plt.style.use('seaborn-v0_8-paper') # Use a clean base style
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "axes.linewidth": 1.2,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "legend.framealpha": 0.9,
        "figure.titlesize": 18,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.8,
        "lines.linewidth": 2.5,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.figsize": (8, 6),
    })

def plot_loss_curve(df, save_dir="graphs/"):
    set_publication_style()
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots()
    epochs = df["epoch"]
    ax.plot(epochs, df["train_loss"], color="#2563EB", marker="o", markersize=4, label="Training Loss")
    ax.plot(epochs, df["val_loss"], color="#DC2626", linestyle="--", marker="s", markersize=4, label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True)
    fig.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close(fig)

def plot_iou_curve(df, save_dir="graphs/"):
    set_publication_style()
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots()
    epochs = df["epoch"]
    ax.plot(epochs, df["val_iou"], color="#059669", marker="^", markersize=5, label="Validation IoU")
    if "train_iou" in df.columns:
        ax.plot(epochs, df["train_iou"], color="#6366F1", linestyle="--", label="Training IoU")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("IoU")
    ax.set_title("Intersection over Union (IoU)")
    ax.legend(loc="lower right")
    ax.grid(True)
    fig.savefig(os.path.join(save_dir, "iou_curve.png"))
    plt.close(fig)

def plot_f1_curve(df, save_dir="graphs/"):
    set_publication_style()
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots()
    epochs = df["epoch"]
    ax.plot(epochs, df["val_f1"], color="#9333EA", marker="D", markersize=4, label="Validation F1 Score")
    if "train_f1" in df.columns:
        ax.plot(epochs, df["train_f1"], color="#F59E0B", linestyle="--", label="Training F1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score Curve")
    ax.legend(loc="lower right")
    ax.grid(True)
    fig.savefig(os.path.join(save_dir, "f1_curve.png"))
    plt.close(fig)

def plot_pr_curve(all_targets, all_preds, save_path="graphs/pr_curve.png"):
    set_publication_style()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    precision, recall, _ = precision_recall_curve(all_targets, all_preds)
    ap = average_precision_score(all_targets, all_preds)
    fig, ax = plt.subplots()
    ax.plot(recall, precision, color="#2563EB", label=f"RADANet (AP = {ap:.4f})")
    ax.fill_between(recall, precision, alpha=0.15, color="#2563EB")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve")
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower left")
    ax.grid(True)
    fig.savefig(save_path)
    plt.close(fig)

def plot_training_results(df, save_dir="graphs/"):
    plot_loss_curve(df, save_dir)
    plot_iou_curve(df, save_dir)
    plot_f1_curve(df, save_dir)

def plot_sample_results(image, target, prediction, filename="prediction.png", model_name="RADANet"):
    set_publication_style()
    img = image.permute(1, 2, 0).cpu().numpy()
    img = (img * 0.5) + 0.5 # de-normalize
    img = np.clip(img, 0.0, 1.0)
    gt = target.squeeze().cpu().numpy()
    pred = prediction.squeeze().cpu().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].set_title("Input Image")
    axes[0].axis("off")
    axes[1].imshow(gt, cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")
    axes[2].imshow(pred, cmap="gray")
    axes[2].set_title(f"{model_name} Prediction")
    axes[2].axis("off")
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close(fig)

def plot_comparison_grid(image, target, rada_pred, oare_pred=None, filename="comparison.png"):
    set_publication_style()
    img = image.permute(1, 2, 0).cpu().numpy()
    img = (img * 0.5) + 0.5
    img = np.clip(img, 0.0, 1.0)
    gt = target.squeeze().cpu().numpy()
    rp = rada_pred.squeeze().cpu().numpy()
    ncols = 4 if oare_pred is not None else 3
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
    axes[0].imshow(img); axes[0].set_title("Input"); axes[0].axis("off")
    axes[1].imshow(gt, cmap="gray"); axes[1].set_title("Ground Truth"); axes[1].axis("off")
    axes[2].imshow(rp, cmap="gray"); axes[2].set_title("RADANet"); axes[2].axis("off")
    if oare_pred is not None:
        op = oare_pred.squeeze().cpu().numpy()
        axes[3].imshow(op, cmap="gray")
        axes[3].set_title("OARENet")
        axes[3].axis("off")
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close(fig)
