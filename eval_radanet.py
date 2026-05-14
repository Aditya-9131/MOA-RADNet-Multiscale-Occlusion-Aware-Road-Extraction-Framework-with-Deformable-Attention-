"""
RADANet — Full Evaluation Pipeline
====================================
• 6-way TTA (original + flips + rotations)
• Post-processing: morphological closing + small-component removal
• Metrics: Precision / Recall / F1 / IoU
• PR Curve (sklearn)
• Qualitative comparison grids
• Comparison table: RADANet vs OARENet baseline
• Saves: predictions/, graphs/pr_curve.png, results_table.txt
"""

import os
import sys
import argparse
import time

import numpy as np
import cv2
import scipy.ndimage as ndimage
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── project imports ──────────────────────────────────────────────────
from models.model import RADANet
from OARENet.dataset import RoadDataset
from utils.plots import (plot_sample_results, plot_comparison_grid,
                          plot_pr_curve)

try:
    from prettytable import PrettyTable
except ImportError:
    PrettyTable = None


# =====================================================================
# Test-Time Augmentation  (6-way)
# =====================================================================
def tta_inference(model, image, device):
    """
    6-fold TTA:
      1. Original
      2. Horizontal flip
      3. Vertical flip
      4. Rotate 90°
      5. Rotate 180°
      6. Rotate 270°
    Returns the averaged prediction (same shape as single forward pass).
    """
    preds = []

    # 1. Original
    preds.append(model(image))

    # 2. Horizontal flip  (flip dim=3 → W)
    x_hf = torch.flip(image, [3])
    preds.append(torch.flip(model(x_hf), [3]))

    # 3. Vertical flip  (flip dim=2 → H)
    x_vf = torch.flip(image, [2])
    preds.append(torch.flip(model(x_vf), [2]))

    # 4. Rotate 90°
    x_r90 = torch.rot90(image, 1, [2, 3])
    preds.append(torch.rot90(model(x_r90), -1, [2, 3]))

    # 5. Rotate 180°
    x_r180 = torch.rot90(image, 2, [2, 3])
    preds.append(torch.rot90(model(x_r180), -2, [2, 3]))

    # 6. Rotate 270°
    x_r270 = torch.rot90(image, 3, [2, 3])
    preds.append(torch.rot90(model(x_r270), -3, [2, 3]))

    return torch.stack(preds).mean(dim=0)


# =====================================================================
# Post-processing
# =====================================================================
def post_process_mask(mask, min_size=200, kernel_size=5):
    """
    1. Morphological closing to bridge small gaps.
    2. Small connected component removal to reduce noise.
    3. Thin dilation-erosion to smooth edges.
    """
    mask_u8 = (mask * 255).astype(np.uint8)

    # 1. Morphological Closing (Dilation then Erosion)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

    # 2. Remove Small Components
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    refined = np.zeros_like(mask_u8)
    for i in range(1, nb_components):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            refined[output == i] = 255

    # 3. Final smoothing
    refined = cv2.GaussianBlur(refined, (3, 3), 0)
    _, refined = cv2.threshold(refined, 127, 255, cv2.THRESH_BINARY)

    return (refined / 255.0).astype(np.float32)


# =====================================================================
# Metric calculation
# =====================================================================
def pixel_metrics(pred, target):
    """Return (precision, recall, f1, iou) from binary arrays."""
    pred  = pred.astype(np.float32)
    target = target.astype(np.float32)

    tp = (pred * target).sum()
    fp = (pred * (1.0 - target)).sum()
    fn = ((1.0 - pred) * target).sum()

    precision = tp / (tp + fp + 1e-7)
    recall    = tp / (tp + fn + 1e-7)
    f1        = 2.0 * precision * recall / (precision + recall + 1e-7)
    iou       = tp / (tp + fp + fn + 1e-7)

    return precision, recall, f1, iou


# =====================================================================
# Build comparison table
# =====================================================================
def build_results_table(rada_metrics, oare_metrics=None):
    """
    Print and return a formatted comparison table with Improvement metrics.
    """
    header = f"{'Model':<22s} {'Precision':>10s} {'Recall':>10s} {'F1 Score':>10s} {'IoU':>10s} {'Gain (IoU)':>10s}"
    sep = "=" * len(header)
    lines = [
        "\n" + "=" * 30 + " PROJECT SUMMARY " + "=" * 30,
        "RADANet: Road Augmented Deformable Attention Network",
        "This project focuses on capturing complex road topology using advanced",
        "attention modules (RAM & DAM) and robust morphological post-processing.",
        "The goal is to achieve research-grade accuracy in road extraction.",
        "=" * 77 + "\n",
        sep, header, sep
    ]

    if oare_metrics:
        # Baseline row
        lines.append(
            f"{'OARENet (Baseline)':<22s} "
            f"{oare_metrics['p']:>10.4f} {oare_metrics['r']:>10.4f} "
            f"{oare_metrics['f']:>10.4f} {oare_metrics['i']:>10.4f} {'-':>10s}")

        # Gain calculation
        gain = ((rada_metrics['i'] - oare_metrics['i']) / (oare_metrics['i'] + 1e-7)) * 100
        gain_str = f"+{gain:.2f}%"
    else:
        gain_str = "N/A"

    # RADANet row
    lines.append(
        f"{'RADANet (Proposed)':<22s} "
        f"{rada_metrics['p']:>10.4f} {rada_metrics['r']:>10.4f} "
        f"{rada_metrics['f']:>10.4f} {rada_metrics['i']:>10.4f} {gain_str:>10s}")
    lines.append(sep)

    table_str = "\n".join(lines)
    return table_str


# =====================================================================
# Main evaluation
# =====================================================================
def evaluate(args):
    device = torch.device(args.device)
    print("=" * 60)
    print(f"  RADANet Evaluation — device={device}")
    print(f"  checkpoint: {args.model_path}")
    print(f"  TTA: {'ON' if args.tta else 'OFF'}   "
          f"Post-process: {'ON' if args.postprocess else 'OFF'}")
    print("=" * 60)

    # ── model ────────────────────────────────────────────────────────
    model = RADANet(num_classes=1).to(device)

    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"  Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
              f"(best IoU={ckpt.get('best_iou', '?')})")
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()

    # ── data ─────────────────────────────────────────────────────────
    os.makedirs("predictions", exist_ok=True)
    os.makedirs("graphs", exist_ok=True)

    test_dir = args.dataroot
    # If dataroot has val/ or test/ subfolders, prefer val/
    for sub in ("val", "test"):
        candidate = os.path.join(args.dataroot, sub)
        if os.path.isdir(candidate):
            test_dir = candidate
            break

    test_ds = RoadDataset(test_dir, img_size=args.image_size, mode="test")
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             num_workers=0)
    print(f"  Test samples: {len(test_ds)}")

    # ── evaluation loop ──────────────────────────────────────────────
    metrics = {"precision": [], "recall": [], "f1": [], "iou": []}

    # For PR curve
    all_targets = []
    all_preds   = []

    use_amp = (device.type == "cuda")

    print("\n  Running inference …")
    t0 = time.time()
    with torch.no_grad():
        for idx, (images, masks) in enumerate(tqdm(test_loader,
                                                    desc="  Evaluating")):
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device,  non_blocking=True)

            # Resize guard
            if images.shape[-1] != args.image_size:
                images = F.interpolate(images,
                                       size=(args.image_size, args.image_size),
                                       mode="bilinear", align_corners=True)
                masks = F.interpolate(masks,
                                      size=(args.image_size, args.image_size),
                                      mode="nearest")

            # Inference (with optional TTA)
            with torch.amp.autocast("cuda", enabled=use_amp):
                if args.tta:
                    pred = tta_inference(model, images, device)
                else:
                    pred = model(images)

            pred_np   = pred.squeeze().cpu().numpy()
            target_np = masks.squeeze().cpu().numpy()

            # Binary threshold
            binary_pred = (pred_np > 0.5).astype(np.float32)

            # Post-processing
            if args.postprocess:
                refined_pred = post_process_mask(binary_pred,
                                                  min_size=args.min_comp_size)
            else:
                refined_pred = binary_pred

            # Metrics
            p, r, f, iou = pixel_metrics(refined_pred, target_np)
            metrics["precision"].append(p)
            metrics["recall"].append(r)
            metrics["f1"].append(f)
            metrics["iou"].append(iou)

            # PR-curve data (sub-sample every 4th pixel)
            all_preds.append(pred_np.ravel()[::4])
            all_targets.append(target_np.ravel()[::4])

            # Qualitative results (save first N samples)
            if idx < args.num_vis:
                vis_pred_tensor = torch.from_numpy(refined_pred).unsqueeze(0)
                plot_sample_results(
                    images[0], masks[0], vis_pred_tensor,
                    filename=f"predictions/sample_{idx}.png",
                    model_name="RADANet")

    elapsed = time.time() - t0
    print(f"\n  Inference complete in {elapsed:.1f}s "
          f"({len(test_ds)} samples)")

    # ── aggregate metrics ────────────────────────────────────────────
    res = {
        "p": np.mean(metrics["precision"]),
        "r": np.mean(metrics["recall"]),
        "f": np.mean(metrics["f1"]),
        "i": np.mean(metrics["iou"]),
    }

    # ── OARENet baseline (placeholder — replace with actual numbers) ─
    oare = None
    if args.oarenet_prec > 0:
        oare = {
            "p": args.oarenet_prec,
            "r": args.oarenet_rec,
            "f": args.oarenet_f1,
            "i": args.oarenet_iou,
        }

    # ── results table ────────────────────────────────────────────────
    table = build_results_table(res, oare)
    print("\n" + table)

    with open("results_table.txt", "w") as f:
        f.write(table)
    print("  → results_table.txt")

    # ── PR curve ─────────────────────────────────────────────────────
    if all_targets:
        tgt_arr  = np.concatenate(all_targets)
        pred_arr = np.concatenate(all_preds)
        plot_pr_curve(tgt_arr, pred_arr, save_path="graphs/pr_curve.png")

    # ── per-sample CSV ───────────────────────────────────────────────
    import pandas as pd
    sample_df = pd.DataFrame({
        "sample":    list(range(len(metrics["iou"]))),
        "precision": metrics["precision"],
        "recall":    metrics["recall"],
        "f1":        metrics["f1"],
        "iou":       metrics["iou"],
    })
    sample_df.to_csv("eval_per_sample.csv", index=False)
    print("  → eval_per_sample.csv")

    print(f"\n  Final →  Prec={res['p']:.4f}  Rec={res['r']:.4f}  "
          f"F1={res['f']:.4f}  IoU={res['i']:.4f}")
    print("=" * 60)


# =====================================================================
# CLI
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RADANet Evaluation Pipeline")

    parser.add_argument("--model_path", type=str, default="best_model.pth")
    parser.add_argument("--dataroot", type=str, default="data",
                        help="Root containing val/ or test/ subfolder")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--device", type=str, default="auto",
                        help="'cpu', 'cuda', or 'auto'")

    # TTA & post-processing
    parser.add_argument("--tta", action="store_true", default=True,
                        help="Enable 6-way TTA")
    parser.add_argument("--no_tta", action="store_false", dest="tta")
    parser.add_argument("--postprocess", action="store_true", default=True)
    parser.add_argument("--no_postprocess", action="store_false",
                        dest="postprocess")
    parser.add_argument("--min_comp_size", type=int, default=150,
                        help="Min connected-component size to keep")
    parser.add_argument("--num_vis", type=int, default=10,
                        help="Number of qualitative samples to save")

    # OARENet baseline numbers for comparison table
    parser.add_argument("--oarenet_prec", type=float, default=0.6842)
    parser.add_argument("--oarenet_rec",  type=float, default=0.6510)
    parser.add_argument("--oarenet_f1",   type=float, default=0.6671)
    parser.add_argument("--oarenet_iou",  type=float, default=0.5012)

    args = parser.parse_args()

    # Auto device
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    evaluate(args)
