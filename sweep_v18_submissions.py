"""
Generate ConvNeXt-only and threshold-swept v18 submissions from saved or fresh test probs.

Usage (venv312):
  python sweep_v18_submissions.py              # infer all ckpts, save probs, write CSVs
  python sweep_v18_submissions.py --from-npz   # reuse test_probs_v18.npz only
"""
from env_check import relaunch_with_venv312

relaunch_with_venv312()

import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from env_check import require_venv312
import inference_v18 as inf

BASE_DIR = r"D:\CV"
PROBS_FILE = os.path.join(BASE_DIR, "test_probs_v18.npz")
OOF_FILE = os.path.join(BASE_DIR, "oof_probs_v18.npz")
TRAIN_POS_RATE = 0.583


def threshold_for_positive_rate(probs: np.ndarray, target_rate: float) -> float:
    """Threshold so fraction(probs >= t) ~= target_rate."""
    q = 1.0 - target_rate
    return float(np.quantile(probs, q))


def oof_threshold_sweep(probs: np.ndarray, y: np.ndarray, steps: int = 199):
    rows = []
    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0.01, 0.99, steps):
        f1 = f1_score(y, (probs >= t).astype(int), zero_division=0)
        pos = (probs >= t).mean()
        rows.append((t, f1, pos))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return rows, best_t, best_f1


def write_submission(image_ids, probs: np.ndarray, threshold: float, path: str) -> dict:
    preds = (probs >= threshold).astype(int)
    pd.DataFrame({"image_id": image_ids, "target": preds}).to_csv(path, index=False)
    return {"path": path, "threshold": threshold, "pos_rate": float(preds.mean()), "positives": int(preds.sum())}


@torch.inference_mode()
def collect_test_probs(folds: int, tta: int, use_full: bool) -> dict[str, np.ndarray]:
    require_venv312(require_cuda=True)
    models = ["convnext_large", "tf_efficientnetv2_m"]
    df = pd.read_csv(os.path.join(BASE_DIR, "images", "sample_submission.csv"))
    test_dir = os.path.join(BASE_DIR, "images", "test_images")
    rois = inf.load_rois()
    loader = DataLoader(
        inf.TestDS(df, test_dir, rois), batch_size=8, shuffle=False, num_workers=0, collate_fn=inf.collate,
    )
    device = torch.device("cuda")
    per_ckpt: dict[str, np.ndarray] = {}

    for m in models:
        ckpts = inf.discover_checkpoints(m, folds, use_full=use_full)
        for kind, path, model_name in ckpts:
            key = f"{model_name}_{kind}_{os.path.basename(os.path.dirname(path))}"
            imgsz = inf.get_imgsz(model_name, 0)
            print(f"Infer {key} | imgsz={imgsz}", flush=True)
            prob_sum = np.zeros(len(df), dtype=np.float64)
            inf.run_checkpoint(path, model_name, loader, device, tta, prob_sum, imgsz)
            per_ckpt[key] = prob_sum.copy()

    conv_keys = [k for k in per_ckpt if k.startswith("convnext_large")]
    eff_keys = [k for k in per_ckpt if k.startswith("tf_efficientnetv2_m")]
    stacks = {
        "dual_all": np.mean(np.stack(list(per_ckpt.values())), axis=0),
        "convnext_only": np.mean(np.stack([per_ckpt[k] for k in conv_keys]), axis=0),
        "efficientnet_only": np.mean(np.stack([per_ckpt[k] for k in eff_keys]), axis=0),
        "dual_folds_only": np.mean(
            np.stack([per_ckpt[k] for k in per_ckpt if "_fold_" in k]), axis=0,
        ),
        "convnext_folds_only": np.mean(
            np.stack([per_ckpt[k] for k in conv_keys if "_fold_" in k]), axis=0,
        ),
    }
    return {"image_id": df["image_id"].values, **stacks, **{f"ckpt_{k}": v for k, v in per_ckpt.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-npz", action="store_true", help="Skip GPU infer; use existing test_probs_v18.npz")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--tta", type=int, default=12)
    parser.add_argument("--no-full", action="store_true")
    args = parser.parse_args()

    if args.from_npz:
        if not os.path.exists(PROBS_FILE):
            raise FileNotFoundError(f"Missing {PROBS_FILE}. Run without --from-npz first.")
        data = np.load(PROBS_FILE, allow_pickle=True)
        image_ids = data["image_id"]
        prob_sets = {
            "dual_all": data["dual_all"],
            "convnext_only": data["convnext_only"],
            "efficientnet_only": data["efficientnet_only"],
            "dual_folds_only": data["dual_folds_only"],
            "convnext_folds_only": data["convnext_folds_only"],
        }
    else:
        out = collect_test_probs(args.folds, args.tta, use_full=not args.no_full)
        image_ids = out["image_id"]
        prob_sets = {k: out[k] for k in ("dual_all", "convnext_only", "efficientnet_only", "dual_folds_only", "convnext_folds_only")}
        save = {"image_id": image_ids}
        save.update(prob_sets)
        np.savez(PROBS_FILE, **save)
        print(f"Saved test probs: {PROBS_FILE}", flush=True)

    oof = np.load(OOF_FILE)
    oof_probs, oof_y = oof["prob"], oof["target"]
    rows, oof_best_t, oof_best_f1 = oof_threshold_sweep(oof_probs, oof_y)
    rate_t = threshold_for_positive_rate(oof_probs, TRAIN_POS_RATE)

    print(f"\nOOF best F1={oof_best_f1:.4f} @ threshold={oof_best_t:.4f}", flush=True)
    print(f"OOF train-rate threshold (~{100*TRAIN_POS_RATE:.1f}% pos): {rate_t:.4f}", flush=True)
    print("\nOOF sweep (selected thresholds):", flush=True)
    for t, f1, pos in rows:
        if abs(t - oof_best_t) < 0.002 or abs(t - rate_t) < 0.002 or abs(t - 0.5) < 0.001:
            print(f"  t={t:.4f}  OOF F1={f1:.4f}  pos_rate={pos:.3f}", flush=True)

    thresholds = {
        "oof_best": oof_best_t,
        "train_rate": rate_t,
        "fixed_050": 0.5,
        "oof_best_m005": max(0.01, oof_best_t - 0.005),
        "oof_best_p005": min(0.99, oof_best_t + 0.005),
    }

    variants = [
        ("convnext_only", prob_sets["convnext_only"]),
        ("dual_all", prob_sets["dual_all"]),
        ("convnext_folds_only", prob_sets["convnext_folds_only"]),
        ("dual_folds_only", prob_sets["dual_folds_only"]),
    ]

    print(f"\n{'='*60}\nWriting submissions\n{'='*60}", flush=True)
    for variant_name, probs in variants:
        for th_name, th in thresholds.items():
            fname = os.path.join(BASE_DIR, f"submission_v18_{variant_name}_{th_name}.csv")
            info = write_submission(image_ids, probs, th, fname)
            print(
                f"{os.path.basename(fname)} | pos={info['positives']}/{len(probs)} "
                f"({100*info['pos_rate']:.2f}%) | t={info['threshold']:.4f}",
                flush=True,
            )

    primary = os.path.join(BASE_DIR, "submission_v18_convnext_oof.csv")
    write_submission(image_ids, prob_sets["convnext_only"], oof_best_t, primary)
    print(f"\nRecommended first upload: {primary}", flush=True)


if __name__ == "__main__":
    main()
