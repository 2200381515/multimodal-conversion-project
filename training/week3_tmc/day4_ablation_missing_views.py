from __future__ import annotations

import os
import argparse
import warnings
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from models.tmc.tmc_model_minimal import TMCMinimalModel

# Silence optional pandas bottleneck warning (doesn't affect correctness)
warnings.filterwarnings("ignore", message=r".*Pandas requires version.*bottleneck.*")


DEFAULT_KEYS = {
    "x_scale": "x_scale",
    "x_sc": "x_sc",
    "x_fc": "x_fc",
    "y": "y",
    "modality_mask": "modality_mask",
}

DEFAULT_COHORT_CSV = r"F:\multimodal-conversion-project\results\week2\day1\cohort_filtered.csv"
DEFAULT_FOLDS_DIR = r"F:\multimodal-conversion-project\results\week2\day1\folds"
DEFAULT_CKPT_DIR = r"F:\multimodal-conversion-project\results\week3_tmc"
DEFAULT_OUT_CSV = r"F:\multimodal-conversion-project\results\week3_tmc\day4_missingness_ablation.csv"


def parse_args():
    p = argparse.ArgumentParser(
        description="Day4: Missing-modality robustness ablation + low-cost diagnostics."
    )

    p.add_argument("--cohort_csv", type=str, default=DEFAULT_COHORT_CSV)
    p.add_argument("--data_root", type=str, default="")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--folds_dir", type=str, default=DEFAULT_FOLDS_DIR)
    p.add_argument("--ckpt_dir", type=str, default=DEFAULT_CKPT_DIR)
    p.add_argument("--fold", type=int, default=-1, help="0..K-1; -1 means evaluate all folds.")

    # Conditions
    p.add_argument(
        "--conditions",
        type=str,
        default="all",
        help=(
            "Use 'all' (default) or a comma list among: "
            "none,drop_scale,drop_sc,drop_fc,only_scale,only_sc,only_fc"
        ),
    )

    p.add_argument(
        "--base_mask",
        type=str,
        default="dataset",
        choices=["dataset", "all1"],
        help=(
            "How to initialize modality masks before overriding.\n"
            "dataset: start from dataset-provided modality_mask if present; otherwise all ones.\n"
            "all1: always start from all ones (treat as all-present baseline)."
        ),
    )

    p.add_argument(
        "--drop_mode",
        type=str,
        default="mask_only",
        choices=["mask_only", "mask_and_zero"],
        help=(
            "How to simulate missing view.\n"
            "mask_only: only override modality_masks (recommended with current fusion).\n"
            "mask_and_zero: also set the dropped view input tensors to 0."
        ),
    )

    # Threshold scan (for best_F1)
    p.add_argument("--scan_thresholds", action="store_true", default=True)
    p.add_argument("--no_scan_thresholds", action="store_false", dest="scan_thresholds")
    p.add_argument("--thr_min", type=float, default=0.05)
    p.add_argument("--thr_max", type=float, default=0.95)
    p.add_argument("--thr_steps", type=int, default=91)

    p.add_argument("--out_csv", type=str, default=DEFAULT_OUT_CSV)

    # ---- Low-cost diagnostics ----
    p.add_argument("--diag", action="store_true", help="Enable low-cost diagnostics prints.")
    p.add_argument("--diag_n_batches", type=int, default=2, help="How many first batches to print diag for (per fold).")
    p.add_argument("--diag_eps_zero", type=float, default=1e-8, help="Threshold to treat values as zero for sparsity.")

    return p.parse_args()


def build_dataset(cohort_csv: str, data_root: str = ""):
    from pipeline.dataset import PatientT0Dataset

    if data_root:
        return PatientT0Dataset(cohort_csv=cohort_csv, data_root=data_root)
    return PatientT0Dataset(cohort_csv=cohort_csv)


def build_loader(dataset, indices: np.ndarray, batch_size: int) -> DataLoader:
    subset = Subset(dataset, indices.tolist())
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)


def infer_dims_from_batch(batch: Dict[str, Any]) -> Tuple[int, int, int]:
    x_scale = batch[DEFAULT_KEYS["x_scale"]].float()
    x_sc = batch[DEFAULT_KEYS["x_sc"]].float()
    x_fc = batch[DEFAULT_KEYS["x_fc"]].float()
    scale_dim = x_scale.size(1)
    sc_dim = x_sc.size(1) * x_sc.size(2)
    fc_dim = x_fc.size(1) * x_fc.size(2)
    return scale_dim, sc_dim, fc_dim


def _read_subjects_txt(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        subs = [line.strip() for line in f.readlines()]
    return [s for s in subs if len(s) > 0]


def list_folds(folds_dir: str) -> List[int]:
    if not os.path.isdir(folds_dir):
        return []
    folds = set()
    for fn in os.listdir(folds_dir):
        if fn.startswith("fold_") and fn.endswith("_test_subjects.txt"):
            try:
                k = int(fn.split("_")[1])
                folds.add(k)
            except Exception:
                pass
    return sorted(list(folds))


def load_val_indices_from_subjects(dataset, folds_dir: str, fold: int) -> np.ndarray:
    test_path = os.path.join(folds_dir, f"fold_{fold}_test_subjects.txt")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Cannot find {test_path}")

    test_subjects = set(_read_subjects_txt(test_path))
    df = dataset.df
    subj_series = df["subject_id"].astype(str).values

    val_idx = [i for i, sid in enumerate(subj_series) if sid in test_subjects]
    if len(val_idx) == 0:
        raise RuntimeError(
            f"Fold {fold} val mapping produced empty indices. "
            f"Check subject_id alignment between cohort_csv and folds subject list."
        )
    return np.array(val_idx, dtype=np.int64)


def find_fold_ckpt(ckpt_dir: str, fold: int) -> str:
    cand = os.path.join(ckpt_dir, f"fold{fold}_minimal_ckpt.pt")
    if os.path.exists(cand):
        return cand
    for fn in os.listdir(ckpt_dir):
        if f"fold{fold}_" in fn and fn.endswith(".pt"):
            return os.path.join(ckpt_dir, fn)
    raise FileNotFoundError(f"Cannot find fold {fold} checkpoint under {ckpt_dir}")


def _torch_load_weights_only(path: str, map_location: str):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def extract_modality_masks(batch: Dict[str, Any]) -> Optional[List[torch.Tensor]]:
    mm = batch.get(DEFAULT_KEYS["modality_mask"], None)
    if mm is None:
        return None
    if isinstance(mm, dict):
        return [mm["scale"].long(), mm["sc"].long(), mm["fc"].long()]
    if torch.is_tensor(mm):
        if mm.dim() == 2 and mm.size(1) == 3:
            return [mm[:, 0].long(), mm[:, 1].long(), mm[:, 2].long()]
        if mm.dim() == 1:
            return [mm.long(), mm.long(), mm.long()]
    return None


def override_masks(
    base_masks: Optional[List[torch.Tensor]],
    batch_size: int,
    device: str,
    base_mask_mode: str,
    condition: str,
) -> List[torch.Tensor]:
    """
    Return a list of 3 masks [B] for scale/sc/fc.

    Supported conditions:
      - none
      - drop_scale / drop_sc / drop_fc
      - only_scale / only_sc / only_fc
    """
    if base_mask_mode == "all1" or base_masks is None:
        masks = [torch.ones(batch_size, dtype=torch.long, device=device) for _ in range(3)]
    else:
        masks = [m.to(device).long() for m in base_masks]

    if condition == "none":
        return masks

    if condition == "drop_scale":
        masks[0] = torch.zeros_like(masks[0])
    elif condition == "drop_sc":
        masks[1] = torch.zeros_like(masks[1])
    elif condition == "drop_fc":
        masks[2] = torch.zeros_like(masks[2])
    elif condition == "only_scale":
        masks[0] = torch.ones_like(masks[0])
        masks[1] = torch.zeros_like(masks[1])
        masks[2] = torch.zeros_like(masks[2])
    elif condition == "only_sc":
        masks[0] = torch.zeros_like(masks[0])
        masks[1] = torch.ones_like(masks[1])
        masks[2] = torch.zeros_like(masks[2])
    elif condition == "only_fc":
        masks[0] = torch.zeros_like(masks[0])
        masks[1] = torch.zeros_like(masks[1])
        masks[2] = torch.ones_like(masks[2])
    else:
        raise ValueError(f"Unknown condition: {condition}")

    return masks


def f1_at_threshold(y: np.ndarray, p: np.ndarray, thr: float) -> float:
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    return float(2 * precision * recall / (precision + recall + 1e-12))


def scan_best_f1(y: np.ndarray, p: np.ndarray, thr_min: float, thr_max: float, steps: int) -> Tuple[float, float]:
    thrs = np.linspace(thr_min, thr_max, steps)
    best_f1 = -1.0
    best_thr = 0.5
    for t in thrs:
        f1 = f1_at_threshold(y, p, float(t))
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(t)
    return float(best_f1), float(best_thr)


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y, p))
    except Exception:
        return float("nan")


def _tensor_stats(x: torch.Tensor, eps_zero: float) -> Dict[str, float]:
    x = x.detach()
    absx = x.abs()
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std().item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "abs_mean": float(absx.mean().item()),
        "zero_frac": float((absx < eps_zero).float().mean().item()),
    }


def eval_one_fold_condition(
    args,
    dataset,
    fold: int,
    ckpt_path: str,
    condition: str,
) -> Dict[str, Any]:
    val_idx = load_val_indices_from_subjects(dataset, args.folds_dir, fold)
    loader = build_loader(dataset, val_idx, args.batch_size)

    # infer dims
    b0 = next(iter(loader))
    scale_dim, sc_dim, fc_dim = infer_dims_from_batch(b0)

    model = TMCMinimalModel(scale_dim, sc_dim, fc_dim).to(args.device)
    sd = _torch_load_weights_only(ckpt_path, map_location=args.device)
    model.load_state_dict(sd)
    model.eval()

    ys, ps, us = [], [], []

    # ---- diagnostics accumulators ----
    diag_done_batches = 0
    mask_presence_sum = torch.zeros(3, dtype=torch.float64)
    mask_presence_n = 0

    with torch.no_grad():
        for batch in loader:
            x_scale = batch[DEFAULT_KEYS["x_scale"]].float().to(args.device)
            x_sc = batch[DEFAULT_KEYS["x_sc"]].float().to(args.device)
            x_fc = batch[DEFAULT_KEYS["x_fc"]].float().to(args.device)
            y = batch[DEFAULT_KEYS["y"]].long().to(args.device)

            base_masks = extract_modality_masks(batch)

            # track dataset mask presence (val-set)
            if base_masks is not None:
                m0, m1, m2 = [m.detach().cpu().float() for m in base_masks]
                mask_presence_sum += torch.tensor([m0.mean(), m1.mean(), m2.mean()], dtype=torch.float64)
                mask_presence_n += 1

            masks = override_masks(
                base_masks=base_masks,
                batch_size=x_scale.size(0),
                device=args.device,
                base_mask_mode=args.base_mask,
                condition=condition,
            )

            if args.drop_mode == "mask_and_zero":
                if condition in ("drop_scale", "only_sc", "only_fc"):
                    x_scale = torch.zeros_like(x_scale)
                if condition in ("drop_sc", "only_scale", "only_fc"):
                    x_sc = torch.zeros_like(x_sc)
                if condition in ("drop_fc", "only_scale", "only_sc"):
                    x_fc = torch.zeros_like(x_fc)

            # ---- low-cost prints: a few first batches per fold per condition ----
            if args.diag and diag_done_batches < args.diag_n_batches:
                ms = [int(m.sum().item()) for m in masks]
                print(f"[DiagMask] fold={fold} cond={condition} | sum_masks(scale/sc/fc)={ms}")

                # input distribution check
                s_sc = _tensor_stats(x_sc, args.diag_eps_zero)
                s_fc = _tensor_stats(x_fc, args.diag_eps_zero)
                diff = (x_sc - x_fc).abs()
                s_diff = _tensor_stats(diff, args.diag_eps_zero)

                # quick “are they basically identical?” heuristic
                # mean abs diff very tiny + max abs diff tiny => almost same tensor
                print(
                    f"[DiagInput] fold={fold} cond={condition} | "
                    f"x_sc(abs_mean={s_sc['abs_mean']:.4g}, std={s_sc['std']:.4g}, zero%={100*s_sc['zero_frac']:.1f}%) | "
                    f"x_fc(abs_mean={s_fc['abs_mean']:.4g}, std={s_fc['std']:.4g}, zero%={100*s_fc['zero_frac']:.1f}%) | "
                    f"|sc-fc|(abs_mean={s_diff['abs_mean']:.4g}, max={s_diff['max']:.4g}, zero%={100*s_diff['zero_frac']:.1f}%)"
                )
                diag_done_batches += 1

            out = model(x_scale, x_sc, x_fc, modality_masks=masks)

            prob_pos = out.fused_prob[:, 1].detach().cpu().numpy()
            u = out.fused_uncertainty.detach().cpu().numpy().reshape(-1)

            ys.append(y.detach().cpu().numpy())
            ps.append(prob_pos)
            us.append(u)

    y_np = np.concatenate(ys)
    p_np = np.concatenate(ps)
    u_np = np.concatenate(us)

    pos = int(y_np.sum())
    neg = int((1 - y_np).sum())

    # dataset mask presence summary (for this fold; printed once per condition is ok but we keep it short)
    if args.diag and mask_presence_n > 0 and condition == "none":
        mean_presence = (mask_presence_sum / float(mask_presence_n)).tolist()
        print(
            f"[DiagDatasetMask] fold={fold} | base_mask_mode=dataset | "
            f"mean_presence(scale/sc/fc)={[round(x,4) for x in mean_presence]}"
        )

    # diagnostics
    print(f"[Diag] fold={fold} | cond={condition} | val size={len(y_np)} | pos={pos} | neg={neg}")
    print(
        f"[Diag] fold={fold} | cond={condition} | prob_pos mean={float(p_np.mean()):.4f} "
        f"| min={float(p_np.min()):.4f} | max={float(p_np.max()):.4f}"
    )

    auc = safe_auc(y_np, p_np)
    f1_05 = f1_at_threshold(y_np, p_np, 0.5)

    best_f1, best_thr = float("nan"), float("nan")
    if args.scan_thresholds:
        best_f1, best_thr = scan_best_f1(y_np, p_np, args.thr_min, args.thr_max, args.thr_steps)

    result = {
        "condition": condition,
        "fold": fold,
        "ckpt": ckpt_path,
        "val_n": int(len(y_np)),
        "val_pos": pos,
        "val_neg": neg,
        "auc": float(auc),
        "f1_05": float(f1_05),
        "best_f1": float(best_f1),
        "best_thr": float(best_thr),
        "mean_u": float(u_np.mean()),
        "p90_u": float(np.quantile(u_np, 0.9)),
        "prob_mean": float(p_np.mean()),
        "prob_min": float(p_np.min()),
        "prob_max": float(p_np.max()),
    }

    print(
        f"[Eval] fold={fold} | cond={condition:<9} | AUC={result['auc']:.4f} "
        f"| F1@0.5={result['f1_05']:.4f} | best_F1={result['best_f1']:.4f}@{result['best_thr']:.2f} "
        f"| mean_u={result['mean_u']:.4f} | u(p90)={result['p90_u']:.4f} | ckpt={ckpt_path}"
    )

    return result


def summarize(results: List[Dict[str, Any]]):
    conds = sorted({r["condition"] for r in results})
    print("[Day4-Summary] condition-wise mean (over folds):")
    for c in conds:
        rr = [r for r in results if r["condition"] == c]
        aucs = [r["auc"] for r in rr if np.isfinite(r["auc"])]
        best_f1s = [r["best_f1"] for r in rr if np.isfinite(r["best_f1"])]
        mean_us = [r["mean_u"] for r in rr if np.isfinite(r["mean_u"])]

        def _m(x: List[float]) -> float:
            return float(np.mean(x)) if len(x) > 0 else float("nan")

        print(
            f"  {c:<9} | AUC_mean={_m(aucs):.4f} | best_F1_mean={_m(best_f1s):.4f} | mean_u_mean={_m(mean_us):.4f}"
        )


def save_csv(results: List[Dict[str, Any]], out_csv: str):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    cols = [
        "condition",
        "fold",
        "val_n",
        "val_pos",
        "val_neg",
        "auc",
        "f1_05",
        "best_f1",
        "best_thr",
        "mean_u",
        "p90_u",
        "prob_mean",
        "prob_min",
        "prob_max",
        "ckpt",
    ]
    lines = [",".join(cols)]
    for r in results:
        row = [
            str(r.get("condition", "")),
            str(int(r.get("fold", -1))),
            str(int(r.get("val_n", 0))),
            str(int(r.get("val_pos", 0))),
            str(int(r.get("val_neg", 0))),
            f"{float(r.get('auc', float('nan'))):.6f}",
            f"{float(r.get('f1_05', float('nan'))):.6f}",
            f"{float(r.get('best_f1', float('nan'))):.6f}",
            f"{float(r.get('best_thr', float('nan'))):.6f}",
            f"{float(r.get('mean_u', float('nan'))):.6f}",
            f"{float(r.get('p90_u', float('nan'))):.6f}",
            f"{float(r.get('prob_mean', float('nan'))):.6f}",
            f"{float(r.get('prob_min', float('nan'))):.6f}",
            f"{float(r.get('prob_max', float('nan'))):.6f}",
            str(r.get("ckpt", "")),
        ]
        lines.append(",".join(row))

    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()

    if not args.cohort_csv:
        args.cohort_csv = DEFAULT_COHORT_CSV
    if not os.path.exists(args.cohort_csv):
        raise FileNotFoundError(f"cohort_csv not found: {args.cohort_csv}")

    dataset = build_dataset(args.cohort_csv, args.data_root)

    if args.fold >= 0:
        folds_to_run = [args.fold]
    else:
        folds_to_run = list_folds(args.folds_dir)
        if len(folds_to_run) == 0:
            raise FileNotFoundError(f"No folds found under folds_dir={args.folds_dir}")

    if args.conditions.strip().lower() == "all":
        conditions = ["none", "drop_scale", "drop_sc", "drop_fc"]
    else:
        conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
        valid = {"none", "drop_scale", "drop_sc", "drop_fc", "only_scale", "only_sc", "only_fc"}
        for c in conditions:
            if c not in valid:
                raise ValueError(f"Unknown condition '{c}'. Valid: {sorted(valid)}")

    print("[Day4] Missing-modality ablation evaluation")
    print(f"  folds: {folds_to_run}")
    print(f"  conditions: {conditions}")
    print(f"  ckpt_dir: {args.ckpt_dir}")
    print(f"  out_csv: {args.out_csv}")
    if args.diag:
        print(f"  [Diag] enabled | base_mask={args.base_mask} | drop_mode={args.drop_mode} | diag_n_batches={args.diag_n_batches}")

    results: List[Dict[str, Any]] = []

    for fold in folds_to_run:
        ckpt_path = find_fold_ckpt(args.ckpt_dir, fold)
        for cond in conditions:
            results.append(eval_one_fold_condition(args, dataset, fold, ckpt_path, cond))

    summarize(results)
    save_csv(results, args.out_csv)
    print(f"[Day4] Saved CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
