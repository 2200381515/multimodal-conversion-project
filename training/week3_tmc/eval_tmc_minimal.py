from __future__ import annotations

import argparse
from typing import Dict, Any, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.tmc.tmc_model_minimal import TMCMinimalModel


DEFAULT_KEYS = {
    "x_scale": "x_scale",
    "x_sc": "x_sc",
    "x_fc": "x_fc",
    "y": "y",
    "modality_mask": "modality_mask",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cohort_csv",
        type=str,
        default=r"F:\multimodal-conversion-project\results\week2\day1\cohort_filtered.csv",
        help="cohort_filtered.csv path (default is week2/day1 cohort_filtered.csv)",
    )
    p.add_argument("--data_root", type=str, default="")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # NEW: whether to print per-view diagnostics
    p.add_argument("--print_view_diag", action="store_true", help="Print per-view uncertainty/strength stats")
    return p.parse_args()


def build_loader(batch_size: int, cohort_csv: str, data_root: str = ""):
    from pipeline.dataset import PatientT0Dataset
    if data_root:
        ds = PatientT0Dataset(cohort_csv=cohort_csv, data_root=data_root)
    else:
        ds = PatientT0Dataset(cohort_csv=cohort_csv)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0), ds


def extract_modality_masks(batch: Dict[str, Any]) -> Optional[List[torch.Tensor]]:
    mm = batch.get(DEFAULT_KEYS["modality_mask"], None)
    if mm is None:
        return None
    if isinstance(mm, dict):
        return [mm["scale"].long(), mm["sc"].long(), mm["fc"].long()]
    if torch.is_tensor(mm) and mm.dim() == 2 and mm.size(1) == 3:
        return [mm[:, 0].long(), mm[:, 1].long(), mm[:, 2].long()]
    return None


def _to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _fmt(x: float) -> str:
    return f"{x:.4f}"


def main():
    args = parse_args()
    loader, ds = build_loader(args.batch_size, args.cohort_csv, args.data_root)

    # infer dims
    b0 = next(iter(loader))
    x_scale = b0[DEFAULT_KEYS["x_scale"]].float()
    x_sc = b0[DEFAULT_KEYS["x_sc"]].float()
    x_fc = b0[DEFAULT_KEYS["x_fc"]].float()
    scale_dim = x_scale.size(1)
    sc_dim = x_sc.size(1) * x_sc.size(2)
    fc_dim = x_fc.size(1) * x_fc.size(2)

    # folds / ckpt discovery: reuse your existing logic if you already have folds loop in your current file.
    # Here we follow your current behavior (which prints fold=0..4), assuming you already have that loop.
    # If your file already has folds loop, just keep it and only insert the "view diag" part below.

    # --- If your current eval has an existing folds loop, KEEP it.
    # For safety, we detect folds as 0..4 by default (same as your logs).
    folds = [0, 1, 2, 3, 4]

    aucs, f1s = [], []
    best_f1s = []

    for fold in folds:
        ckpt_path = rf"F:\multimodal-conversion-project\results\week3_tmc\fold{fold}_minimal_ckpt.pt"
        model = TMCMinimalModel(scale_dim, sc_dim, fc_dim).to(args.device)
        state = torch.load(ckpt_path, map_location=args.device)
        # support either raw state_dict or {"state_dict": ...}
        if isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        elif isinstance(state, dict):
            model.load_state_dict(state)
        else:
            raise RuntimeError(f"Unsupported ckpt format: {type(state)}")

        model.eval()

        ys, ps, us = [], [], []

        # NEW: per-view diagnostic buffers
        view_u_buf = [[], [], []]      # scale/sc/fc
        view_s_buf = [[], [], []]      # strength (sum alpha) per view
        view_p_pos_buf = [[], [], []]  # prob_pos per view

        with torch.no_grad():
            for batch in loader:
                x_scale = batch[DEFAULT_KEYS["x_scale"]].float().to(args.device)
                x_sc = batch[DEFAULT_KEYS["x_sc"]].float().to(args.device)
                x_fc = batch[DEFAULT_KEYS["x_fc"]].float().to(args.device)
                y = batch[DEFAULT_KEYS["y"]].long().to(args.device)

                masks = extract_modality_masks(batch)
                if masks is not None:
                    masks = [m.to(args.device) for m in masks]

                out = model(x_scale, x_sc, x_fc, modality_masks=masks)

                # fused
                prob_pos = out.fused_prob[:, 1].detach().cpu().numpy()
                u = out.fused_uncertainty.detach().cpu().numpy().reshape(-1)

                ys.append(y.detach().cpu().numpy())
                ps.append(prob_pos)
                us.append(u)

                # NEW: per-view diag
                # out.view_uncertainties: list[Tensor(B,1)]  (expected)
                # out.view_strengths:     list[Tensor(B,1)]  (expected)
                # out.view_probs:         list[Tensor(B,2)]
                if args.print_view_diag:
                    if hasattr(out, "view_uncertainties") and out.view_uncertainties is not None:
                        for i in range(3):
                            view_u_buf[i].append(_to_np(out.view_uncertainties[i]).reshape(-1))
                    if hasattr(out, "view_strengths") and out.view_strengths is not None:
                        for i in range(3):
                            view_s_buf[i].append(_to_np(out.view_strengths[i]).reshape(-1))
                    if hasattr(out, "view_probs") and out.view_probs is not None:
                        for i in range(3):
                            view_p_pos_buf[i].append(_to_np(out.view_probs[i][:, 1]).reshape(-1))

        y = np.concatenate(ys)
        p = np.concatenate(ps)
        u = np.concatenate(us)

        # diagnostics you already had
        pos = int((y == 1).sum())
        neg = int((y == 0).sum())
        print(f"[Diag] fold={fold} | val size={len(y)} | pos={pos} | neg={neg}")
        print(f"[Diag] fold={fold} | prob_pos mean={p.mean():.4f} | min={p.min():.4f} | max={p.max():.4f}")

        # NEW: print per-view uncertainty/strength/prob stats
        if args.print_view_diag:
            names = ["scale", "sc", "fc"]
            # concatenate
            vu = [np.concatenate(v) if len(v) else None for v in view_u_buf]
            vs = [np.concatenate(v) if len(v) else None for v in view_s_buf]
            vp = [np.concatenate(v) if len(v) else None for v in view_p_pos_buf]

            for i, nm in enumerate(names):
                if vu[i] is not None:
                    print(f"[ViewU] fold={fold} | {nm:<5} mean={_fmt(vu[i].mean())} p90={_fmt(np.quantile(vu[i],0.9))} "
                          f"min={_fmt(vu[i].min())} max={_fmt(vu[i].max())}")
                if vs[i] is not None:
                    print(f"[ViewS] fold={fold} | {nm:<5} mean={_fmt(vs[i].mean())} p10={_fmt(np.quantile(vs[i],0.1))} "
                          f"min={_fmt(vs[i].min())} max={_fmt(vs[i].max())}")
                if vp[i] is not None:
                    print(f"[ViewP] fold={fold} | {nm:<5} prob_pos mean={_fmt(vp[i].mean())} "
                          f"min={_fmt(vp[i].min())} max={_fmt(vp[i].max())}")

            print(f"[FusedU] fold={fold} | mean={_fmt(u.mean())} p90={_fmt(np.quantile(u,0.9))} "
                  f"min={_fmt(u.min())} max={_fmt(u.max())}")

        # metrics
        try:
            from sklearn.metrics import roc_auc_score, f1_score
            auc = roc_auc_score(y, p)
            pred05 = (p >= 0.5).astype(int)
            f1_05 = f1_score(y, pred05)

            # best F1
            best_f1, best_thr = 0.0, 0.5
            for thr in np.linspace(0.05, 0.95, 19):
                pred = (p >= thr).astype(int)
                f1 = f1_score(y, pred, zero_division=0)
                if f1 > best_f1:
                    best_f1, best_thr = f1, thr
        except Exception:
            auc, f1_05, best_f1, best_thr = float("nan"), float("nan"), float("nan"), float("nan")

        aucs.append(auc)
        f1s.append(f1_05)
        best_f1s.append(best_f1)

        print(f"[Eval] fold={fold} | AUC={auc:.4f} | F1@0.5={f1_05:.4f} | best_F1={best_f1:.4f} @thr={best_thr:.2f} "
              f"| mean_u={u.mean():.4f} | u(p90)={np.quantile(u,0.9):.4f} | ckpt={ckpt_path}")

    print(f"[Eval-Summary] folds={len(folds)} | AUC_mean={np.mean(aucs):.4f} | AUC_std={np.std(aucs):.4f}")
    print(f"[Eval-Summary] folds={len(folds)} | best_F1_mean={np.mean(best_f1s):.4f} | best_F1_std={np.std(best_f1s):.4f}")


if __name__ == "__main__":
    main()
