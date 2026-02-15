from __future__ import annotations

import os
import argparse
import warnings
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from models.tmc.tmc_model_minimal import TMCMinimalModel

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


def parse_args():
    p = argparse.ArgumentParser(description="Day4: Train TMC minimal with modality dropout for robustness.")

    p.add_argument("--cohort_csv", type=str, default=DEFAULT_COHORT_CSV)
    p.add_argument("--data_root", type=str, default="")
    p.add_argument("--folds_dir", type=str, default=DEFAULT_FOLDS_DIR)
    p.add_argument("--fold", type=int, default=-1, help="Train a single fold, or -1 for all folds.")

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)

    # save
    p.add_argument("--save_dir", type=str, default=r"F:\multimodal-conversion-project\results\week3_tmc_moddrop")
    p.add_argument("--ckpt_name", type=str, default="minimal_moddrop_ckpt.pt")

    # imbalance handling (use your Day3 best practice)
    p.add_argument("--use_class_weight", action="store_true", default=True)
    p.add_argument("--no_class_weight", action="store_false", dest="use_class_weight")
    p.add_argument("--pos_weight_cap", type=float, default=4.0)

    # evidence reg
    p.add_argument("--lambda_view", type=float, default=1.0)
    p.add_argument("--lambda_fused", type=float, default=1.0)
    p.add_argument("--lambda_evidence", type=float, default=1e-4)

    # modality dropout
    p.add_argument("--moddrop_p", type=float, default=0.2,
                   help="Probability to drop each modality independently per sample (training only).")

    return p.parse_args()


def seed_all(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataset(cohort_csv: str, data_root: str = ""):
    from pipeline.dataset import PatientT0Dataset
    if data_root:
        return PatientT0Dataset(cohort_csv=cohort_csv, data_root=data_root)
    return PatientT0Dataset(cohort_csv=cohort_csv)


def build_loader_from_subset(dataset, indices: np.ndarray, batch_size: int, shuffle: bool):
    subset = Subset(dataset, indices.tolist())
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False)


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
        if fn.startswith("fold_") and fn.endswith("_train_subjects.txt"):
            try:
                k = int(fn.split("_")[1])
                folds.add(k)
            except Exception:
                pass
    return sorted(list(folds))


def load_fold_indices_from_subjects(dataset, folds_dir: str, fold: int) -> Tuple[np.ndarray, np.ndarray]:
    train_path = os.path.join(folds_dir, f"fold_{fold}_train_subjects.txt")
    test_path = os.path.join(folds_dir, f"fold_{fold}_test_subjects.txt")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing fold subject files: {train_path} / {test_path}")

    train_subjects = set(_read_subjects_txt(train_path))
    test_subjects = set(_read_subjects_txt(test_path))

    df = dataset.df
    subj_series = df["subject_id"].astype(str).values
    train_idx = [i for i, sid in enumerate(subj_series) if sid in train_subjects]
    val_idx = [i for i, sid in enumerate(subj_series) if sid in test_subjects]

    if len(train_idx) == 0 or len(val_idx) == 0:
        raise RuntimeError(f"Fold {fold} produced empty indices.")
    return np.array(train_idx, dtype=np.int64), np.array(val_idx, dtype=np.int64)


@torch.no_grad()
def compute_class_weight_from_loader(loader: DataLoader, device: str, pos_weight_cap: float):
    pos = 0
    neg = 0
    for batch in loader:
        y = batch[DEFAULT_KEYS["y"]].long()
        pos += int((y == 1).sum().item())
        neg += int((y == 0).sum().item())
    raw = (float(neg) / float(pos)) if pos > 0 else 1.0
    used = float(min(raw, pos_weight_cap))
    w = torch.tensor([1.0, used], dtype=torch.float32, device=device)
    return w, raw, used, pos, neg


def weighted_nll_from_prob(prob: torch.Tensor, y: torch.Tensor, weight: Optional[torch.Tensor]) -> torch.Tensor:
    prob = prob.clamp_min(1e-12)
    return F.nll_loss(torch.log(prob), y, weight=weight, reduction="mean")


def evidence_reg_mean(view_evidences: List[torch.Tensor]) -> torch.Tensor:
    regs = []
    for e in view_evidences:
        regs.append(e.sum(dim=1).mean())
    return torch.stack(regs).mean() if len(regs) else torch.tensor(0.0)


def make_random_moddrop_masks(batch_size: int, p: float, device: str) -> List[torch.Tensor]:
    """
    Per-sample random mask for [scale, sc, fc].
    Each modality dropped independently with prob p.
    Ensure at least one modality remains per sample.
    """
    keep = (torch.rand(batch_size, 3, device=device) > p).long()
    all_zero = keep.sum(dim=1) == 0
    if all_zero.any():
        keep[all_zero] = torch.tensor([1, 1, 1], device=device, dtype=torch.long)
    return [keep[:, 0], keep[:, 1], keep[:, 2]]


def apply_masks_to_inputs(
    x_scale: torch.Tensor, x_sc: torch.Tensor, x_fc: torch.Tensor, masks: List[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Zero out inputs where mask==0.
    """
    m_scale, m_sc, m_fc = masks
    # broadcast: [B,1] for scale vector
    x_scale = x_scale * m_scale.unsqueeze(1).to(x_scale.dtype)
    # broadcast: [B,1,1] for matrices
    x_sc = x_sc * m_sc.view(-1, 1, 1).to(x_sc.dtype)
    x_fc = x_fc * m_fc.view(-1, 1, 1).to(x_fc.dtype)
    return x_scale, x_sc, x_fc


def train_one_fold(args, dataset, fold: int) -> str:
    train_idx, val_idx = load_fold_indices_from_subjects(dataset, args.folds_dir, fold)
    train_loader = build_loader_from_subset(dataset, train_idx, args.batch_size, shuffle=True)
    val_loader = build_loader_from_subset(dataset, val_idx, args.batch_size, shuffle=False)

    batch0 = next(iter(train_loader))
    scale_dim, sc_dim, fc_dim = infer_dims_from_batch(batch0)

    model = TMCMinimalModel(
        scale_dim=scale_dim,
        sc_dim=sc_dim,
        fc_dim=fc_dim,
        embed_dim=128,
        hidden=256,
        num_classes=2,
        prior=1.0,
        evidence_activation="softplus",
        dropout=0.1,
    ).to(args.device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    cls_weight = None
    if args.use_class_weight:
        cls_weight, raw, used, pos, neg = compute_class_weight_from_loader(
            train_loader, device=args.device, pos_weight_cap=args.pos_weight_cap
        )
        cap_note = "" if raw <= args.pos_weight_cap else " (capped)"
        print(f"[Day4-Train] Fold {fold} | train pos={pos} neg={neg} | raw_w_pos={raw:.3f} -> used_w_pos={used:.3f}{cap_note}")

    print(f"[Day4-Train] Fold {fold} | device={args.device} | moddrop_p={args.moddrop_p} | lambda_evidence={args.lambda_evidence:g}")

    model.train()
    for epoch in range(1, args.epochs + 1):
        running = {"loss": 0.0, "fused_ce": 0.0, "view_ce": 0.0, "ereg": 0.0}
        n = 0

        for batch in train_loader:
            x_scale = batch[DEFAULT_KEYS["x_scale"]].float().to(args.device)
            x_sc = batch[DEFAULT_KEYS["x_sc"]].float().to(args.device)
            x_fc = batch[DEFAULT_KEYS["x_fc"]].float().to(args.device)
            y = batch[DEFAULT_KEYS["y"]].long().to(args.device)

            masks = make_random_moddrop_masks(x_scale.size(0), args.moddrop_p, args.device)
            x_scale2, x_sc2, x_fc2 = apply_masks_to_inputs(x_scale, x_sc, x_fc, masks)

            out = model(x_scale2, x_sc2, x_fc2, modality_masks=masks)

            fused_ce = weighted_nll_from_prob(out.fused_prob, y, cls_weight)
            view_ces = [weighted_nll_from_prob(vp, y, cls_weight) for vp in out.view_probs]
            view_ce_mean = torch.stack(view_ces).mean() if len(view_ces) else torch.tensor(0.0, device=args.device)

            ereg = evidence_reg_mean(out.view_evidences).to(args.device)

            total = args.lambda_fused * fused_ce + args.lambda_view * view_ce_mean + args.lambda_evidence * ereg

            optim.zero_grad()
            total.backward()
            optim.step()

            running["loss"] += float(total.item())
            running["fused_ce"] += float(fused_ce.item())
            running["view_ce"] += float(view_ce_mean.item())
            running["ereg"] += float(ereg.item())
            n += 1

        for k in running:
            running[k] /= max(n, 1)

        # simple val fused CE (unweighted, just monitoring)
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                xs = batch[DEFAULT_KEYS["x_scale"]].float().to(args.device)
                xsc = batch[DEFAULT_KEYS["x_sc"]].float().to(args.device)
                xfc = batch[DEFAULT_KEYS["x_fc"]].float().to(args.device)
                yv = batch[DEFAULT_KEYS["y"]].long().to(args.device)
                outv = model(xs, xsc, xfc, modality_masks=None)
                prob = outv.fused_prob.clamp_min(1e-12)
                val_ce = F.nll_loss(torch.log(prob), yv, reduction="mean")
                val_losses.append(float(val_ce.item()))
        model.train()

        print(f"Fold {fold} | Epoch {epoch:02d} | loss={running['loss']:.4f} | fusedCE={running['fused_ce']:.4f} "
              f"| viewCE={running['view_ce']:.4f} | eReg={running['ereg']:.6f} | val_fusedCE={np.mean(val_losses):.4f}")

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, f"fold{fold}_{args.ckpt_name}")
    torch.save(model.state_dict(), ckpt_path)
    print(f"[Day4-Train] Fold {fold} | Saved checkpoint to {ckpt_path}")
    return ckpt_path


def main():
    args = parse_args()
    seed_all(args.seed)

    if not args.cohort_csv:
        args.cohort_csv = DEFAULT_COHORT_CSV
    if not os.path.exists(args.cohort_csv):
        raise FileNotFoundError(f"cohort_csv not found: {args.cohort_csv}")

    dataset = build_dataset(args.cohort_csv, args.data_root)
    available = list_folds(args.folds_dir)
    folds = [args.fold] if args.fold >= 0 else available
    if len(folds) == 0:
        raise FileNotFoundError(f"No folds found under folds_dir={args.folds_dir}")

    ckpts = []
    for f in folds:
        ckpts.append(train_one_fold(args, dataset, f))

    print("[Day4-Train] Finished. Checkpoints:")
    for c in ckpts:
        print("  -", c)


if __name__ == "__main__":
    main()
