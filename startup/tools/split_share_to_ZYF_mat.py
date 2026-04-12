# tools/split_share_to_ZYF_mat.py
import os
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio

MAT_PATH = r"F:\2025.12.13交大新项目\share_to_ZYF_v2\share_to_ZYF_v2\combined_aligned_v2.mat"
OUT_ROOT = Path(r"F:\2025.12.13交大新项目\share_to_ZYF_v2\derived_per_subject")  # 你也可以改成别的目录
OUT_ROOT.mkdir(parents=True, exist_ok=True)

def _to_str(x) -> str:
    """Robust conversion for MATLAB-loaded strings."""
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    if isinstance(x, np.ndarray):
        # e.g. array(['xxx'], dtype='<U3') or array([b'xxx'], dtype='|S3')
        if x.size == 0:
            return ""
        return _to_str(x.item())
    return str(x)

def uppervec_to_symm(v: np.ndarray, n_roi: int) -> np.ndarray:
    """Fill upper-tri (k=1) then mirror to get symmetric matrix."""
    m = np.zeros((n_roi, n_roi), dtype=np.float32)
    iu = np.triu_indices(n_roi, k=1)
    m[iu] = v.astype(np.float32)
    m = m + m.T
    return m

def infer_n_roi_from_vec_len(L: int) -> int:
    """
    L = n*(n-1)/2  -> n^2 - n - 2L = 0
    n = (1 + sqrt(1 + 8L)) / 2
    """
    n = (1.0 + (1.0 + 8.0 * L) ** 0.5) / 2.0
    n_int = int(round(n))
    if n_int * (n_int - 1) // 2 != L:
        raise ValueError(f"Vector length {L} cannot be n*(n-1)/2. inferred n={n}.")
    return n_int

def main():
    mat = sio.loadmat(MAT_PATH)

    # Required keys in your .mat
    IDs_raw = mat["ID"].squeeze()
    mind_vec = mat["MIND_Vector"]     # [N, L]
    fmri_vec = mat["fMRI_Vector"]     # [N, L]
    Q = mat["Questionnaire"]          # [N, D]
    Q_labels_raw = mat["Questionnaire_Label"].squeeze()

    IDs = [_to_str(x) for x in IDs_raw]
    Q_labels = [_to_str(x).strip() for x in Q_labels_raw]  # ✅ 去掉前后空格

    N = len(IDs)
    if mind_vec.shape[0] != N or fmri_vec.shape[0] != N or Q.shape[0] != N:
        raise RuntimeError("N mismatch among ID / MIND_Vector / fMRI_Vector / Questionnaire")

    L = mind_vec.shape[1]
    n_roi = infer_n_roi_from_vec_len(L)
    print(f"[INFO] N subjects = {N}, vec_len = {L}, inferred n_roi = {n_roi}, Q_dim = {Q.shape[1]}")

    # 将 scale 保存为单行 CSV，最大兼容你的 generate_cohort_table.py（一般会扫 csv/json/xlsx）
    scale_columns = Q_labels

    # （可选）如果你希望后续直接拿 label_convert：
    # 数据里常用 DATASET-DIAG2 作为 0/1 标签
    label_name = "DATASET-DIAG2"
    label_idx = scale_columns.index(label_name) if label_name in scale_columns else None

    rows = []
    for i, sid in enumerate(IDs):
        sid = sid.strip()
        if not sid:
            sid = f"sub_{i:04d}"

        subj_dir = OUT_ROOT / sid
        subj_dir.mkdir(parents=True, exist_ok=True)

        sc = uppervec_to_symm(mind_vec[i], n_roi=n_roi)
        fc = uppervec_to_symm(fmri_vec[i], n_roi=n_roi)

        sc_path = subj_dir / "sc.npy"
        fc_path = subj_dir / "fc.npy"
        np.save(sc_path, sc)
        np.save(fc_path, fc)

        # scale 单行 csv
        scale_path = subj_dir / "scale.csv"
        df_scale = pd.DataFrame([Q[i].tolist()], columns=scale_columns)
        df_scale.to_csv(scale_path, index=False, encoding="utf-8-sig")

        # 可选：把 label_convert 写到 cohort 表里（方便你后面不用日期也能跑 baseline）
        label_convert = ""
        if label_idx is not None:
            try:
                label_convert = int(df_scale.iloc[0, label_idx])
            except Exception:
                label_convert = ""

        rows.append({
            "subject_id": sid,
            "sc_path": str(sc_path),
            "fc_path": str(fc_path),
            "scale_path": str(scale_path),
            "label_convert": label_convert,  # 可为空
            # Week1 若你要走“按日期窗口自动打标签”，这几个日期字段后续再补
            "t0_date": "",
            "followup_end_date": "",
            "conversion_date": "",
            "dvars": "",
            "homogeneity": "",
        })

    # 同时写一份 full cohort_table（不是必须，但很实用）
    out_full = Path(r"F:\multimodal-conversion-project\startup\cohort_table\cohort_table_full_from_mat_v2.csv")
    out_full.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_full, index=False, encoding="utf-8-sig")
    print(f"[DONE] per-subject files written to: {OUT_ROOT}")
    print(f"[DONE] cohort_table_full_from_mat.csv written to: {out_full.resolve()}")

if __name__ == "__main__":
    main()
