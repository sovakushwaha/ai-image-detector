"""V2-7: Frozen CLIP embeddings + small MLP nonlinear probe (MLP-A / MLP-B).

No CLIP fine-tuning, LoRA, re-extraction, threshold tuning, calibration, or NTIRE.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from v2_final_test_contamination_guard_v1 import assert_path_not_final_external_test

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED = 42
CAP = 300
EXPECTED_SHA = "cba0cf3176fd8e61d828a102505edb991d5093c7979eb900f61031fe63acd7d0"
EXPECTED_N = 11377
LR = 1e-3
WD = 1e-4
BATCH = 64
MAX_EPOCHS = 100
PATIENCE = 12

OUT = PROJECT_ROOT / "results" / "v2"
FIG = PROJECT_ROOT / "figures" / "v2"
MODELS = PROJECT_ROOT / "models" / "v2"
NPZ = OUT / "v2_clip_embeddings_v1.npz"
MANIFEST = PROJECT_ROOT / "metadata" / "v2_clip_embedding_manifest_v1.csv"
REGISTRY = PROJECT_ROOT / "metadata" / "v2_generator_registry_v1.csv"

HARD_GENS = [
    "mllm::GPT_Image_2",
    "mllm::Nano_Banana_2",
    "qwen::FLUX.2_max",
    "qwen::GPT-Image-1.5",
    "qwen::Seedream-5.0",
]


def stop_if(cond: bool, msg: str) -> None:
    if cond:
        raise SystemExit(f"STOP: {msg}")


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def resolve_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MLPA(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MLPB(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


ARCHS = {
    "MLP-A": MLPA,
    "MLP-B": MLPB,
}


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def train_indices_for_fold(man: pd.DataFrame, fold: int) -> np.ndarray:
    col = f"fold_{fold}_role"
    ai = man[(man["binary_label"] == 1) & (man[col] == "TRAIN")].copy()
    keep_ai = []
    for _, gdf in ai.groupby("generator_id"):
        gdf = gdf.sort_values("image_id")
        if len(gdf) > CAP:
            gdf = gdf.iloc[:CAP]
        keep_ai.extend(gdf["embedding_row"].tolist())
    real = man[(man["binary_label"] == 0) & (man[col] == "TRAIN")]
    return np.array(sorted(set(keep_ai + real["embedding_row"].tolist())), dtype=int)


def val_indices_for_fold(man: pd.DataFrame, fold: int) -> np.ndarray:
    col = f"fold_{fold}_role"
    mask = man[col].isin(["HOLDOUT_VALIDATION", "REAL_VALIDATION"])
    return man.loc[mask, "embedding_row"].to_numpy(dtype=int)


def metrics_at_050(y_true: np.ndarray, p: np.ndarray) -> dict:
    y_pred = (p >= 0.5).astype(int)
    out = {
        "n": int(len(y_true)),
        "roc_auc": float("nan"),
        "ap": float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "ai_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float("nan"),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": float("nan"),
    }
    real = y_true == 0
    if real.any():
        out["specificity"] = float((y_pred[real] == 0).mean())
        out["fpr"] = float((y_pred[real] == 1).mean())
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, p))
        out["ap"] = float(average_precision_score(y_true, p))
    return out


def domain_specs(y: np.ndarray, p: np.ndarray, domains: np.ndarray) -> dict:
    out = {}
    for d in ["Tiny", "MLLM", "COCO", "Smartphone"]:
        m = (y == 0) & (domains == d)
        if m.any():
            pred = (p[m] >= 0.5).astype(int)
            out[d] = float((pred == 0).mean())
        else:
            out[d] = float("nan")
    vals = [out[d] for d in out if not np.isnan(out[d])]
    out["worst_domain"] = float(min(vals)) if vals else float("nan")
    return out


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, device: torch.device, batch: int = 512) -> np.ndarray:
    model.eval()
    probs = []
    xt = torch.from_numpy(X.astype(np.float32))
    for i in range(0, len(xt), batch):
        logits = model(xt[i : i + batch].to(device))
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs, axis=0)


def better_checkpoint(cand: dict, best: dict | None) -> bool:
    """True if cand should replace best (AUC primary; within 0.003 use Real guards)."""
    if best is None:
        return True
    if cand["roc_auc"] > best["roc_auc"] + 1e-12:
        return True
    if cand["roc_auc"] >= best["roc_auc"] - 0.003:
        # tie / practical equivalence band relative to current best
        if cand["roc_auc"] < best["roc_auc"] - 0.003:
            return False
        # both in band of each other around the higher AUC
        top = max(cand["roc_auc"], best["roc_auc"])
        if cand["roc_auc"] < top - 0.003:
            return False
        if best["roc_auc"] < top - 0.003:
            return True
        # both within 0.003 of top
        if cand["worst_domain_spec"] > best["worst_domain_spec"] + 1e-12:
            return True
        if abs(cand["worst_domain_spec"] - best["worst_domain_spec"]) < 1e-12:
            if cand["mllm_spec"] > best["mllm_spec"] + 1e-12:
                return True
    return False


def train_one_fold(
    arch_name: str,
    fold: int,
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    domains_va: np.ndarray,
    device: torch.device,
) -> tuple[nn.Module, dict, list[dict]]:
    set_seed(SEED + fold * 17 + (0 if arch_name == "MLP-A" else 101))
    model = ARCHS[arch_name]().to(device)
    n_real = int((ytr == 0).sum())
    n_ai = int((ytr == 1).sum())
    ratio = max(n_real, n_ai) / max(1, min(n_real, n_ai))
    if ratio > 1.25:
        # pos_weight = n_neg / n_pos for BCEWithLogits (weight on positive class)
        pos_weight = torch.tensor([n_real / max(1, n_ai)], dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        weight_note = f"pos_weight={float(pos_weight.item()):.4f}"
    else:
        criterion = nn.BCEWithLogitsLoss()
        weight_note = "None"

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    ds = TensorDataset(
        torch.from_numpy(Xtr.astype(np.float32)),
        torch.from_numpy(ytr.astype(np.float32)),
    )
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=False)

    history: list[dict] = []
    best: dict | None = None
    best_state = None
    patience_left = PATIENCE
    best_auc_strict = -1.0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        train_loss = float(np.mean(losses)) if losses else float("nan")

        # val
        model.eval()
        with torch.no_grad():
            vlogits = []
            for i in range(0, len(Xva), 512):
                vlogits.append(model(torch.from_numpy(Xva[i : i + 512].astype(np.float32)).to(device)))
            vlogit = torch.cat(vlogits, dim=0)
            val_loss = float(criterion(vlogit, torch.from_numpy(yva.astype(np.float32)).to(device)).item())
            pva = torch.sigmoid(vlogit).cpu().numpy()

        m = metrics_at_050(yva, pva)
        dspec = domain_specs(yva, pva, domains_va)
        row = {
            "architecture": arch_name,
            "fold": fold,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_roc_auc": m["roc_auc"],
            "val_ap": m["ap"],
            "ai_recall": m["ai_recall"],
            "real_specificity": m["specificity"],
            "mllm_specificity": dspec["MLLM"],
            "smartphone_specificity": dspec["Smartphone"],
            "tiny_specificity": dspec["Tiny"],
            "coco_specificity": dspec["COCO"],
            "worst_domain_specificity": dspec["worst_domain"],
        }
        history.append(row)

        cand = {
            "epoch": epoch,
            "roc_auc": m["roc_auc"],
            "ap": m["ap"],
            "worst_domain_spec": dspec["worst_domain"],
            "mllm_spec": dspec["MLLM"],
            "phone_spec": dspec["Smartphone"],
            "real_spec": m["specificity"],
            "val_loss": val_loss,
            "train_loss": train_loss,
        }
        if better_checkpoint(cand, best):
            best = cand
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # early stopping on strict AUC improvement
        if m["roc_auc"] > best_auc_strict + 1e-12:
            best_auc_strict = m["roc_auc"]
            patience_left = PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    assert best is not None and best_state is not None
    model.load_state_dict(best_state)
    model.to(device)
    meta = {
        "architecture": arch_name,
        "fold": fold,
        "selected_epoch": int(best["epoch"]),
        "best_val_auc": float(best["roc_auc"]),
        "n_epochs_ran": int(history[-1]["epoch"]),
        "class_ratio_max_min": float(ratio),
        "pos_weight": weight_note,
        "n_real_train": n_real,
        "n_ai_train": n_ai,
        "trainable_params": count_params(model),
        "final_train_loss": float(best["train_loss"]),
        "final_val_loss": float(best["val_loss"]),
        "overfit_gap_loss": float(best["val_loss"] - best["train_loss"]),
    }
    return model, meta, history


def paired_boot_auc_ap(y, p_a, p_b, n_boot=5000, seed=SEED):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    real_idx = np.where(y == 0)[0]
    ai_idx = np.where(y == 1)[0]
    out = {}
    for metric in ["roc_auc", "ap"]:
        diffs = []
        for _ in range(n_boot):
            ri = rng.choice(real_idx, size=len(real_idx), replace=True)
            ai = rng.choice(ai_idx, size=len(ai_idx), replace=True)
            idx = np.concatenate([ri, ai])
            yy = y[idx]
            if len(np.unique(yy)) < 2:
                continue
            if metric == "roc_auc":
                diffs.append(roc_auc_score(yy, p_a[idx]) - roc_auc_score(yy, p_b[idx]))
            else:
                diffs.append(average_precision_score(yy, p_a[idx]) - average_precision_score(yy, p_b[idx]))
        diffs = np.asarray(diffs, dtype=float)
        out[metric] = {
            "mean_diff": float(diffs.mean()),
            "ci_low": float(np.percentile(diffs, 2.5)),
            "ci_high": float(np.percentile(diffs, 97.5)),
            "n_valid_boot": int(len(diffs)),
        }
    return out


def paired_boot_spec(p_a, p_b, n_boot=5000, seed=SEED):
    rng = np.random.default_rng(seed)
    p_a = np.asarray(p_a)
    p_b = np.asarray(p_b)
    n = len(p_a)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sa = float((p_a[idx] < 0.5).mean())
        sb = float((p_b[idx] < 0.5).mean())
        diffs.append(sa - sb)
    diffs = np.asarray(diffs)
    return {
        "mean_diff": float(diffs.mean()),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "n_valid_boot": n_boot,
    }


def load_data():
    assert_path_not_final_external_test(str(NPZ), str(MANIFEST))
    data = np.load(NPZ, allow_pickle=False)
    emb = data["embeddings"].astype(np.float32)
    ids = data["image_ids"].astype(str)
    sha = hashlib.sha256(emb.tobytes()).hexdigest()
    stop_if(sha != EXPECTED_SHA, f"SHA mismatch {sha}")
    stop_if(emb.shape != (EXPECTED_N, 512), f"shape {emb.shape}")
    stop_if(bool(np.isnan(emb).any() or np.isinf(emb).any()), "NaN/Inf")
    man = pd.read_csv(MANIFEST)
    stop_if(list(man["image_id"].astype(str)) != list(ids), "id order")
    man["embedding_row"] = man["embedding_row"].astype(int)
    man["binary_label"] = man["binary_label"].astype(int)
    return man, emb, sha


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    man, emb, sha = load_data()
    device = resolve_device()
    print(f"device={device} sha={sha[:16]}...")

    # param counts
    param_counts = {name: count_params(cls()) for name, cls in ARCHS.items()}
    param_counts["LogReg_C3"] = 512 + 1  # coef + intercept
    print("params", param_counts)

    fold_idx = {f: {"train": train_indices_for_fold(man, f), "val": val_indices_for_fold(man, f)} for f in range(1, 5)}

    history_rows = []
    fold_metric_rows = []
    real_rows = []
    gen_rows = []
    preds: dict[str, dict[int, dict]] = {a: {} for a in ARCHS}
    train_meta_rows = []

    t0 = time.perf_counter()
    for arch_name in ARCHS:
        for fold in range(1, 5):
            print(f"=== {arch_name} fold {fold} ===")
            tr = fold_idx[fold]["train"]
            va = fold_idx[fold]["val"]
            Xtr, ytr = emb[tr], man.iloc[tr]["binary_label"].to_numpy()
            Xva, yva = emb[va], man.iloc[va]["binary_label"].to_numpy()
            domains_va = man.iloc[va]["real_domain"].astype(str).to_numpy()
            gens_va = man.iloc[va]["generator_id"].astype(str).to_numpy()
            ids_va = man.iloc[va]["image_id"].astype(str).to_numpy()

            model, meta, hist = train_one_fold(arch_name, fold, Xtr, ytr, Xva, yva, domains_va, device)
            history_rows.extend(hist)
            train_meta_rows.append(meta)

            # save checkpoint
            ckpt_path = MODELS / f"clip_{arch_name.replace('-', '').lower()}_fold{fold}_best_v1.pt"
            # names: clip_mlpA_fold1_best_v1.pt
            tag = "mlpA" if arch_name == "MLP-A" else "mlpB"
            ckpt_path = MODELS / f"clip_{tag}_fold{fold}_best_v1.pt"
            torch.save(
                {
                    "architecture": arch_name,
                    "fold": fold,
                    "state_dict": model.state_dict(),
                    "meta": meta,
                    "seed": SEED,
                },
                ckpt_path,
            )

            pva = predict_proba(model, Xva, device)
            m = metrics_at_050(yva, pva)
            dspec = domain_specs(yva, pva, domains_va)
            fold_metric_rows.append(
                {
                    "architecture": arch_name,
                    "fold": fold,
                    **m,
                    "tiny_specificity": dspec["Tiny"],
                    "mllm_specificity": dspec["MLLM"],
                    "coco_specificity": dspec["COCO"],
                    "smartphone_specificity": dspec["Smartphone"],
                    "selected_epoch": meta["selected_epoch"],
                    "n_epochs_ran": meta["n_epochs_ran"],
                    "train_loss_at_selected": meta["final_train_loss"],
                    "val_loss_at_selected": meta["final_val_loss"],
                    "overfit_gap_loss": meta["overfit_gap_loss"],
                    "pos_weight": meta["pos_weight"],
                }
            )
            for d in ["Tiny", "MLLM", "COCO", "Smartphone"]:
                real_rows.append(
                    {
                        "architecture": arch_name,
                        "fold": fold,
                        "real_domain": d,
                        "specificity": dspec[d],
                    }
                )

            hold_mask = yva == 1
            for gid in sorted(set(gens_va[hold_mask].tolist())):
                gmask = hold_mask & (gens_va == gid)
                pp = pva[gmask]
                pred = (pp >= 0.5).astype(int)
                gen_rows.append(
                    {
                        "architecture": arch_name,
                        "fold": fold,
                        "generator_id": gid,
                        "n": int(gmask.sum()),
                        "ai_recall_050": float(pred.mean()),
                        "mean_p_ai": float(pp.mean()),
                    }
                )

            preds[arch_name][fold] = {
                "y": yva,
                "p": pva,
                "domains": domains_va,
                "generators": gens_va,
                "ids": ids_va,
                "meta": meta,
            }
            print(
                f"  selected epoch={meta['selected_epoch']} AUC={m['roc_auc']:.4f} "
                f"phone={dspec['Smartphone']:.3f} MLLM={dspec['MLLM']:.3f}"
            )

    train_time = time.perf_counter() - t0

    hist_df = pd.DataFrame(history_rows)
    fold_df = pd.DataFrame(fold_metric_rows)
    real_df = pd.DataFrame(real_rows)
    gen_df = pd.DataFrame(gen_rows)
    hist_df.to_csv(OUT / "v2_clip_mlp_training_history_v1.csv", index=False)
    fold_df.to_csv(OUT / "v2_clip_mlp_fold_metrics_v1.csv", index=False)
    real_df.to_csv(OUT / "v2_clip_mlp_real_domain_metrics_v1.csv", index=False)
    gen_df.to_csv(OUT / "v2_clip_mlp_generator_metrics_v1.csv", index=False)

    # Architecture summaries + eligibility
    arch_rows = []
    for arch_name in ARCHS:
        sub = fold_df[fold_df["architecture"] == arch_name]
        row = {
            "architecture": arch_name,
            "trainable_params": param_counts[arch_name],
            "mean_roc_auc": float(sub["roc_auc"].mean()),
            "std_roc_auc": float(sub["roc_auc"].std(ddof=1)),
            "min_roc_auc": float(sub["roc_auc"].min()),
            "max_roc_auc": float(sub["roc_auc"].max()),
            "mean_ap": float(sub["ap"].mean()),
            "min_ap": float(sub["ap"].min()),
            "mean_ai_recall": float(sub["ai_recall"].mean()),
            "mean_real_specificity": float(sub["specificity"].mean()),
            "mean_mllm_specificity": float(sub["mllm_specificity"].mean()),
            "min_mllm_specificity": float(sub["mllm_specificity"].min()),
            "mean_smartphone_specificity": float(sub["smartphone_specificity"].mean()),
            "min_smartphone_specificity": float(sub["smartphone_specificity"].min()),
            "mean_selected_epoch": float(sub["selected_epoch"].mean()),
        }
        eligible = (
            row["mean_smartphone_specificity"] >= 0.95
            and row["min_smartphone_specificity"] >= 0.94
            and row["mean_real_specificity"] >= 0.89
            and row["mean_mllm_specificity"] >= 0.68
        )
        reasons = []
        if row["mean_smartphone_specificity"] < 0.95:
            reasons.append(f"mean phone {row['mean_smartphone_specificity']:.4f}<0.95")
        if row["min_smartphone_specificity"] < 0.94:
            reasons.append(f"worst phone {row['min_smartphone_specificity']:.4f}<0.94")
        if row["mean_real_specificity"] < 0.89:
            reasons.append(f"mean Real {row['mean_real_specificity']:.4f}<0.89")
        if row["mean_mllm_specificity"] < 0.68:
            reasons.append(f"mean MLLM {row['mean_mllm_specificity']:.4f}<0.68")
        row["eligible"] = bool(eligible)
        row["eligibility_reason"] = "PASS" if eligible else "; ".join(reasons)
        arch_rows.append(row)
    arch_df = pd.DataFrame(arch_rows)
    arch_df.to_csv(OUT / "v2_clip_mlp_architecture_summary_v1.csv", index=False)

    eligible_df = arch_df[arch_df["eligible"]].copy()
    if len(eligible_df) == 0:
        selected_arch = None
        selection_reason = "NO eligible MLP under reliability guards"
    else:
        best_auc = float(eligible_df["mean_roc_auc"].max())
        near = eligible_df[eligible_df["mean_roc_auc"] >= best_auc - 0.005].copy()
        bits = [f"primary max mean AUC among eligible (best={best_auc:.6f})"]
        if len(near) > 1:
            bits.append(f"{len(near)} arch within 0.005")
            mw = float(near["min_roc_auc"].max())
            near2 = near[near["min_roc_auc"] == mw]
            if len(near2) < len(near):
                bits.append(f"prefer higher worst-fold AUC ({mw:.6f})")
            near = near2
        if len(near) > 1:
            ma = float(near["mean_ap"].max())
            near2 = near[near["mean_ap"] == ma]
            if len(near2) < len(near):
                bits.append(f"prefer higher mean AP ({ma:.6f})")
            near = near2
        if len(near) > 1:
            mm = float(near["mean_mllm_specificity"].max())
            near2 = near[near["mean_mllm_specificity"] == mm]
            if len(near2) < len(near):
                bits.append(f"prefer higher MLLM spec ({mm:.6f})")
            near = near2
        if len(near) > 1:
            # smaller MLP: MLP-A < MLP-B
            near = near.sort_values("trainable_params")
            bits.append("prefer smaller MLP")
        selected_arch = str(near.iloc[0]["architecture"])
        selection_reason = "; ".join(bits) + f"; selected={selected_arch}"

    print("SELECTED", selected_arch, selection_reason)

    # LogReg C=3.0 predictions on same val samples
    logreg_preds = {}
    logreg_fold_rows = []
    logreg_gen_rows = []
    for fold in range(1, 5):
        bundle = joblib.load(MODELS / f"clip_logreg_refined_fold{fold}_v1.joblib")
        stop_if(float(bundle["C"]) != 3.0, f"expected C=3.0 fold{fold}")
        clf = bundle["model"]
        va = fold_idx[fold]["val"]
        Xva = emb[va]
        yva = man.iloc[va]["binary_label"].to_numpy()
        domains_va = man.iloc[va]["real_domain"].astype(str).to_numpy()
        gens_va = man.iloc[va]["generator_id"].astype(str).to_numpy()
        p = clf.predict_proba(Xva)[:, 1]
        m = metrics_at_050(yva, p)
        dspec = domain_specs(yva, p, domains_va)
        logreg_fold_rows.append(
            {
                "architecture": "LogReg_C3",
                "fold": fold,
                **m,
                "tiny_specificity": dspec["Tiny"],
                "mllm_specificity": dspec["MLLM"],
                "coco_specificity": dspec["COCO"],
                "smartphone_specificity": dspec["Smartphone"],
            }
        )
        hold = yva == 1
        for gid in sorted(set(gens_va[hold].tolist())):
            gmask = hold & (gens_va == gid)
            pp = p[gmask]
            logreg_gen_rows.append(
                {
                    "architecture": "LogReg_C3",
                    "fold": fold,
                    "generator_id": gid,
                    "n": int(gmask.sum()),
                    "ai_recall_050": float((pp >= 0.5).mean()),
                    "mean_p_ai": float(pp.mean()),
                }
            )
        logreg_preds[fold] = {"y": yva, "p": p, "domains": domains_va, "generators": gens_va}

    logreg_fold_df = pd.DataFrame(logreg_fold_rows)
    logreg_summary = {
        "mean_roc_auc": float(logreg_fold_df["roc_auc"].mean()),
        "min_roc_auc": float(logreg_fold_df["roc_auc"].min()),
        "mean_ap": float(logreg_fold_df["ap"].mean()),
        "min_ap": float(logreg_fold_df["ap"].min()),
        "mean_ai_recall": float(logreg_fold_df["ai_recall"].mean()),
        "mean_real_specificity": float(logreg_fold_df["specificity"].mean()),
        "mean_mllm_specificity": float(logreg_fold_df["mllm_specificity"].mean()),
        "mean_smartphone_specificity": float(logreg_fold_df["smartphone_specificity"].mean()),
        "fold4_auc": float(logreg_fold_df.loc[logreg_fold_df["fold"] == 4, "roc_auc"].iloc[0]),
        "fold4_ap": float(logreg_fold_df.loc[logreg_fold_df["fold"] == 4, "ap"].iloc[0]),
        "fold4_ai_recall": float(logreg_fold_df.loc[logreg_fold_df["fold"] == 4, "ai_recall"].iloc[0]),
        "fold4_real_specificity": float(logreg_fold_df.loc[logreg_fold_df["fold"] == 4, "specificity"].iloc[0]),
    }

    # Compare selected MLP vs LogReg
    boot_rows = []
    vs_rows = []
    if selected_arch is None:
        decision = "LINEAR_PROBE_PREFERRED"
        decision_evidence = "No MLP passed reliability guards; retain LogReg C=3.0."
        sel_summary = None
        remote_next = "YES"  # representation limit → LoRA next candidate path
    else:
        sel = arch_df[arch_df["architecture"] == selected_arch].iloc[0]
        sel_summary = sel.to_dict()
        # per-fold comparison table
        for fold in range(1, 5):
            mlp_row = fold_df[(fold_df["architecture"] == selected_arch) & (fold_df["fold"] == fold)].iloc[0]
            lr_row = logreg_fold_df[logreg_fold_df["fold"] == fold].iloc[0]
            vs_rows.append(
                {
                    "fold": fold,
                    "mlp_arch": selected_arch,
                    "mlp_auc": float(mlp_row["roc_auc"]),
                    "logreg_auc": float(lr_row["roc_auc"]),
                    "auc_diff": float(mlp_row["roc_auc"] - lr_row["roc_auc"]),
                    "mlp_ap": float(mlp_row["ap"]),
                    "logreg_ap": float(lr_row["ap"]),
                    "ap_diff": float(mlp_row["ap"] - lr_row["ap"]),
                    "mlp_ai_recall": float(mlp_row["ai_recall"]),
                    "logreg_ai_recall": float(lr_row["ai_recall"]),
                    "mlp_real_spec": float(mlp_row["specificity"]),
                    "logreg_real_spec": float(lr_row["specificity"]),
                    "mlp_mllm_spec": float(mlp_row["mllm_specificity"]),
                    "logreg_mllm_spec": float(lr_row["mllm_specificity"]),
                    "mlp_phone_spec": float(mlp_row["smartphone_specificity"]),
                    "logreg_phone_spec": float(lr_row["smartphone_specificity"]),
                }
            )
            # bootstrap
            pm = preds[selected_arch][fold]
            pl = logreg_preds[fold]
            ba = paired_boot_auc_ap(pm["y"], pm["p"], pl["p"], n_boot=5000, seed=SEED)
            for metric, d in ba.items():
                boot_rows.append({"fold": fold, "metric": f"{metric}_mlp_minus_logreg", **d})
            for name, mask in [
                ("overall_Real", pm["y"] == 0),
                ("MLLM_Real", (pm["y"] == 0) & (pm["domains"] == "MLLM")),
                ("Smartphone_Real", (pm["y"] == 0) & (pm["domains"] == "Smartphone")),
            ]:
                if mask.sum() == 0:
                    continue
                d = paired_boot_spec(pm["p"][mask], pl["p"][mask], n_boot=5000, seed=SEED)
                boot_rows.append({"fold": fold, "metric": f"specificity_{name}_mlp_minus_logreg", **d})

        vs_df = pd.DataFrame(vs_rows)
        vs_df.to_csv(OUT / "v2_clip_mlp_vs_logreg_v1.csv", index=False)
        pd.DataFrame(boot_rows).to_csv(OUT / "v2_clip_mlp_bootstrap_v1.csv", index=False)

        d_auc = float(sel["mean_roc_auc"] - logreg_summary["mean_roc_auc"])
        d_worst = float(sel["min_roc_auc"] - logreg_summary["min_roc_auc"])
        d_ap = float(sel["mean_ap"] - logreg_summary["mean_ap"])
        d_mllm = float(sel["mean_mllm_specificity"] - logreg_summary["mean_mllm_specificity"])
        d_phone = float(sel["mean_smartphone_specificity"] - logreg_summary["mean_smartphone_specificity"])

        # difficult generator recall comparison
        hard_deltas = []
        hard_table = []
        for gid in HARD_GENS:
            msub = gen_df[(gen_df["architecture"] == selected_arch) & (gen_df["generator_id"] == gid)]
            lsub = pd.DataFrame(logreg_gen_rows)
            lsub = lsub[lsub["generator_id"] == gid]
            mr = float(msub["ai_recall_050"].mean()) if len(msub) else float("nan")
            lr = float(lsub["ai_recall_050"].mean()) if len(lsub) else float("nan")
            hard_table.append({"generator_id": gid, "mlp_recall": mr, "logreg_recall": lr, "diff": mr - lr})
            if not np.isnan(mr) and not np.isnan(lr):
                hard_deltas.append(mr - lr)
        hard_mean_delta = float(np.mean(hard_deltas)) if hard_deltas else 0.0
        # focus on the three weakest
        weak_ids = ["qwen::FLUX.2_max", "qwen::GPT-Image-1.5", "qwen::Seedream-5.0"]
        weak_deltas = [h["diff"] for h in hard_table if h["generator_id"] in weak_ids and not np.isnan(h["diff"])]
        weak_mean_delta = float(np.mean(weak_deltas)) if weak_deltas else 0.0

        fold4_mlp = fold_df[(fold_df["architecture"] == selected_arch) & (fold_df["fold"] == 4)].iloc[0]

        # overfitting check
        overfit_flags = []
        for fold in range(1, 5):
            meta = preds[selected_arch][fold]["meta"]
            h = hist_df[(hist_df["architecture"] == selected_arch) & (hist_df["fold"] == fold)]
            gap = meta["overfit_gap_loss"]
            # late rise in val loss while train falls
            if gap > 0.15:
                overfit_flags.append(f"fold{fold} val-train loss gap={gap:.3f}")
            if len(h) >= 20:
                early_val = float(h.head(5)["val_loss"].mean())
                late_val = float(h.tail(5)["val_loss"].mean())
                early_tr = float(h.head(5)["train_loss"].mean())
                late_tr = float(h.tail(5)["train_loss"].mean())
                if late_tr + 0.05 < early_tr and late_val > early_val + 0.05:
                    overfit_flags.append(f"fold{fold} train↓ val↑ divergence")

        overfit_yes = len(overfit_flags) > 0

        # Decision
        clear_auc = d_auc >= 0.015 and d_worst >= 0.005
        clear_hard = weak_mean_delta >= 0.05
        modest = (d_auc >= 0.003 or d_ap >= 0.003 or d_mllm >= 0.02) and not clear_hard
        harmed = d_mllm < -0.02 or d_phone < -0.01 or (sel["mean_real_specificity"] < 0.89)

        if harmed or overfit_yes and d_auc < 0.01:
            decision = "LINEAR_PROBE_PREFERRED"
            decision_evidence = (
                f"MLP harms reliability or overfits without clear gain "
                f"(ΔAUC={d_auc:.4f}, ΔMLLM={d_mllm:.4f}, overfit={overfit_yes})."
            )
            remote_next = "YES"
        elif clear_auc or clear_hard:
            decision = "MLP_CLEARLY_BETTER"
            decision_evidence = (
                f"Meaningful gain vs LogReg C=3.0 (ΔAUC={d_auc:.4f}, Δworst={d_worst:.4f}, "
                f"Δweak_gen_recall={weak_mean_delta:.4f}, ΔMLLM={d_mllm:.4f})."
            )
            remote_next = "NO"
        elif modest or d_auc > 0:
            decision = "MLP_MODEST_GAIN"
            decision_evidence = (
                f"Small/consistent improvement only (ΔAUC={d_auc:.4f}, ΔAP={d_ap:.4f}, "
                f"Δweak_gen_recall={weak_mean_delta:.4f}); hard modern generators largely unsolved."
            )
            remote_next = "YES"
        else:
            decision = "LINEAR_PROBE_PREFERRED"
            decision_evidence = (
                f"No meaningful MLP gain (ΔAUC={d_auc:.4f}, Δweak_gen={weak_mean_delta:.4f})."
            )
            remote_next = "YES"

    # Figures
    fig, ax = plt.subplots(figsize=(8, 4))
    for arch_name, color in [("MLP-A", "#4c78a8"), ("MLP-B", "#e15759")]:
        for fold in range(1, 5):
            h = hist_df[(hist_df["architecture"] == arch_name) & (hist_df["fold"] == fold)]
            ax.plot(h["epoch"], h["val_roc_auc"], alpha=0.7, color=color, lw=1, label=f"{arch_name}" if fold == 1 else None)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation ROC-AUC")
    ax.set_ylim(0, 1)
    ax.set_title("MLP validation AUC by epoch (all folds)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "v2_clip_mlp_validation_auc_v1.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(arch_df))
    ax.bar(x - 0.15, arch_df["mean_roc_auc"], width=0.3, label="mean AUC")
    ax.bar(x + 0.15, arch_df["min_roc_auc"], width=0.3, label="worst AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(arch_df["architecture"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("MLP architecture comparison")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "v2_clip_mlp_architecture_comparison_v1.png", dpi=150)
    plt.close()

    if selected_arch is not None:
        fig, ax = plt.subplots(figsize=(7, 4))
        folds = [1, 2, 3, 4]
        mlp_aucs = [float(fold_df[(fold_df.architecture == selected_arch) & (fold_df.fold == f)]["roc_auc"].iloc[0]) for f in folds]
        lr_aucs = [float(logreg_fold_df[logreg_fold_df.fold == f]["roc_auc"].iloc[0]) for f in folds]
        x = np.arange(4)
        ax.bar(x - 0.15, lr_aucs, width=0.3, label="LogReg C=3.0")
        ax.bar(x + 0.15, mlp_aucs, width=0.3, label=selected_arch)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Fold {f}" for f in folds])
        ax.set_ylim(0, 1)
        ax.set_ylabel("ROC-AUC")
        ax.set_title("Same-fold held-out AUC: MLP vs LogReg C=3.0")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / "v2_mlp_vs_logreg_auc_v1.png", dpi=150)
        plt.close()

        fig, ax = plt.subplots(figsize=(7, 4))
        metrics_labels = ["Real", "MLLM", "Phone"]
        lr_vals = [
            logreg_summary["mean_real_specificity"],
            logreg_summary["mean_mllm_specificity"],
            logreg_summary["mean_smartphone_specificity"],
        ]
        mlp_vals = [
            float(sel_summary["mean_real_specificity"]),
            float(sel_summary["mean_mllm_specificity"]),
            float(sel_summary["mean_smartphone_specificity"]),
        ]
        x = np.arange(3)
        ax.bar(x - 0.15, lr_vals, width=0.3, label="LogReg C=3.0")
        ax.bar(x + 0.15, mlp_vals, width=0.3, label=selected_arch)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_labels)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Specificity @0.50")
        ax.set_title("Real-domain specificity: MLP vs LogReg")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / "v2_mlp_vs_logreg_real_specificity_v1.png", dpi=150)
        plt.close()

        fig, ax = plt.subplots(figsize=(8, 4))
        hard_df = pd.DataFrame(hard_table)
        ypos = np.arange(len(hard_df))
        ax.barh(ypos - 0.15, hard_df["logreg_recall"], height=0.3, label="LogReg C=3.0")
        ax.barh(ypos + 0.15, hard_df["mlp_recall"], height=0.3, label=selected_arch)
        ax.set_yticks(ypos)
        ax.set_yticklabels(hard_df["generator_id"], fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_xlabel("AI recall @0.50")
        ax.set_title("Difficult generator recall: MLP vs LogReg")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / "v2_mlp_difficult_generator_recall_v1.png", dpi=150)
        plt.close()

    # Config + report
    config = {
        "stage": "V2-7",
        "selected_architecture": selected_arch,
        "selection_reason": selection_reason,
        "decision": decision,
        "decision_evidence": decision_evidence,
        "REMOTE_GPU_NEEDED_FOR_NEXT_STAGE": remote_next,
        "param_counts": param_counts,
        "clip_encoder_params_frozen": 149_620_737,  # approx from V2-4; frozen
        "architectures": arch_df.to_dict(orient="records"),
        "logreg_c3_summary": logreg_summary,
        "selected_summary": sel_summary,
        "train_wall_seconds": train_time,
        "device": str(device),
        "optimizer": {"name": "AdamW", "lr": LR, "weight_decay": WD},
        "batch_size": BATCH,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "integrity": {
            "clip_weights_changed": False,
            "clip_embeddings_changed": False,
            "clip_fine_tuning": False,
            "mlp_trained": True,
            "architecture_search_beyond_AB": False,
            "lr_search": False,
            "threshold_selection": False,
            "calibration": False,
            "generator_folds_changed": False,
            "smartphone_split_changed": False,
            "ntire_accessed": False,
            "kaggle": False,
        },
        "train_meta": train_meta_rows,
    }
    if selected_arch is not None:
        config["hard_generators"] = hard_table
        config["fold4_mlp"] = {
            "auc": float(fold4_mlp["roc_auc"]),
            "ap": float(fold4_mlp["ap"]),
            "ai_recall": float(fold4_mlp["ai_recall"]),
            "real_specificity": float(fold4_mlp["specificity"]),
        }
        config["overfitting"] = {"observed": overfit_yes, "evidence": overfit_flags or ["no strong divergence flagged"]}
        config["deltas_vs_logreg"] = {
            "auc": d_auc,
            "worst_auc": d_worst,
            "ap": d_ap,
            "mllm_spec": d_mllm,
            "phone_spec": d_phone,
            "weak_gen_recall": weak_mean_delta,
        }

    (OUT / "v2_clip_mlp_selected_config_v1.json").write_text(json.dumps(config, indent=2) + "\n")

    lines = [
        "V2-7 Frozen CLIP + small MLP nonlinear probe",
        f"Decision: {decision}",
        f"Evidence: {decision_evidence}",
        f"Selected architecture: {selected_arch}",
        f"Selection reason: {selection_reason}",
        f"REMOTE_GPU_NEEDED_FOR_NEXT_STAGE: {remote_next}",
        "",
        "Architecture summary:",
        arch_df.to_string(index=False),
        "",
        "Param counts:",
        json.dumps(param_counts, indent=2),
        "",
        "LogReg C=3.0 summary:",
        json.dumps(logreg_summary, indent=2),
        "",
        "Fold metrics:",
        fold_df.to_string(index=False),
    ]
    if selected_arch is not None:
        lines += ["", "MLP vs LogReg:", pd.DataFrame(vs_rows).to_string(index=False)]
        lines += ["", "Hard generators:", pd.DataFrame(hard_table).to_string(index=False)]
        lines += ["", "Overfitting:", json.dumps(config["overfitting"], indent=2)]
    (OUT / "v2_clip_mlp_report_v1.txt").write_text("\n".join(lines) + "\n")

    print("DECISION", decision)
    print(arch_df.to_string(index=False))
    print("REMOTE_GPU_NEEDED_FOR_NEXT_STAGE", remote_next)


if __name__ == "__main__":
    main()
