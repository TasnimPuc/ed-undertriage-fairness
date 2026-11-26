"""Chief-complaint text features + tabular XGBoost (GPU) and baselines."""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

log = logging.getLogger(__name__)


def gpu_available() -> bool:
    if not shutil.which("nvidia-smi"):
        return False
    try:
        return subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False


class TextFeaturizer:
    """Word (1-2 gram) + char_wb (3-5 gram) TF-IDF on the triage chief complaint."""

    def __init__(self, n_word: int, n_char: int):
        self.word = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=n_word,
                                    sublinear_tf=True, dtype=np.float32, token_pattern=r"(?u)\b\w\w+\b")
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=10,
                                    max_features=n_char, sublinear_tf=True, dtype=np.float32)

    def fit(self, texts):
        self.word.fit(texts)
        self.char.fit(texts)
        return self

    def transform(self, texts):
        return sp.hstack([self.word.transform(texts), self.char.transform(texts)], format="csr")

    def names(self):
        return ([f"cc_w:{t}" for t in self.word.get_feature_names_out()] +
                [f"cc_c:{t}" for t in self.char.get_feature_names_out()])


def combine(X_tab, X_txt):
    return sp.hstack([sp.csr_matrix(X_tab.to_numpy(np.float32)), X_txt], format="csr")


class ResumableTraining:
    """Early stopping + periodic checkpoints whose state survives a dead session."""

    def __init__(self, store, key, local_dir, patience, every, state=None, crash_at=None):
        import xgboost as xgb

        outer = self

        class _CB(xgb.callback.TrainingCallback):
            def after_iteration(self, model, epoch, evals_log):
                return outer.step(model, evals_log)

        self.callback = _CB()
        self.store, self.key, self.dir = store, key, Path(local_dir)
        self.patience, self.every, self.crash_at = patience, every, crash_at
        self.state = state or {"best_score": -1.0, "best_iteration": -1, "rounds": 0, "finished": False}

    def step(self, model, evals_log):
        it = model.num_boosted_rounds() - 1                 # absolute iteration
        score = float(evals_log["val"]["aucpr"][-1])
        if score > self.state["best_score"]:
            self.state.update(best_score=score, best_iteration=it)
        self.state["rounds"] = it + 1
        stop = it - self.state["best_iteration"] >= self.patience
        if stop:
            self.state["finished"] = True
        elif (it + 1) % self.every == 0:
            self.save(model)
        if self.crash_at is not None and it + 1 >= self.crash_at:
            raise RuntimeError(f"simulated session death at round {it + 1}")
        return stop

    def save(self, model):
        self.dir.mkdir(parents=True, exist_ok=True)
        model.set_attr(best_iteration=str(self.state["best_iteration"]), best_score=str(self.state["best_score"]))
        model.save_model(str(self.dir / "xgb_ckpt.ubj"))
        (self.dir / "xgb_state.json").write_text(json.dumps(self.state))
        self.store.put(self.dir / "xgb_ckpt.ubj", f"{self.key}/xgb_ckpt.ubj")
        self.store.put(self.dir / "xgb_state.json", f"{self.key}/xgb_state.json")
        log.info("checkpoint: round %d (best %d, aucpr %.4f)", self.state["rounds"],
                 self.state["best_iteration"], self.state["best_score"])


def train_xgb(Xtr, ytr, Xva, yva, cfg, store, key):
    """Resumes from the last checkpoint in the store if one exists (same cfg.run_name)."""
    import xgboost as xgb
    device = "cuda" if gpu_available() else "cpu"
    params = dict(objective="binary:logistic", eval_metric=["auc", "aucpr"], tree_method="hist",
                  device=device, max_depth=7, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5,
                  min_child_weight=5, reg_lambda=1.0, max_bin=256, seed=cfg.seed)
    local = Path(cfg.private_dir) / "xgb"
    state, init = None, None
    if store.get(f"{key}/xgb_state.json", local / "xgb_state.json") and store.get(f"{key}/xgb_ckpt.ubj", local / "xgb_ckpt.ubj"):
        state = json.loads((local / "xgb_state.json").read_text())
        init = xgb.Booster(params={"device": device})
        init.load_model(str(local / "xgb_ckpt.ubj"))
        log.info("loaded checkpoint: %d boosting rounds already done (best round %d, val AUPRC %.4f)%s",
                 state["rounds"], state["best_iteration"] + 1, state["best_score"],
                 " -> training finished, nothing to repeat" if state["finished"] else " -> continuing")
    if state and (state["finished"] or state["rounds"] >= cfg.n_rounds):
        init.set_attr(best_iteration=str(state["best_iteration"]))
        return init
    log.info("XGBoost %s on %s | train=%s val=%s pos_rate=%.4f", xgb.__version__, device, Xtr.shape, Xva.shape, ytr.mean())
    trainer = ResumableTraining(store, key, local, cfg.early_stopping, cfg.ckpt_every, state, cfg.debug_crash_at_round)
    dtr = xgb.QuantileDMatrix(Xtr, ytr)
    dva = xgb.QuantileDMatrix(Xva, yva, ref=dtr)
    done = state["rounds"] if state else 0
    bst = xgb.train(params, dtr, num_boost_round=cfg.n_rounds - done, evals=[(dva, "val")],
                    xgb_model=init, callbacks=[trainer.callback], verbose_eval=250)
    trainer.state["finished"] = True
    trainer.save(bst)
    log.info("best_iteration=%d (of %d rounds)", trainer.state["best_iteration"], bst.num_boosted_rounds())
    return bst


def xgb_history(bst, Xva, yva, points: int = 12) -> pd.DataFrame:
    """Validation AUROC / AUPRC along the boosting rounds of the (saved or new) model; ★ = best round used."""
    import xgboost as xgb
    from sklearn.metrics import average_precision_score, roc_auc_score
    n = bst.num_boosted_rounds()
    best = int(bst.attr("best_iteration") or n - 1) + 1
    d = xgb.DMatrix(Xva)
    rows = []
    for k in sorted(set(np.linspace(1, n, points).astype(int)) | {best}):
        p = bst.predict(d, iteration_range=(0, int(k)))
        rows.append({"rounds": int(k), "val_auroc": roc_auc_score(yva, p), "val_auprc": average_precision_score(yva, p),
                     "best": "★" if k == best else ""})
    return pd.DataFrame(rows)


def predict_xgb(bst, X):
    import xgboost as xgb
    return bst.predict(xgb.DMatrix(X), iteration_range=(0, int(bst.attr("best_iteration")) + 1))


def shap_importance(bst, X, names, top=25):
    import xgboost as xgb
    contrib = bst.predict(xgb.DMatrix(X), pred_contribs=True,
                          iteration_range=(0, int(bst.attr("best_iteration")) + 1))[:, :-1]
    imp = np.abs(contrib).mean(axis=0)
    order = np.argsort(-imp)[:top]
    return [(names[i], float(imp[i])) for i in order]


def train_lr(Xtab_tr, Xtxt_tr, ytr, cfg):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    prep = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    Xtr = sp.hstack([sp.csr_matrix(prep.fit_transform(Xtab_tr)), Xtxt_tr], format="csr")
    lr = LogisticRegression(solver="saga", max_iter=200, tol=1e-3, C=1.0, random_state=cfg.seed)
    lr.fit(Xtr, ytr)
    return prep, lr


def predict_lr(prep, lr, Xtab, Xtxt):
    return lr.predict_proba(sp.hstack([sp.csr_matrix(prep.transform(Xtab)), Xtxt], format="csr"))[:, 1]


# ---------------- metrics ----------------
def threshold_for_recall(y, p, recall):
    pos = np.sort(p[y == 1])[::-1]
    if len(pos) == 0:
        return 0.5
    k = max(int(np.ceil(recall * len(pos))) - 1, 0)
    return float(pos[k])


def binary_metrics(y, p, thr):
    y = np.asarray(y).astype(int)
    f = (np.asarray(p) >= thr).astype(int)
    tp, fp = int(((f == 1) & (y == 1)).sum()), int(((f == 1) & (y == 0)).sum())
    fn, tn = int(((f == 0) & (y == 1)).sum()), int(((f == 0) & (y == 0)).sum())
    out = dict(n=int(len(y)), n_pos=int(y.sum()),
               sensitivity=tp / max(tp + fn, 1), specificity=tn / max(tn + fp, 1),
               ppv=tp / max(tp + fp, 1), npv=tn / max(tn + fn, 1), flag_rate=float(f.mean()))
    if 0 < y.sum() < len(y):
        out.update(auroc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)),
                   brier=float(brier_score_loss(y, np.clip(p, 0, 1))))
    return out


def bootstrap_auc(y, p, n_boot, seed):
    y, p = np.asarray(y), np.asarray(p)
    if y.sum() == 0 or y.sum() == len(y):
        return (np.nan, np.nan)
    rng, vals = np.random.default_rng(seed), []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            vals.append(roc_auc_score(y[i], p[i]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def ece(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)
