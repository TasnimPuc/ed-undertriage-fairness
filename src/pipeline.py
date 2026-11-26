"""End-to-end run: cohort -> audit -> model -> fairness -> artifacts -> GCS."""
from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import data, fairness, model
from .store import make_store
from .config import Config

log = logging.getLogger("ed")


def _setup(cfg):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
    out, priv = Path(cfg.work_dir), Path(cfg.private_dir)
    for d in (out / "model", out / "tables", out / "figures", priv):
        d.mkdir(parents=True, exist_ok=True)
    return out, priv


def _roc_plot(curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for name, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend(fontsize=8); ax.set_title("Test ROC")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _shap_plot(imp, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names, vals = zip(*imp[::-1])
    fig, ax = plt.subplots(figsize=(6, 0.28 * len(names) + 1))
    ax.barh(names, vals, color="#55A868"); ax.set_xlabel("mean |SHAP|")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def run(cfg: Config, cohort: pd.DataFrame | None = None) -> dict:
    t0 = time.time()
    out, priv = _setup(cfg)
    np.random.seed(cfg.seed)
    tgt = cfg.target

    store = make_store(cfg)
    ck = f"ckpt/{cfg.run_name}"
    # %% Load the cohort | One BigQuery query (cached in the bucket), then cleaning and the outcome definition.
    # show: pd.Series({"stays": len(df), "patients": df.subject_id.nunique(), f"{tgt} rate": round(df[tgt].mean(), 4)})
    raw = cohort if cohort is not None else data.load_cohort(cfg, store)
    df = data.clean(raw, cfg)
    log.info("Cohort: %d stays, %d patients, %s rate=%.4f", len(df), df.subject_id.nunique(), tgt, df[tgt].mean())

    # %% Under-triage audit (full cohort) | Rates by group, chi² tests and adjusted odds ratios.
    # show: aor
    desc, tests = fairness.descriptive_audit(df, cfg.audit_attrs, tgt)
    desc_s, _ = fairness.descriptive_audit(df, cfg.audit_attrs, tgt, "under_triage_s", "low_acuity_s")
    aor = fairness.adjusted_odds_ratios(df, cfg.audit_attrs, tgt)
    aor_s = fairness.adjusted_odds_ratios(df, cfg.audit_attrs, tgt, "low_acuity_s")

    # %% Features + patient-level split | Vitals, history and TF-IDF on the chief complaint.
    split = data.patient_split(df, cfg)
    Xtab = data.tabular_features(df, cfg)
    y = df[tgt].to_numpy()
    tr, va, te = (split == 0), (split == 1), (split == 2)
    fz_local = priv / "tfidf_featurizer.joblib"
    if store.get(f"{ck}/tfidf_featurizer.joblib", fz_local):
        txt = joblib.load(fz_local)
        log.info("RESUME text featurizer from checkpoint")
    else:
        txt = model.TextFeaturizer(cfg.tfidf_word_features, cfg.tfidf_char_features).fit(df.loc[tr, "chiefcomplaint"])
        joblib.dump(txt, fz_local)
        store.put(fz_local, f"{ck}/tfidf_featurizer.joblib")
    Xtxt = txt.transform(df["chiefcomplaint"])
    X = model.combine(Xtab, Xtxt)
    names = list(Xtab.columns) + txt.names()
    log.info("Features: %d tabular + %d text", Xtab.shape[1], Xtxt.shape[1])

    # %% Train XGBoost (resumable) + baselines | Checkpoint every 100 rounds; compared with ESI alone and logistic regression. The first table shows validation scores along the boosting rounds of the loaded / trained model (★ = best round).
    # show: xgb_hist
    # show: pd.DataFrame({k: v for k, v in metrics.items() if isinstance(v, dict)}).T
    bst = model.train_xgb(X[tr], y[tr], X[va], y[va], cfg, store, ck)
    xgb_hist = model.xgb_history(bst, X[va], y[va])
    p_va, p_te = model.predict_xgb(bst, X[va]), model.predict_xgb(bst, X[te])
    thr = model.threshold_for_recall(y[va], p_va, cfg.target_recall)
    test = df[te].reset_index(drop=True)
    esi_score = (6 - test["acuity"]).to_numpy(float)

    metrics = {"xgb": model.binary_metrics(y[te], p_te, thr), "threshold": thr,
               "esi_only": model.binary_metrics(y[te], esi_score, 4.0),   # flag = ESI 1-2
               "esi_le3": model.binary_metrics(y[te], esi_score, 3.0)}    # flag = ESI 1-3
    metrics["xgb"]["auroc_ci"] = model.bootstrap_auc(y[te], p_te, cfg.n_boot, cfg.seed)
    metrics["esi_only"]["auroc_ci"] = model.bootstrap_auc(y[te], esi_score, cfg.n_boot, cfg.seed)
    curves = {"XGB (tab+text)": (y[te], p_te), "ESI only": (y[te], esi_score)}

    if cfg.run_lr_baseline:
        lr_local = priv / "logreg.joblib"
        if store.get(f"{ck}/logreg.joblib", lr_local):
            prep, lr = (lambda d: (d["prep"], d["lr"]))(joblib.load(lr_local))
            log.info("RESUME logistic regression from checkpoint")
        else:
            prep, lr = model.train_lr(Xtab[tr], Xtxt[tr], y[tr], cfg)
            joblib.dump({"prep": prep, "lr": lr}, lr_local)
            store.put(lr_local, f"{ck}/logreg.joblib")
        p_lr = model.predict_lr(prep, lr, Xtab[te], Xtxt[te])
        thr_lr = model.threshold_for_recall(y[va], model.predict_lr(prep, lr, Xtab[va], Xtxt[va]), cfg.target_recall)
        metrics["logreg"] = model.binary_metrics(y[te], p_lr, thr_lr)
        curves["LogReg (tab+text)"] = (y[te], p_lr)
        joblib.dump({"prep": prep, "lr": lr}, out / "model" / "logreg.joblib")

    # under-triage detection: inside the low-acuity group only
    for key, col in (("low_acuity", "low_acuity"), ("low_acuity_s", "low_acuity_s")):
        m = test[col].to_numpy() == 1
        metrics[f"within_{key}"] = model.binary_metrics(y[te][m], p_te[m], thr)

    # %% Model fairness on test | Cells with fewer than `min_cell` patients are suppressed before anything is shown or saved.
    # show: disp
    mf = fairness.model_fairness(test, p_te, thr, cfg.audit_attrs, tgt, cfg.n_boot, cfg.seed)
    cnt = ["n_stays", "n_outcome", "n_under_triaged", "n_low_acuity", "n", "n_pos"]
    desc_pub = fairness.suppress_small(desc, cnt, cfg.min_cell)
    mf_pub = fairness.suppress_small(mf, cnt, cfg.min_cell)
    disp = pd.concat([   # gaps computed only from non-suppressed cells
        fairness.disparity_summary(desc_pub, ["under_triage_rate", "outcome_rate_in_low_acuity"]),
        fairness.disparity_summary(mf_pub, ["auroc", "tpr", "fpr", "under_triage_caught", "ece"])])

    # %% SHAP feature importance
    # show: pd.DataFrame(imp, columns=["feature", "mean_abs_shap"]).head(15)
    rng = np.random.default_rng(cfg.seed)
    idx = np.where(te)[0]
    if cfg.shap_rows and cfg.shap_rows < len(idx):
        idx = rng.choice(idx, cfg.shap_rows, replace=False)
    imp = model.shap_importance(bst, X[idx], names)

    # %% Save tables, figures, model + report | Aggregate results only; patient-level predictions stay private.
    tdir, fdir, mdir = out / "tables", out / "figures", out / "model"
    desc_pub.to_csv(tdir / "undertriage_by_group.csv", index=False)
    fairness.suppress_small(desc_s, cnt, cfg.min_cell).to_csv(tdir / "undertriage_by_group_esi_strict.csv", index=False)
    mf_pub.to_csv(tdir / "model_fairness_test.csv", index=False)
    tests.to_csv(tdir / "chi2_tests.csv", index=False)
    aor.to_csv(tdir / "adjusted_odds_ratios.csv", index=False)
    aor_s.to_csv(tdir / "adjusted_odds_ratios_esi_strict.csv", index=False)
    disp.to_csv(tdir / "disparity_summary.csv", index=False)
    xgb_hist.to_csv(tdir / "xgb_validation_history.csv", index=False)
    pd.DataFrame(imp, columns=["feature", "mean_abs_shap"]).to_csv(tdir / "shap_top_features.csv", index=False)

    fairness.plot_groups(desc_pub, "under_triage_rate", "ut_lo", "ut_hi", f"Under-triage rate (ESI {cfg.low_acuity} | {tgt})", fdir / "undertriage_rate.png")
    fairness.plot_groups(mf_pub, "tpr", None, None, "Model sensitivity (test)", fdir / "model_tpr.png")
    fairness.plot_groups(mf_pub, "auroc", "auroc_lo", "auroc_hi", "Model AUROC (test)", fdir / "model_auroc.png")
    _roc_plot(curves, fdir / "roc_test.png")
    _shap_plot(imp, fdir / "shap_top.png")

    bst.save_model(str(mdir / "xgb_model.json"))
    joblib.dump(txt, mdir / "tfidf_featurizer.joblib")
    (mdir / "features.json").write_text(json.dumps({"tabular": list(Xtab.columns), "all": names, "threshold": thr,
                                                    "best_iteration": int(bst.attr("best_iteration"))}, indent=1))

    cohort_stats = dict(n_stays=len(df), n_patients=int(df.subject_id.nunique()),
                        outcome_rate=float(df[tgt].mean()),
                        acuity_dist={int(k): int(v) for k, v in df.acuity.value_counts().sort_index().items()},
                        n_train=int(tr.sum()), n_val=int(va.sum()), n_test=int(te.sum()))
    summary = dict(config=cfg.to_dict(), cohort=cohort_stats, metrics=metrics,
                   runtime_min=round((time.time() - t0) / 60, 2))
    (out / "metrics.json").write_text(json.dumps(summary, indent=1, default=float))

    # patient-level predictions -> private only
    (priv / "preds").mkdir(exist_ok=True)
    pd.DataFrame({"stay_id": test.stay_id, "y": y[te], "p": p_te, "acuity": test.acuity}).to_parquet(priv / "preds" / "test_predictions.parquet")

    from .report import build as build_report
    build_report(out)
    log.info("Report -> %s", out / "report" / "REPORT.md")
    store.put_dir(out, f"runs/{cfg.run_name}/results")
    store.put_dir(priv / "preds", f"runs/{cfg.run_name}/private") if (priv / "preds").exists() else None
    summary["outputs"] = store.uri(f"runs/{cfg.run_name}")
    if cfg.cleanup_private:
        shutil.rmtree(priv, ignore_errors=True)

    m = metrics["xgb"]
    log.info("DONE in %.1f min | XGB AUROC=%.3f AUPRC=%.3f | ESI AUROC=%.3f | within-low-acuity AUROC=%s",
             summary["runtime_min"], m.get("auroc", np.nan), m.get("auprc", np.nan),
             metrics["esi_only"].get("auroc", np.nan), metrics["within_low_acuity"].get("auroc"))
    return summary
