"""Builds a GitHub-ready results report (aggregate only, small cells suppressed).
Output: <work_dir>/report/REPORT.md + report/figures/*.png -> copy the folder into the repo as results/."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def md_table(df: pd.DataFrame, floatfmt="{:.3f}") -> str:
    def f(v):
        if isinstance(v, (float, np.floating)):
            return "–" if np.isnan(v) else floatfmt.format(v)
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def _ci(m, key="auroc_ci"):
    lo, hi = m.get(key, (np.nan, np.nan))
    return f"{m.get('auroc', np.nan):.3f} ({lo:.3f}–{hi:.3f})" if m.get("auroc") is not None else "–"


def _read(p):
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def build(work_dir: str | Path) -> str:
    out = Path(work_dir)
    rep = out / "report"
    (rep / "figures").mkdir(parents=True, exist_ok=True)
    for f in (out / "figures").glob("*.png"):
        shutil.copy(f, rep / "figures" / f.name)
    s = json.loads((out / "metrics.json").read_text())
    c, m, cfg = s["cohort"], s["metrics"], s["config"]
    t = out / "tables"

    perf = []
    for key, name in (("xgb", "XGBoost (tabular + chief complaint)"), ("logreg", "Logistic regression (tabular + text)"),
                      ("esi_only", "ESI only (flag ESI 1–2)"), ("esi_le3", "ESI only (flag ESI 1–3)")):
        if key in m:
            x = m[key]
            perf.append({"Model": name, "AUROC (95% CI)": _ci(x) if "auroc_ci" in x else f"{x.get('auroc', np.nan):.3f}",
                         "AUPRC": x.get("auprc", np.nan), "Sensitivity": x["sensitivity"],
                         "Specificity": x["specificity"], "PPV": x["ppv"], "Flag rate": x["flag_rate"]})
    within = []
    for key, name in (("within_low_acuity", f"ESI {tuple(cfg['low_acuity'])}"),
                      ("within_low_acuity_s", f"ESI {tuple(cfg['low_acuity_sensitivity'])}")):
        x = m.get(key, {})
        if x:
            small = x["n_pos"] < cfg["min_cell"]
            within.append({"Low-acuity group": name, "n": x["n"], "n with outcome": f"<{cfg['min_cell']}" if small else x["n_pos"],
                           "AUROC": np.nan if small else x.get("auroc", np.nan),
                           "Under-triaged caught (sens.)": np.nan if small else x["sensitivity"],
                           "Flag rate": x["flag_rate"]})

    ut = pd.read_csv(t / "undertriage_by_group.csv")
    ut_view = ut.assign(**{"Under-triage rate (95% CI)": [
        "–" if pd.isna(r.under_triage_rate) else f"{r.under_triage_rate:.3f} ({r.ut_lo:.3f}–{r.ut_hi:.3f})" for r in ut.itertuples()]})
    ut_view = ut_view[["attribute", "group", "n_stays", "n_outcome", "n_under_triaged", "Under-triage rate (95% CI)"]]

    mf = pd.read_csv(t / "model_fairness_test.csv")[["attribute", "group", "n", "auroc", "tpr", "fpr", "ppv", "ece", "under_triage_caught"]]
    disp = pd.read_csv(t / "disparity_summary.csv")
    chi, aor = _read(t / "chi2_tests.csv"), _read(t / "adjusted_odds_ratios.csv")
    if len(aor):
        aor = aor[~aor.term.str.endswith("_z") & (aor.term != "ambulance")].copy()
        aor["term"] = aor.term.str.replace(r"C\((\w+), Treatment\(reference='(.+?)'\)\)\[T\.(.+)\]", r"\1: \3 vs \2", regex=True)
    shap = pd.read_csv(t / "shap_top_features.csv").head(15)

    md = f"""# Results — ED Under-triage Detector + Fairness Audit

*MIMIC-IV-ED, full cohort. Aggregate results only; cells with fewer than {cfg['min_cell']} patients are suppressed.*

## Cohort
| | |
|---|---|
| ED stays (adults, triage acuity recorded) | {c['n_stays']:,} |
| Patients | {c['n_patients']:,} |
| Outcome (`{cfg['target']}`) rate | {c['outcome_rate']:.3%} |
| Split by patient (train / val / test) | {c['n_train']:,} / {c['n_val']:,} / {c['n_test']:,} |
| ESI distribution | {', '.join(f'ESI {k}: {v:,}' for k, v in c['acuity_dist'].items())} |

## Prediction of the critical outcome (test set)
Threshold chosen on validation for sensitivity ≈ {cfg['target_recall']:.0%}. The model does not see the nurse's ESI.

{md_table(pd.DataFrame(perf))}

### Catching under-triage: performance inside the low-acuity group
{md_table(pd.DataFrame(within)) if within else '–'}

![ROC](figures/roc_test.png)

## Fairness audit: who gets under-triaged?
Under-triage rate = share of patients **with** the critical outcome who were triaged ESI {tuple(cfg['low_acuity'])}.
A strict ESI {tuple(cfg['low_acuity_sensitivity'])} definition is in `tables/undertriage_by_group_esi_strict.csv` (most cells suppressed: too few events).

{md_table(ut_view)}

![Under-triage rate](figures/undertriage_rate.png)

### Statistical tests (χ², among patients with the outcome)
{md_table(chi) if len(chi) else '–'}

### Adjusted odds of being triaged low-acuity
Adjusted for triage vitals, pain and ambulance arrival, among patients with the outcome. OR > 1 means more likely to be under-triaged than the reference group.

{md_table(aor[['term', 'odds_ratio', 'or_lo', 'or_hi', 'p_value']]) if len(aor) else '–'}

## Model fairness by group (test set)
{md_table(mf)}

![AUROC by group](figures/model_auroc.png)
![Sensitivity by group](figures/model_tpr.png)

### Largest gaps
{md_table(disp[['attribute', 'metric', 'max_group', 'max', 'min_group', 'min', 'difference', 'ratio']])}

## What drives the model (mean |SHAP|, test set)
{md_table(shap)}

![SHAP](figures/shap_top.png)

---
Runtime of the last session: {s['runtime_min']} min.
"""
    (rep / "REPORT.md").write_text(md, encoding="utf-8")
    return md
