# ED Under-triage Detector + Fairness Audit (MIMIC-IV-ED)

Detects emergency-department patients who were triaged **non-urgent (ESI 3–5)** but went on to a
**critical outcome** (ICU within 12 h of ED departure or in-hospital death), and audits whether
under-triage differs by **age, sex, insurance and race**. A strict ESI 4–5 definition is reported as a
sensitivity analysis. In MIMIC-IV-ED it has fewer than 10 critical events per group, which is too few for an audit.

**Headline results (MIMIC-IV-ED, full cohort: 418,025 adult ED stays, 201,214 patients; test set 41,628 stays)**

| Model (no access to nurse ESI) | AUROC (95% CI) | AUPRC |
|---|---|---|
| XGBoost, triage tabular features + chief-complaint text | **0.909** (0.903–0.914) | **0.524** |
| Logistic regression, same features | 0.895 | 0.494 |
| ESI triage level alone | 0.809 (0.802–0.817) | 0.200 |

The full fairness audit is in [results/REPORT.md](results/REPORT.md).

**What's in it**
- **Leakage-safe features**: triage vitals, pain, arrival mode, derived vitals (shock index, MAP),
  prior ED/hospital/ICU use, time of day. Nothing recorded after triage goes in.
- **Chief complaint text**: word + character TF-IDF, stacked with the tabular features in one **GPU XGBoost**.
- The model ignores the nurse's ESI by default (`use_acuity_feature=False`), so it works as an
  independent second opinion. It is compared against **ESI-only** and **logistic regression** baselines.
- **Fairness audit**:
  - under-triage rate per group with Wilson 95% CI, plus χ² tests;
  - **adjusted odds ratios** for being triaged low acuity, controlling for presentation (vitals, ambulance arrival);
  - per-group model AUROC (bootstrap CI), TPR/FPR, PPV, ECE, and the share of under-triaged patients the model catches;
  - a disparity summary with max–min difference and ratio.
- Splits are by **patient**, and the outcome definition follows the MIMIC-IV-ED benchmark (Xie et al., 2022),
  so you can compare your results with theirs.
- **Privacy**: small cells (<10) are suppressed in every exported table. Patient-level files never
  reach `/kaggle/working` or git.

## Run on Kaggle (≈15–25 min, full cohort)
1. Import `notebooks/kaggle_run.ipynb` into Kaggle. Set GPU T4, Internet On, Private.
2. Add-ons → Google Cloud Services → BigQuery + Cloud Storage.
3. Set `project_id`, then **Run All**. The private bucket `<project_id>-mimic-ckpt` is created
   automatically (uniform access, public access prevention enforced). To use your own, set `bucket`.

### Resume after a crash
Everything expensive is checkpointed to `gs://<bucket>/ed-undertriage/ckpt/<run_name>/`:
the cohort (one BigQuery scan), the TF-IDF featurizer, XGBoost every `ckpt_every=100` rounds together
with its early-stopping state, and the LogReg baseline. If the session dies, **Run All again with the
same `run_name`**. Training continues from the last checkpointed round, so at most 100 rounds are redone.

BigQuery datasets are auto-detected (`physionet-data.mimiciv_ed`, `mimiciv_v3_1_hosp`, `mimiciv_v3_1_icu`, …).
To pin them, set `ed_dataset`, `hosp_dataset`, `icu_dataset` in `Config`.

## Outputs (`gs://<bucket>/ed-undertriage/runs/<run_name>/`)
| path | content | shareable? |
|---|---|---|
| `results/metrics.json` | AUROC/AUPRC/sens/spec, baselines, cohort stats | yes (aggregate) |
| `results/tables/*.csv` | fairness tables, odds ratios, SHAP top features | yes (suppressed) |
| `results/figures/*.png` | under-triage rates, per-group AUROC/TPR, ROC, SHAP | yes |
| `results/model/xgb_model.json`, `tfidf_featurizer.joblib` | trained model | **no, keep private** (trained on credentialed data) |
| `private/`, `cache/`, `ckpt/` | test predictions, cohort cache, checkpoints | **never** |

## Results
After a run, the notebook builds `report/REPORT.md` with figures. It holds aggregate numbers only,
with small cells suppressed, and is zipped as `results_for_github.zip`. Unzip it into this repo as
`results/` and commit it. The full report is in **[results/REPORT.md](results/REPORT.md)**.

## Local / CI
```bash
pip install -r requirements.txt
python -m pytest -q             # runs on synthetic data, no PhysioNet access needed
python tools/build_notebook.py  # regenerate the Kaggle notebook after editing src/
```

## Data use
This repo contains code only. Each user needs their own PhysioNet credentialed access to MIMIC-IV
and MIMIC-IV-ED. Do not commit data, patient-level outputs or trained weights.
