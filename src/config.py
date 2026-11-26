from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Config:
    # --- GCP (fill these two) ---
    project_id: str = ""                 # GCP billing project linked to Kaggle
    bucket: str = ""                     # empty = auto-create private "<project_id>-mimic-ckpt"
    bucket_location: str = "US"          # same multi-region as physionet-data
    gcs_prefix: str = "ed-undertriage"
    run_name: str = "main"               # checkpoints live under <prefix>/ckpt/<run_name>/ ; same name = resume
    local_store: str = ""                # tests only: use a local folder instead of GCS

    # --- BigQuery datasets (empty = auto-detect) ---
    ed_dataset: str = ""
    hosp_dataset: str = ""
    icu_dataset: str = ""
    ed_candidates: list = field(default_factory=lambda: ["mimiciv_ed", "mimiciv_v2_2_ed", "mimiciv_ed_v2_2"])
    hosp_candidates: list = field(default_factory=lambda: ["mimiciv_v3_1_hosp", "mimiciv_hosp", "mimiciv_3_1_hosp", "mimiciv_v2_2_hosp"])
    icu_candidates: list = field(default_factory=lambda: ["mimiciv_v3_1_icu", "mimiciv_icu", "mimiciv_3_1_icu", "mimiciv_v2_2_icu"])

    # --- paths ---
    work_dir: str = "/kaggle/working/ed_undertriage"   # public-safe outputs (model, aggregates, figures)
    private_dir: str = "/tmp/ed_undertriage_private"   # patient-level files (never saved as notebook output)

    # --- cohort / task ---
    target: str = "critical"             # critical (ICU<=12h or in-hospital death) | icu_12h | death | hospitalized
    low_acuity: tuple = (3, 4, 5)        # primary: ESI 3-5 (ESI 4-5 has <10 critical events per group in MIMIC-IV-ED)
    low_acuity_sensitivity: tuple = (4, 5)     # strict definition, reported as sensitivity analysis
    use_acuity_feature: bool = False     # False = independent "second opinion" model
    min_age: int = 18
    seed: int = 42
    split: tuple = (0.8, 0.1, 0.1)       # patient-level train/val/test

    # --- model ---
    tfidf_word_features: int = 3000
    tfidf_char_features: int = 2000
    n_rounds: int = 4000
    early_stopping: int = 150
    ckpt_every: int = 100                # boosting rounds between checkpoints
    target_recall: float = 0.80          # threshold chosen on validation
    run_lr_baseline: bool = True
    shap_rows: int | None = None         # None = whole test set

    # --- audit ---
    audit_attrs: tuple = ("age_group", "gender", "insurance", "race_group")
    n_boot: int = 200
    min_cell: int = 10                   # small-cell suppression for exported tables

    # --- run control ---
    use_cache: bool = True               # reuse cohort from checkpoint store instead of re-querying
    cleanup_private: bool = True         # delete local patient-level files at the end
    debug_crash_at_round: int | None = None   # tests: simulate a session dying mid-training

    def to_dict(self):
        return asdict(self)
