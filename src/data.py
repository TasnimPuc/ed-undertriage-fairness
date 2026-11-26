"""Cohort extraction (BigQuery), cleaning and triage-time features."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import gcp

log = logging.getLogger(__name__)

COHORT_SQL = """
WITH ed AS (
  SELECT e.subject_id, e.hadm_id, e.stay_id, e.intime, e.outtime, e.gender, e.race,
         e.arrival_transport, e.disposition,
         t.temperature, t.heartrate, t.resprate, t.o2sat, t.sbp, t.dbp,
         CAST(t.pain AS STRING) AS pain, t.acuity, t.chiefcomplaint
  FROM `{ED}.edstays` e
  JOIN `{ED}.triage` t ON e.stay_id = t.stay_id AND e.subject_id = t.subject_id
),
ed_hist AS (
  SELECT stay_id,
    COUNT(*) OVER w30  AS n_ed_30d,
    COUNT(*) OVER w365 AS n_ed_365d
  FROM `{ED}.edstays`
  WINDOW
    w30  AS (PARTITION BY subject_id ORDER BY UNIX_SECONDS(TIMESTAMP(intime)) RANGE BETWEEN 2592000 PRECEDING AND 1 PRECEDING),
    w365 AS (PARTITION BY subject_id ORDER BY UNIX_SECONDS(TIMESTAMP(intime)) RANGE BETWEEN 31536000 PRECEDING AND 1 PRECEDING)
),
adm_hist AS (
  SELECT ed.stay_id,
    COUNTIF(a.admittime < ed.intime AND a.admittime >= DATETIME_SUB(ed.intime, INTERVAL 30 DAY))  AS n_hosp_30d,
    COUNTIF(a.admittime < ed.intime AND a.admittime >= DATETIME_SUB(ed.intime, INTERVAL 365 DAY)) AS n_hosp_365d,
    ARRAY_AGG(a.insurance IGNORE NULLS ORDER BY ABS(DATETIME_DIFF(a.admittime, ed.intime, HOUR)) LIMIT 1)[SAFE_OFFSET(0)] AS insurance_nearest
  FROM ed JOIN `{HOSP}.admissions` a ON ed.subject_id = a.subject_id
  GROUP BY ed.stay_id
),
icu_hist AS (
  SELECT ed.stay_id,
    COUNTIF(i.intime < ed.intime AND i.intime >= DATETIME_SUB(ed.intime, INTERVAL 365 DAY)) AS n_icu_365d,
    MAX(IF(i.hadm_id = ed.hadm_id AND i.intime <= DATETIME_ADD(ed.outtime, INTERVAL 12 HOUR), 1, 0)) AS icu_12h
  FROM ed JOIN `{ICU}.icustays` i ON ed.subject_id = i.subject_id
  GROUP BY ed.stay_id
)
SELECT ed.*,
  p.anchor_age + (EXTRACT(YEAR FROM ed.intime) - p.anchor_year) AS age,
  COALESCE(cur.insurance, ah.insurance_nearest, 'Unknown') AS insurance,
  IF(ed.hadm_id IS NOT NULL, 1, 0) AS hospitalized,
  COALESCE(ih.icu_12h, 0) AS icu_12h,
  IF(COALESCE(cur.hospital_expire_flag, 0) = 1 OR ed.disposition = 'EXPIRED', 1, 0) AS death,
  COALESCE(eh.n_ed_30d, 0) AS n_ed_30d, COALESCE(eh.n_ed_365d, 0) AS n_ed_365d,
  COALESCE(ah.n_hosp_30d, 0) AS n_hosp_30d, COALESCE(ah.n_hosp_365d, 0) AS n_hosp_365d,
  COALESCE(ih.n_icu_365d, 0) AS n_icu_365d
FROM ed
LEFT JOIN `{HOSP}.patients` p ON ed.subject_id = p.subject_id
LEFT JOIN `{HOSP}.admissions` cur ON ed.hadm_id = cur.hadm_id
LEFT JOIN ed_hist eh ON ed.stay_id = eh.stay_id
LEFT JOIN adm_hist ah ON ed.stay_id = ah.stay_id
LEFT JOIN icu_hist ih ON ed.stay_id = ih.stay_id
"""

VITALS = {"temperature": (90, 110), "heartrate": (10, 300), "resprate": (3, 80),
          "o2sat": (50, 100), "sbp": (30, 300), "dbp": (10, 250), "pain": (0, 10)}
ARRIVAL = ["AMBULANCE", "WALK IN", "HELICOPTER", "OTHER", "UNKNOWN"]
HIST = ["n_ed_30d", "n_ed_365d", "n_hosp_30d", "n_hosp_365d", "n_icu_365d"]


def load_cohort(cfg, store) -> pd.DataFrame:
    """Full MIMIC-IV-ED cohort. Cached once in the checkpoint store (no repeated BigQuery cost)."""
    cache = Path(cfg.private_dir) / "cohort.parquet"
    key = "cache/cohort.parquet"
    if cfg.use_cache and (cache.exists() or store.get(key, cache)):
        log.info("Cohort from cache (%s)", store.uri(key))
        return pd.read_parquet(cache)
    client = gcp.bq_client(cfg.project_id)
    ed = gcp.resolve_dataset(client, cfg.ed_dataset, cfg.ed_candidates, "triage")
    hosp = gcp.resolve_dataset(client, cfg.hosp_dataset, cfg.hosp_candidates, "admissions")
    icu = gcp.resolve_dataset(client, cfg.icu_dataset, cfg.icu_candidates, "icustays")
    df = gcp.query_df(client, COHORT_SQL.format(ED=ed, HOSP=hosp, ICU=icu))
    log.info("Cohort rows from BigQuery: %d", len(df))
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    store.put(cache, key)
    return df


def _race_group(r: str) -> str:
    r = (r or "").upper()
    if "HISPANIC" in r or "LATINO" in r or "SOUTH AMERICAN" in r:
        return "Hispanic"
    for key, name in (("WHITE", "White"), ("BLACK", "Black"), ("ASIAN", "Asian")):
        if r.startswith(key):
            return name
    if r in ("", "UNKNOWN", "UNABLE TO OBTAIN", "PATIENT DECLINED TO ANSWER"):
        return "Unknown"
    return "Other"


def clean(df: pd.DataFrame, cfg) -> pd.DataFrame:
    df = df.copy()
    for c in VITALS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    t = df["temperature"]
    df.loc[t.between(30, 45), "temperature"] = t * 9 / 5 + 32      # Celsius entries -> F
    for c, (lo, hi) in VITALS.items():
        df.loc[~df[c].between(lo, hi), c] = np.nan
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["acuity"] = pd.to_numeric(df["acuity"], errors="coerce")
    n0 = len(df)
    df = df[(df["age"] >= cfg.min_age) & df["acuity"].between(1, 5)].copy()
    log.info("Exclusions (age<%d / missing acuity): %d -> %d", cfg.min_age, n0, len(df))
    df["acuity"] = df["acuity"].astype(int)
    df["intime"] = pd.to_datetime(df["intime"])
    for c in ("icu_12h", "death", "hospitalized", *HIST):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    df["critical"] = ((df["icu_12h"] == 1) | (df["death"] == 1)).astype(int)
    y = df[cfg.target].astype(bool)
    df["low_acuity"] = df["acuity"].isin(cfg.low_acuity).astype(int)
    df["under_triage"] = (df["low_acuity"].astype(bool) & y).astype(int)
    df["low_acuity_s"] = df["acuity"].isin(cfg.low_acuity_sensitivity).astype(int)
    df["under_triage_s"] = (df["low_acuity_s"].astype(bool) & y).astype(int)
    df["age_group"] = pd.cut(df["age"], [0, 39, 64, 79, 200], labels=["18-39", "40-64", "65-79", "80+"]).astype(str)
    df["gender"] = df["gender"].fillna("Unknown").astype(str)
    df["insurance"] = df["insurance"].fillna("Unknown").astype(str)
    df["race_group"] = df["race"].map(_race_group)
    df["arrival_transport"] = df["arrival_transport"].fillna("UNKNOWN").str.upper()
    df["chiefcomplaint"] = (df["chiefcomplaint"].fillna("").str.lower()
                            .str.replace(r"[^a-z0-9,/ ]+", " ", regex=True).str.strip())
    return df.reset_index(drop=True)


def tabular_features(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Only information available at triage time (no disposition / LOS / labs)."""
    X = pd.DataFrame(index=df.index)
    X["age"] = df["age"]
    X["male"] = (df["gender"] == "M").astype(float)
    for c in VITALS:
        X[c] = df[c]
    X["shock_index"] = df["heartrate"] / df["sbp"]
    X["map"] = (df["sbp"] + 2 * df["dbp"]) / 3
    X["pulse_pressure"] = df["sbp"] - df["dbp"]
    X["n_missing_vitals"] = df[list(VITALS)].isna().sum(axis=1)
    for a in ARRIVAL:
        X["arrival_" + a.lower().replace(" ", "_")] = (df["arrival_transport"] == a).astype(float)
    for c in HIST:
        X[c] = df[c]
    X["hour"] = df["intime"].dt.hour
    X["dow"] = df["intime"].dt.dayofweek
    X["cc_len"] = df["chiefcomplaint"].str.len()
    X["cc_n_items"] = df["chiefcomplaint"].str.count(",") + (df["chiefcomplaint"].str.len() > 0)
    if cfg.use_acuity_feature:
        X["acuity"] = df["acuity"]
    return X.astype("float32")


def patient_split(df: pd.DataFrame, cfg) -> np.ndarray:
    """0=train 1=val 2=test, grouped by subject_id (no patient leakage)."""
    rng = np.random.default_rng(cfg.seed)
    subs = df["subject_id"].unique()
    u = pd.Series(rng.random(len(subs)), index=subs)
    r = df["subject_id"].map(u).to_numpy()
    tr, va, _ = cfg.split
    return np.where(r < tr, 0, np.where(r < tr + va, 1, 2))
