"""Synthetic cohort with the same schema as the BigQuery query (no real patient data)."""
import numpy as np
import pandas as pd

CC = ["chest pain", "abd pain", "dyspnea", "fever", "headache", "fall", "back pain", "syncope",
      "nausea, vomiting", "laceration", "altered mental status", "weakness", "s/p fall, head injury"]


def make_cohort(n=6000, seed=0):
    r = np.random.default_rng(seed)
    acuity = r.choice([1, 2, 3, 4, 5], n, p=[.05, .3, .45, .17, .03])
    sev = (6 - acuity) + r.normal(0, 1, n)
    hr = 80 + 8 * sev + r.normal(0, 10, n)
    crit_p = 1 / (1 + np.exp(-(sev - 5.5)))
    icu = (r.random(n) < crit_p).astype(int)
    death = (r.random(n) < crit_p * 0.15).astype(int)
    subj = r.integers(10_000_000, 10_000_000 + n // 2, n)
    return pd.DataFrame({
        "subject_id": subj, "hadm_id": np.where(r.random(n) < .4, r.integers(2e7, 3e7, n), None),
        "stay_id": np.arange(n) + 30_000_000,
        "intime": pd.Timestamp("2150-01-01") + pd.to_timedelta(r.integers(0, 3e7, n), unit="s"),
        "outtime": pd.NaT, "gender": r.choice(["F", "M"], n), "race": r.choice(["WHITE", "BLACK/AFRICAN AMERICAN", "HISPANIC/LATINO - PUERTO RICAN", "ASIAN - CHINESE", "UNKNOWN"], n),
        "arrival_transport": r.choice(["AMBULANCE", "WALK IN", "OTHER", "UNKNOWN"], n),
        "disposition": "HOME", "temperature": r.normal(98.4, 1, n), "heartrate": hr,
        "resprate": r.normal(18, 3, n), "o2sat": np.clip(r.normal(97, 2, n), 70, 100),
        "sbp": r.normal(130, 20, n), "dbp": r.normal(75, 12, n),
        "pain": r.choice(["0", "5", "10", "unable", None], n), "acuity": acuity.astype(float),
        "chiefcomplaint": r.choice(CC, n), "age": r.integers(18, 95, n),
        "insurance": r.choice(["Medicare", "Medicaid", "Private", "Other"], n),
        "hospitalized": (r.random(n) < .4).astype(int), "icu_12h": icu, "death": death,
        "n_ed_30d": r.poisson(.2, n), "n_ed_365d": r.poisson(1, n), "n_hosp_30d": r.poisson(.1, n),
        "n_hosp_365d": r.poisson(.4, n), "n_icu_365d": r.poisson(.05, n)})
