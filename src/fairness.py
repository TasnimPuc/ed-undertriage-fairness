"""Under-triage fairness audit: descriptive rates, adjusted odds ratios, model fairness."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from .model import binary_metrics, bootstrap_auc, ece

log = logging.getLogger(__name__)


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(c - h, 0.0), min(c + h, 1.0))


def descriptive_audit(df: pd.DataFrame, attrs, target: str, ut_col="under_triage", low_col="low_acuity"):
    """Under-triage rate = share of patients WITH the critical outcome who were triaged low acuity."""
    rows, tests = [], []
    for a in attrs:
        crit = df[df[target] == 1]
        for g, d in df.groupby(a, observed=True):
            dc = d[d[target] == 1]
            k, n = int(dc[ut_col].sum()), len(dc)
            r, lo, hi = wilson(k, n)
            dl = d[d[low_col] == 1]
            rl, rl_lo, rl_hi = wilson(int(dl[target].sum()), len(dl))
            rows.append(dict(attribute=a, group=g, n_stays=len(d), n_outcome=n, n_under_triaged=k,
                             outcome_rate=d[target].mean(), under_triage_rate=r, ut_lo=lo, ut_hi=hi,
                             n_low_acuity=len(dl), outcome_rate_in_low_acuity=rl,
                             low_lo=rl_lo, low_hi=rl_hi))
        ct = pd.crosstab(crit[a], crit[ut_col])
        if ct.shape[0] > 1 and ct.shape[1] == 2:
            chi2, p, dof, _ = chi2_contingency(ct)
            tests.append(dict(attribute=a, chi2=chi2, dof=dof, p_value=p))
    return pd.DataFrame(rows), pd.DataFrame(tests)


def adjusted_odds_ratios(df: pd.DataFrame, attrs, target: str, low_col="low_acuity", min_events=20):
    """Among patients with the outcome: logit(low acuity) ~ demographics + triage presentation."""
    import statsmodels.formula.api as smf
    d = df[df[target] == 1].copy()
    if d[low_col].sum() < 5 * min_events or (len(d) - d[low_col].sum()) < 5 * min_events:
        log.warning("Adjusted model skipped for %s: only %d events (too few)", low_col, int(d[low_col].sum()))
        return pd.DataFrame()
    covs = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "pain"]
    for c in covs:
        d[c + "_z"] = (d[c].fillna(d[c].median()) - d[c].mean()) / (d[c].std() + 1e-9)
    d["ambulance"] = (d["arrival_transport"] == "AMBULANCE").astype(int)
    terms = []
    for a in attrs:
        # levels with too few events/non-events make the design singular -> pool them
        ct = d.groupby(a)[low_col].agg(["sum", "count"])
        rare = ct.index[(ct["sum"] < min_events) | (ct["count"] - ct["sum"] < min_events)]
        d[a] = d[a].where(~d[a].isin(rare), "Other/small")
        if d[a].nunique() < 2:
            continue
        ref = d[a].value_counts().idxmax()
        terms.append(f"C({a}, Treatment(reference={ref!r}))")
    formula = f"{low_col} ~ " + " + ".join(terms + [c + "_z" for c in covs] + ["ambulance"])
    res = None
    for method in ("newton", "bfgs"):
        try:
            r = smf.logit(formula, data=d).fit(disp=0, maxiter=300, method=method)
            if r.mle_retvals.get("converged", True) and np.isfinite(r.bse).all():
                res = r
                break
        except Exception as e:
            log.warning("Adjusted logit (%s) failed: %s", method, e)
    if res is None:
        return pd.DataFrame()
    ci = res.conf_int()
    out = pd.DataFrame({"term": res.params.index, "odds_ratio": np.exp(res.params.values),
                        "or_lo": np.exp(ci[0].values), "or_hi": np.exp(ci[1].values),
                        "p_value": res.pvalues.values})
    return out[out.term != "Intercept"].reset_index(drop=True)


def model_fairness(test: pd.DataFrame, p, thr, attrs, target, n_boot, seed, ut_col="under_triage"):
    rows = []
    test = test.assign(_p=p)
    for a in attrs:
        for g, d in test.groupby(a, observed=True):
            y, pp = d[target].to_numpy(), d["_p"].to_numpy()
            m = binary_metrics(y, pp, thr)
            lo, hi = bootstrap_auc(y, pp, n_boot, seed)
            ut = d[d[ut_col] == 1]
            rows.append(dict(attribute=a, group=g, n=m["n"], n_pos=m["n_pos"],
                             auroc=m.get("auroc", np.nan), auroc_lo=lo, auroc_hi=hi,
                             auprc=m.get("auprc", np.nan), tpr=m["sensitivity"],
                             fpr=1 - m["specificity"], ppv=m["ppv"], flag_rate=m["flag_rate"],
                             ece=ece(y, pp) if len(y) >= 50 else np.nan,
                             n_under_triaged=len(ut),
                             under_triage_caught=float((ut["_p"] >= thr).mean()) if len(ut) else np.nan))
    return pd.DataFrame(rows)


def disparity_summary(table: pd.DataFrame, metrics):
    rows = []
    for a, d in table.groupby("attribute"):
        for m in metrics:
            v = d[m].dropna()
            if len(v) >= 2:
                rows.append(dict(attribute=a, metric=m, max=v.max(), min=v.min(), difference=v.max() - v.min(),
                                 ratio=(v.max() / v.min()) if v.min() > 0 else np.nan,
                                 max_group=d.loc[v.idxmax(), "group"], min_group=d.loc[v.idxmin(), "group"]))
    return pd.DataFrame(rows)


# a rate is hidden when its numerator OR denominator is a small cell (otherwise count = rate x n leaks it)
DEPENDENT = {"n_outcome": ["under_triage_rate", "ut_lo", "ut_hi", "outcome_rate"],
             "n_low_acuity": ["outcome_rate_in_low_acuity", "low_lo", "low_hi"],
             "n_under_triaged": ["under_triage_rate", "ut_lo", "ut_hi", "outcome_rate_in_low_acuity",
                                 "low_lo", "low_hi", "under_triage_caught"],
             "n_pos": ["auroc", "auroc_lo", "auroc_hi", "auprc", "tpr", "ppv", "ece", "under_triage_caught"],
             "n_stays": ["outcome_rate"], "n": ["fpr", "flag_rate"]}


def suppress_small(table: pd.DataFrame, count_cols, min_cell: int):
    """PhysioNet-style small-cell suppression: hide small counts and rates computed from them."""
    t = table.copy()
    for c, deps in DEPENDENT.items():
        if c in t:
            small = pd.to_numeric(t[c], errors="coerce") < min_cell
            for d in deps:
                if d in t:
                    t.loc[small, d] = np.nan
    for c in count_cols:
        if c in t:
            small = t[c] < min_cell
            t[c] = t[c].astype(object)
            t.loc[small, c] = f"<{min_cell}"
    return t


def plot_groups(table, metric, lo, hi, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    attrs = list(table["attribute"].unique())
    fig, axes = plt.subplots(1, len(attrs), figsize=(4.2 * len(attrs), 3.6), squeeze=False)
    for ax, a in zip(axes[0], attrs):
        d = table[table.attribute == a]
        v = d[metric].to_numpy(float)
        err = None
        if lo and hi:
            err = np.vstack([v - d[lo].to_numpy(float), d[hi].to_numpy(float) - v])
            err = np.nan_to_num(np.clip(err, 0, None))
        ax.bar(d["group"].astype(str), v, yerr=err, capsize=3, color="#4C72B0")
        ax.set_title(a)
        ax.tick_params(axis="x", rotation=35)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
