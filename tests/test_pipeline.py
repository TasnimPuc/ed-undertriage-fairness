import pytest
import json

import numpy as np

from src import fairness
from src.config import Config
from src.pipeline import run
from tests.synthetic import make_cohort


def test_wilson():
    p, lo, hi = fairness.wilson(10, 100)
    assert abs(p - 0.1) < 1e-9 and lo < p < hi


def test_suppression():
    import pandas as pd
    t = fairness.suppress_small(pd.DataFrame({"n": [3, 50]}), ["n"], 10)
    assert t.n.tolist() == ["<10", 50]


def _cfg(tmp_path, **kw):
    return Config(work_dir=str(tmp_path / "out"), private_dir=str(tmp_path / "priv"),
                  local_store=str(tmp_path / "bucket"), n_rounds=60, early_stopping=1000, ckpt_every=10,
                  n_boot=20, shap_rows=500, tfidf_word_features=50, tfidf_char_features=50, **kw)


@pytest.mark.parametrize("mode", ["run", "notebook"])
def test_end_to_end(tmp_path, mode):
    s = (run if mode == "run" else _steps)(_cfg(tmp_path), cohort=make_cohort())
    assert 0.5 < s["metrics"]["xgb"]["auroc"] <= 1
    assert (tmp_path / "out" / "model" / "xgb_model.json").exists()
    assert (tmp_path / "out" / "tables" / "undertriage_by_group.csv").exists()
    rep = (tmp_path / "out" / "report" / "REPORT.md").read_text()
    assert "Fairness audit" in rep and (tmp_path / "out" / "report" / "figures" / "roc_test.png").exists()
    json.loads((tmp_path / "out" / "metrics.json").read_text())
    # patient-level file must not be in the public folder
    assert not list((tmp_path / "out").rglob("*.parquet"))


def test_resume_after_crash(tmp_path):
    import pytest
    with pytest.raises(RuntimeError, match="simulated"):
        run(_cfg(tmp_path, debug_crash_at_round=35), cohort=make_cohort())
    st = json.loads((tmp_path / "bucket" / "ed-undertriage" / "ckpt" / "main" / "xgb_state.json").read_text())
    assert st["rounds"] == 30 and not st["finished"]          # last checkpoint before the crash
    s = run(_cfg(tmp_path), cohort=make_cohort())               # same run_name -> resumes
    st = json.loads((tmp_path / "bucket" / "ed-undertriage" / "ckpt" / "main" / "xgb_state.json").read_text())
    assert st["finished"] and st["rounds"] == 60
    assert (tmp_path / "bucket" / "ed-undertriage" / "runs" / "main" / "results" / "metrics.json").exists()


def _steps(cfg, **kw):
    """Runs the notebook's step cells (built from src/pipeline.py::run) instead of run()."""
    import sys
    from pathlib import Path as _P
    root = _P(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "tools"))
    from nbsteps import exec_steps
    return exec_steps(root / "src" / "pipeline.py", cfg, **kw)["result"]


def test_rerun_shows_saved_rounds(tmp_path):
    """New session, same bucket: XGBoost is loaded (not retrained) and the saved model's history is shown."""
    import pandas as pd
    run(_cfg(tmp_path), cohort=make_cohort())
    st = json.loads((tmp_path / "bucket" / "ed-undertriage" / "ckpt" / "main" / "xgb_state.json").read_text())
    s = _steps(_cfg(tmp_path), cohort=make_cohort())
    h = pd.read_csv(tmp_path / "out" / "tables" / "xgb_validation_history.csv")
    assert st["finished"] and h.rounds.max() == st["rounds"] and (h.best == "★").sum() == 1
    assert s["metrics"]["xgb"]["auroc"] > 0.5
