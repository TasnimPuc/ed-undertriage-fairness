"""Builds notebooks/kaggle_run.ipynb from src/ so the Kaggle notebook is self-contained
and never drifts from the repo code.  Run:  python tools/build_notebook.py"""
from pathlib import Path

import sys

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nbsteps import cells as step_cells  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

INTRO = """# ED Under-triage Detector + Fairness Audit (MIMIC-IV-ED, Kaggle)

**Before Run All**
1. Settings → Accelerator **GPU T4** (XGBoost runs on GPU) · Internet **On** · keep this notebook **Private**.
2. Add-ons → **Google Cloud Services** → attach your linked account, tick **BigQuery** and **Cloud Storage**.
3. Fill `project_id` in the config cell. A private bucket `<project_id>-mimic-ckpt` is created automatically
   (public access blocked) unless you set `bucket`.

**Resume:** cohort, text featurizer, XGBoost (every 100 rounds + best-iteration/early-stopping state) and the
LogReg baseline are checkpointed to the bucket. If the session dies, just **Run All again with the same
`run_name`** — it continues from the last checkpoint. Patient-level files never go to `/kaggle/working`."""

INSTALL = "!pip install -q -U google-cloud-bigquery-storage db-dtypes 2>/dev/null | tail -1"

CONFIG = """import sys, os
sys.path.insert(0, "/kaggle/working/repo")
from src.config import Config

cfg = Config(
    project_id="YOUR_GCP_PROJECT_ID",   # <-- GCP project linked in Kaggle
    bucket="",                          # empty = auto-create private "<project_id>-mimic-ckpt"
    run_name="main",                    # keep the same name to resume; new name = fresh run
    target="critical",                  # critical | icu_12h | death | hospitalized
    low_acuity=(3, 4, 5),               # primary under-triage definition (strict ESI 4-5 also reported)
)  # full MIMIC-IV-ED cohort, full test set for SHAP"""

CHECK = """# pre-flight: both must say OK (otherwise follow the printed fix)
from src.gcp import check_access
check_access(cfg.project_id, ["mimiciv_ed.triage", "mimiciv_hosp.admissions", "mimiciv_icu.icustays"])"""


SHOW = """# Full results report (aggregate only -> safe for GitHub)
import re, shutil, glob
from IPython.display import Markdown, Image, display
rep = f"{cfg.work_dir}/report"
md = open(f"{rep}/REPORT.md", encoding="utf-8").read()
display(Markdown(re.sub(r"!\\[.*?\\]\\(figures/.*?\\)", "", md)))
for f in sorted(glob.glob(f"{rep}/figures/*.png")):
    display(Image(f))
shutil.make_archive("/kaggle/working/results_for_github", "zip", rep)
print("Download /kaggle/working/results_for_github.zip (Output tab) -> unzip into the repo as results/")"""


def main():
    nb = nbf.v4.new_notebook()
    cells = [nbf.v4.new_markdown_cell(INTRO), nbf.v4.new_markdown_cell("## 1 · Setup"), nbf.v4.new_code_cell(INSTALL),
             nbf.v4.new_markdown_cell("## 2 · Project code\nThese cells write the package to `/kaggle/working/repo`. "
                                      "No need to edit them; the steps further down call this code."),
             nbf.v4.new_code_cell("!mkdir -p /kaggle/working/repo/src")]
    for f in sorted((ROOT / "src").glob("*.py")):
        cells.append(nbf.v4.new_code_cell(f"%%writefile /kaggle/working/repo/src/{f.name}\n" + (f.read_text() or "# package\n")))
    cells += [nbf.v4.new_markdown_cell("## 3 · Config\nChange only this cell."), nbf.v4.new_code_cell(CONFIG)]
    cells += [nbf.v4.new_markdown_cell("### Pre-flight check\nEverything should say OK."), nbf.v4.new_code_cell(CHECK)]
    steps, n = step_cells(ROOT / "src" / "pipeline.py", 4)
    cells += steps + [nbf.v4.new_markdown_cell(f"## {n} · Results report\nAggregate results only (safe for GitHub)."),
                      nbf.v4.new_code_cell(SHOW)]
    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                      "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}}
    out = ROOT / "notebooks" / "kaggle_run.ipynb"
    nbf.write(nb, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
