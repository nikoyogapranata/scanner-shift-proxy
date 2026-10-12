# scanner-shift-proxy

Code for the Track B project *Validity of Simulated Scanner Shift as a Proxy for
Real Cross-Vendor Failure in Medical Image Segmentation*.

The question: across a population of segmentation models trained on one scanner
vendor, does the Dice lost under **simulated** corruption rank the models the
same way as the Dice lost on **real** scans from other vendors?

The method is defined in `docs/Evaluation_Metrics_Subsection_6.6_v3.docx`. This
repository implements it.

## Layout

```
configs/
  models.yaml           the model population (12 models: seed x augmentation level)   DRAFT
  severity_grid.yaml    the simulated-shift levels; has a draft/frozen switch         DRAFT
  analysis.yaml         decision thresholds, bootstrap sizes, baseline floor
  data.yaml             target resolution, split sizes, training vendor
  split.json            the one fixed patient split (created by ssp.split, then committed)
src/ssp/
  data.py               find cases and load volumes from an M&Ms folder
  split.py              make the patient-level split, once
  preprocess.py         resample by pixel spacing, normalise intensity
  corruptions.py        the four simulated corruptions
  train.py              train one model on vendor B
  evaluate.py           score a model: clean, simulated, real
  metrics.py            Dice and HD95, with the empty-mask rules
  results.py            the results table and run records
  analyze.py            final metric F and critical metrics C1 to C7
  status.py             progress table
  compact.py            keep only the ED and ES frames: about 70 GB down to a few GB
  fetch.py              on Colab: build the compact copy once, keep it on Google Drive
  mock.py               small fake dataset for testing the pipeline
  synth.py              made-up results for testing the analysis
notebooks/
  00_colab_runner.ipynb   run training and evaluation on Colab, push results back
  01_dataset_audit.ipynb  check that a dataset copy has vendors and labels
  02_severity_grid.ipynb  look at the corruptions and choose the levels
results/
  per_model/            one CSV per model (committed)
  runs/                 one JSON record per model (committed)
  analysis/             report, summary, per-model table, scatter plot
docs/                   the methodology subsection, the freeze checklist
tests/                  23 tests; run with `python -m pytest`
```

Not in git: `data/` (datasets), `runs/` (model weights), `results_mock/`, `results_demo/`.

## 1. Set up on your computer

Needs Python 3.10 or newer and Git.

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest -m "not slow"
```

That installs what the analysis needs. All tests except the two slow ones should pass.

Then see the analysis work on invented numbers:

```bash
python -m ssp.synth --rho 0.8
python -m ssp.analyze --results-dir results_demo
```

Open `results_demo/analysis/report.md` and `scatter.png`.

**Optional, to run the whole pipeline locally.** Install PyTorch for your
machine (the command for your GPU is on pytorch.org), then:

```bash
pip install -e ".[train,dev]"
python -m ssp.mock --out data/mock
python -m ssp.split  --data-root data/mock
python -m ssp.train  --data-root data/mock --model m01 --epochs 5
python -m ssp.evaluate --data-root data/mock --model m01 --arm all
python -m ssp.status --mock
```

The mock dataset is drawn circles, not hearts. It only proves the plumbing
works. Its results go to `results_mock/` and can never mix with real ones.
Delete `configs/split.json` afterwards: the real split must be made on the real data.

## 2. Connect GitHub

```bash
git init
git add .
git commit -m "Project skeleton"
```

Create an empty **private** repository on GitHub named `scanner-shift-proxy`, then:

```bash
git branch -M main
git remote add origin https://github.com/<your-username>/scanner-shift-proxy.git
git push -u origin main
```

Add your teammate under Settings, Collaborators. She clones it and works from the same repository.

## 3. Connect Colab

1. On GitHub: Settings, Developer settings, Personal access tokens, Fine-grained tokens.
   Create one limited to this repository with **Contents: Read and write**. Each person makes their own.
2. In Colab: File, Open notebook, GitHub tab, pick `notebooks/00_colab_runner.ipynb` from this repository.
3. In Colab's left bar, open Secrets (the key icon), add `GH_TOKEN` with the token, and allow the notebook to read it.
4. Runtime, Change runtime type, GPU.
5. Edit the settings cell and run the cells in order.

The runner clones this repository, installs it, fetches the data from Kaggle
onto the Colab machine, keeps model weights on your Google Drive, and pushes
result files back to GitHub. A Kaggle Notebooks variant is at the end of the runner.

## Pointing the code at the data

Every command that reads data takes `--data-root <folder>`, or reads the
`MNMS_ROOT` environment variable. The folder can have any structure as long as
it contains, somewhere below it:

- a metadata CSV with a patient code, a vendor letter, and `ED` and `ES` columns;
- per patient, `<code>_sa.nii.gz` and `<code>_sa_gt.nii.gz`.

That is the layout of the official M&Ms release, and of the compact copy that
`ssp.compact` makes from it (only the ED and ES frames, so ED = 0 and ES = 1).
On Colab the runner builds the compact copy once and keeps it on Google Drive
as `ssp_data/mnms_compact.tar`. To give your teammate access, share the
`ssp_data` folder with her and have her add a shortcut to it in her own
"My Drive", so the same path works in her notebooks. **Run
`notebooks/01_dataset_audit.ipynb` on your copy first.** If the copy is laid
out differently, `discover_cases` in `src/ssp/data.py` is the one function to adapt.

## The workflow, in order

| Step | Command or file | Who | Notes |
|---|---|---|---|
| 1. Audit the data | `notebooks/01_dataset_audit.ipynb` | Niko | Vendors and labels present? |
| 2. Make the split | `python -m ssp.split` | one person, once | Commit `configs/split.json` |
| 3. Agree the model population | `configs/models.yaml` | both | Currently a draft |
| 4. Choose the severity grid | `notebooks/02_severity_grid.ipynb` | Niko | Training patients only |
| 5. Train the models | `python -m ssp.train --model m01` ... | teammate | On Colab or Kaggle |
| 6. Clean baseline | `python -m ssp.evaluate --model m01 --arm clean` | teammate | Allowed before the freeze |
| 7. Set the baseline floor | `configs/analysis.yaml` | both | From step 6, before step 9 |
| 8. **Freeze** | `docs/FREEZE_CHECKLIST.md` | both | Commit and tag |
| 9. Simulated and real arms | `--arm sim`, `--arm real` | teammate | Refused by the code before step 8 |
| 10. Analyse | `python -m ssp.analyze` | Niko | Writes `results/analysis/` |

## Seeing progress

```bash
git pull
python -m ssp.status
```

prints one line per model: whether it is trained, its validation Dice, its mean
Dice on each arm, and the commit it was trained from. `git log` shows who pushed
which results and when.

## Rules that keep the study valid

1. **The split is made once.** `ssp.split` refuses to overwrite it. Held-out vendor B patients are used for scoring only.
2. **Freeze before looking.** `ssp.evaluate` refuses the simulated and real arms on real data while the severity grid says `status: draft`.
3. **The grid is tuned on training patients only.** The grid notebook loads no one else.
4. **Nothing is dropped after its result is seen.** Tier 3 analyses (HD95, sensitivity) may be skipped for lack of time, and the report says so.
5. **Data and weights stay out of git.** `.gitignore` covers this.
6. **Commit before you run.** Each run record stores the commit hash. A hash ending in `+dirty` means uncommitted code was used.

## What is a draft and still needs a decision

- `configs/models.yaml`: 12 models as 4 augmentation levels x 3 seeds, and the number of epochs.
- `configs/severity_grid.yaml`: all 20 values are starting points.
- `configs/data.yaml`: 30 held-out and 15 validation patients, 1.25 mm target resolution.
- `configs/analysis.yaml`: the baseline floor is unset.

## What has and has not been tested

Tested: the analysis against hand-computed cases and synthetic results; the
split, metrics, preprocessing and results storage; and the whole pipeline
(mock data, split, training, three evaluation arms, the freeze guard) on a CPU.

Not tested: anything on the real M&Ms data. The data loader assumes the
official layout, and the training settings (epochs, learning rate, network
size) are sensible defaults that have never seen a real scan. Expect to adjust
them after the first real training run.

One implementation note: bias field, blur and gamma use TorchIO, as the
proposal states. Resolution reduction is implemented with SciPy, because
TorchIO's resampler did not degrade the image monotonically with the factor.
Section 6.3 of the proposal should say so.
