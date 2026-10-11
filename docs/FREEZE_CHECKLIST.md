# Freeze checklist

Subsection 6.6 promises that the metric definitions, the severity grid and the
analysis code are fixed in a dated commit before any model is evaluated on
vendors A, C and D. This is that step. Do it together.

Before freezing, all of these must be true:

- [ ] `configs/split.json` exists and is committed.
- [ ] `configs/models.yaml` is agreed. All models in it are trained.
- [ ] The clean arm has been run for every model (`python -m ssp.status`).
- [ ] `min_baseline_dice` in `configs/analysis.yaml` is set, from the clean results only.
- [ ] The severity values in `configs/severity_grid.yaml` are agreed, chosen on training patients only.
- [ ] The decision thresholds in `configs/analysis.yaml` match Subsection 6.6 (0.70 and 0.40).
- [ ] `python -m pytest -m "not slow"` passes.
- [ ] Which Tier 3 analyses will be attempted is written down here: ______________________

Then:

1. Set `status: frozen` in `configs/severity_grid.yaml`.
2. Commit and tag:

   ```bash
   git add -A
   git commit -m "Freeze: metrics, severity grid and analysis fixed before cross-vendor evaluation"
   git tag -a freeze-v1 -m "Pre-registration freeze"
   git push && git push --tags
   ```

3. Write the commit hash and date into the proposal (Section 6.6.8).

After the freeze, do not edit `configs/`, `src/ssp/analyze.py`, `src/ssp/metrics.py`
or `src/ssp/corruptions.py`. If a real bug forces a change, make it in a new
commit, tag it `freeze-v2`, and state the change and the reason in the report.
