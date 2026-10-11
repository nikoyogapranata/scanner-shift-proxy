# results/

Committed to git. Written by the code, not by hand.

- `per_model/<model>.csv`: one row per model x patient x phase x structure x condition.
- `runs/<model>.json`: when the model was trained and evaluated, from which commit, with what scores.
- `analysis/`: output of `python -m ssp.analyze`.

Columns of the per-model tables:

| column | values |
|---|---|
| model_id | m01 ... |
| arm | clean, sim, real |
| vendor | A, B, C, D |
| corruption | none, bias, blur, gamma, resolution |
| severity | 0 for clean and real; 1 to S for sim |
| patient | patient code |
| phase | ED, ES |
| structure | LV, MYO, RV |
| dice | 0 to 1; empty if the structure is absent from the reference |
| hd95 | millimetres; empty unless evaluation ran with `--hd95` |
| pred_empty | True if the model predicted nothing for this structure |
