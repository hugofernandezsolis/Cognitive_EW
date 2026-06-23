# Accepted multi-seed full-profile anchor results

Provenance of the numbers used in the paper (`paper/main.tex`).

- **Source run:** `notebooks/run_anchors.py --profile full --anchors all --seeds 1,3,6,8,9`
  executed on Colab/GPU. Each seed runs every anchor independently in its own `seed_<n>/`
  subtree; `anchors_multiseed_report.json` is the aggregate (mean, sample std, SEM, 95% CI via
  Student's t, and per-seed pass rate per anchor, plus latency stats).
- **Why this copy exists:** the live `runs/` tree is gitignored, so the accepted JSON artifacts
  are staged here, under version control, as the single source of truth for the manuscript. Only
  JSON is copied — model checkpoints (`best.pt`) and the synthetic dataset (`synthetic.h5`) are
  intentionally excluded.
- **All data is synthetic.** Every emitter, waveform and scenario parameter comes from the
  repository's own published/synthetic catalogs. No classified or operational EW data is present.

## Layout

```
anchors_multiseed_report.json     # aggregate: per-anchor mean/std/sem/ci95/pass_rate + latency
seed_<n>/elint/metrics.json       # M2: macro accuracies, confusion matrices, latency
seed_<n>/elint/run_meta.json      # M2: seed, hyperparameters, data_config_hash, deps
seed_<n>/jamming/metrics.json     # M1: win_rate + latency
seed_<n>/marl/qmix/metrics.json   # M3: coordinated suppressed_fraction + latency
seed_<n>/marl/iql/metrics.json    # M3: independent baseline suppressed_fraction
seed_<n>/gan/metrics.json         # M4: validity, coverage, n_windows, n_types
seed_<n>/gan/robustness/metrics.json  # M4: held-out + global accuracy (baseline vs augmented)
```
Seeds: 1, 3, 6, 8, 9 (n = 5).

## Reproducibility key

- Dependencies: Python 3.11.15, Torch 2.6.0+cu124, NumPy 2.4.6 (see report).

## Honest reading of the ELINT anchor

In `anchors_multiseed_report.json` the `elint` anchor shows `achieved.mean = 0.0,
pass_rate = 0.0`. This is **not** an accuracy failure: across all five seeds the per-seed
`macro_acc_type` is `0.984 ± 0.005` and `macro_acc_mode/threat/lpi` are `≈ 1.0`. The composite
`strict_elint_score` is hard-gated to `0.0` because `latency_p99_ms = 2.06 ± 0.58 ms` exceeds the
`< 1 ms` target on every seed. The paper reports the accuracy targets as met and the p99 latency
target as not met (future work), exactly as the artifacts say.
