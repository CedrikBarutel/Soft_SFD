# AI Assisted 2 stage simulations

This folder starts a clean, staged simulation pipeline based on
`AI_Assisted/PLAN/CODEX_PLAN_diffusion_project(1)(2).md`.

First dry-run check:

```bash
python AI_assisted_2/run_stage_simulations.py --stage stage0 --dry-run
```

Equivalent custom example:

```bash
python AI_assisted_2/run_stage_simulations.py \
  --stage custom \
  --mode diffusion \
  --N-values 1,5 \
  --width-factors 0.5 \
  --runs 1 \
  --parallel 1 \
  --output-root results_validation_1d \
  --dry-run
```

The runner writes `manifest.csv` and `manifest.json` before launching jobs. The
manifest records the corrected collision time, runtime, dump frequency, seeds,
paths, and LAMMPS command metadata.

Create images from a successful manifest:

```bash
python AI_assisted_2/analyze_stage_results.py \
  --manifest AI_assisted_2/results_stage0/manifest.csv \
  --analyze-dir AI_assisted_2/analyze_stage0 \
  --figures-dir AI_assisted_2/figures_stage0
```
