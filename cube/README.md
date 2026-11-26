# Cube manipulation

Evaluate ExAbs, single-level LeWM, and the online and staged Hi-LeWM planners on the same cube-manipulation tasks. The release includes pretrained models, initial simulator states, and goal observations for 100 tasks.

## Setup

Use a separate Python environment for Cube. From the repository root:

```bash
pip install -r cube/requirements.txt
```

For headless rendering, install `libosmesa6` on Linux and use `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa`. An EGL-capable system can instead set both variables to `egl`.

## Evaluate

Each command below evaluates 50 tasks at distance 50:

```bash
python -m cube.evaluate --method exabs --device cuda --distance 50
python -m cube.evaluate --method lewm --device cuda --distance 50
python -m cube.evaluate --method hwm_online --device cuda --distance 50
python -m cube.evaluate --method hwm_staged --device cuda --distance 50
```

Use `--distance 75` for the other 50 tasks. Distance measures the number of environment steps between the source and goal observations; the interaction budget is twice this distance. The released seed is `260933` and initializes both the task list and planner random generators.

The constrained Hi-LeWM variants use an empirical macro-action bank:

```bash
python -m cube.evaluate --method hwm_online_constrained --device cuda --distance 50
python -m cube.evaluate --method hwm_staged_constrained --device cuda --distance 50
```

| Method | Planning procedure |
| --- | --- |
| `exabs` | Closed-loop hierarchical planning with AM and PETAL |
| `lewm` | Single-level CEM over primitive action blocks |
| `hwm_online` | Hi-LeWM with online high-level replanning |
| `hwm_staged` | Hi-LeWM with one high-level plan and sequential fixed waypoint tracking |
| `hwm_online_constrained` | Online Hi-LeWM with empirical macro-action search |
| `hwm_staged_constrained` | Staged Hi-LeWM with empirical macro-action search |

Results contain per-task outcomes and aggregate success rate, and are saved under `outputs/cube/` with the method name in the filename. `--limit N` runs the first N tasks; `--output PATH` selects a result file.

Check installation and the complete planning/simulation pipeline with a small candidate population and two interaction steps:

```bash
python -m cube.evaluate --method exabs --device cpu --smoke-test
python -m cube.evaluate --method lewm --device cpu --smoke-test
python -m cube.evaluate --method hwm_online --device cpu --smoke-test
python -m cube.evaluate --method hwm_staged --device cpu --smoke-test
```

`--smoke-test` is also available for the constrained variants. CUDA is recommended for the full planning budgets.

## Model files

| File | Role |
| --- | --- |
| `checkpoints/l1.pt` | Shared visual encoder, low-level world model, action encoder, and projection layers |
| `checkpoints/ExAbs_cube.pt` | Complete ExAbs: L2, macro-action encoder, AM, and PETAL |
| `checkpoints/hwm.pt` | Hi-LeWM high-level world model and macro-action encoder, shared by all Hi-LeWM variants |

Online and staged Hi-LeWM use the same trained weights with different planners. Single-level LeWM uses the shared L1 tensors and its original single-level rollout implementation. All checkpoints load with `weights_only=True`.

Evaluation inputs and normalization are in `assets/`; automatically loaded duration estimates, baseline sampling bounds, and empirical macro-action banks are in `assets/runtime/`. Keep these directories with the checkpoints when moving the package.

## ExAbs planning flow

1. Encode the current observation and goal image.
2. Use high-level CEM to propose macro actions. AM scores the predicted waypoint using the available interaction budget and estimated duration distribution.
3. Use low-level CEM to predict candidate endpoints. PETAL compares their effects with the duration-weighted target effect and contributes a bounded residual to the endpoint objective.
4. Execute the first five primitive actions and replan.

Deployment settings are in `configs/distance_50.json` and `configs/distance_75.json`. `configs/baselines.json` defines the baseline-specific planner settings. Network definitions are in `models/`; cost evaluation and controllers are in `planning/`.

## Data and training interfaces

The public Cube trajectories are available from the [LeWM Cube dataset](https://huggingface.co/datasets/quentinll/lewm-cube/tree/main). Follow the upstream dataset instructions to download and extract `cube_single_expert.h5`. The compact task files included here are sufficient to run the evaluation commands above.

| Path | Training interface |
| --- | --- |
| `models/world_model.py` | High-level prediction and freezing of low-level modules |
| `models/macro_encoder.py` | Encoding variable-length action segments into macro actions |
| `models/components.py` | AM predictions for completion, residual, progress, and utility |
| `models/petal.py` | Shared effect encoder and differentiable endpoint comparison |
| `models/duration.py` | Fitting a duration posterior from macro-action latents and segment lengths |

The current Cube package provides these model interfaces and pretrained evaluation. Training from scratch also requires high-level segment construction, AM controller-rollout labels, PETAL teacher candidates and execution-outcome labels, and the training loops. See the repository [data guide](../datasets/README.md) for dataset organization.

## References

The visual backbone and expert trajectories build on [LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels](https://arxiv.org/abs/2603.19312), with [official code](https://github.com/lucas-maes/le-wm) and the [Cube dataset](https://huggingface.co/datasets/quentinll/lewm-cube/tree/main).

The hierarchical world model and online/staged controllers build on [Mind the Gap: Promises and Pitfalls of Hierarchical Planning in LeWorldModel](https://arxiv.org/abs/2607.12547), with [code and checkpoints archived on Zenodo](https://doi.org/10.5281/zenodo.21353240). Transformer building blocks retain their [LeWorldModel license](third_party/LeWorldModel-LICENSE). Simulation and CEM interfaces use [StableWorldModel](https://github.com/galilai-group/stable-worldmodel) and [OGBench](https://github.com/seohongpark/ogbench). Bibliographic entries are in the repository's `references.bib`.
