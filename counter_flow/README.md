# Counter-flow

Visual navigation through a downward current field. ExAbs combines a learned high-level world model, Achievability Model (AM) scoring, and PETAL effect comparison to select low-level actions.

## Quick start

Run from the repository root after installing its dependencies:

```bash
python -m counter_flow.evaluate --device cuda --limit 4
python -m counter_flow.evaluate --device cuda
```

The first command runs four episodes. The second evaluates all 200 bundled tasks. Each episode has a 250-step budget. An episode succeeds when the agent reaches within 0.04 normalized distance of the goal. Evaluation writes actions, positions, success flags, and a `summary.json` under `outputs/counter_flow/exabs/`.

Run each comparison with the same task suite and execution budget:

```bash
# Single-level world model
python -m counter_flow.evaluate --device cuda --method single

# Hierarchical world model (HWM)
python -m counter_flow.evaluate --device cuda --method hwm

# ExAbs
python -m counter_flow.evaluate --device cuda --method exabs
```

The single-level method uses L1 to plan directly toward the final goal. HWM uses its L2 model to propose intermediate goals for the shared L1 controller. ExAbs adds AM scoring and PETAL effect comparison. Outputs are stored in a separate directory for each method.

The deployment environment uses Python 3.10, PyTorch 2.5.1 with CUDA 12.4, NumPy 1.26.4, and Pillow 10.4.0. CUDA is recommended for the full suite. `--device cpu` supports inspecting model behavior and small runs; CPU and CUDA use different random-number streams.

## What is included

```text
counter_flow/
├── evaluate.py       # Evaluation command and model loading
├── environment.py    # Current-field physics and RGB rendering
├── models.py         # Visual backbone, high-level model, AM and PETAL
├── planner.py        # Hierarchical planning and local execution
├── train.py          # PETAL → AM → segmented L2 → export
├── data.py           # Trajectory loading and feature caching
├── collection.py     # Controller decisions and execution labels
├── calibration.py    # AM probability calibration
├── training.py       # Model objectives and optimization step
├── segmentation.py   # Temporal boundaries and training windows
├── networks/         # Visual encoder and predictor layers
├── checkpoints/
│   ├── l1.pt         # Shared low-level world model
│   ├── ExAbs_counter_flow.pt  # L2, AM, and PETAL
│   └── hwm.pt        # HWM high-level world model
└── assets/
    ├── config.json
    ├── tasks.json
    ├── prior.json
    ├── calibration.json
    ├── baseline_prior.json
    └── runtime/
```

The checkpoint files contain tensor state dictionaries. `tasks.json` supplies the environment layout, start, and goal for every episode. The evaluation command loads the selected model and its runtime assets automatically. Use `--checkpoints PATH` and `--assets PATH` to select another checkpoint or task directory.

## Evaluation options

```bash
python -m counter_flow.evaluate --device cuda:0 --output outputs/current
python -m counter_flow.evaluate --device cuda --start-batch 10 --end-batch 20
```

Tasks run in batches of four. Batch order and planner seeds are fixed in `assets/config.json`. `--limit` rounds up to a complete batch of four to preserve each task's sampling sequence. Batch intervals use an inclusive start and exclusive end; the full suite is batches 0 through 49.

The low-level controller predicts 15 steps and executes 10 before replanning. The high-level planner predicts four macro transitions. During the final 30 environment steps, the low-level planner targets the goal directly. All three methods share the low-level model, task suite, local planning horizon, and execution budget. The single-level method targets the final goal at every replanning step.

## Training

Install the root `requirements.txt`, download `ExAbs-Training-Data.zip` from the [releases page](https://github.com/JunhaoChan0803/ExAbs/releases), and extract it beside the repository. The trainer reads the Counter-flow raw trajectory pools and checks their archive and episode checksums. Compressed pools are extracted automatically on first use.

```bash
export EXABS_DATA_DIR=/path/to/datasets
export EXABS_RUN=outputs/counter_flow/trained

# PETAL: learn effect comparison from the frozen low-level planner.
python -m counter_flow.train --data-root "$EXABS_DATA_DIR/counter_flow" --output "$EXABS_RUN" --stage petal

# AM: collect fixed-waypoint executions, fit first-hit distributions, and calibrate.
python -m counter_flow.train --data-root "$EXABS_DATA_DIR/counter_flow" --output "$EXABS_RUN" --stage am

# L2: select temporal boundaries with AM and fit the high-level world model.
python -m counter_flow.train --data-root "$EXABS_DATA_DIR/counter_flow" --output "$EXABS_RUN" --stage l2

# Evaluate the newly trained model.
python -m counter_flow.evaluate --device cuda --method exabs \
  --checkpoints "$EXABS_RUN/checkpoints" --assets "$EXABS_RUN/assets" \
  --output "$EXABS_RUN/evaluation"
```

`--stage all` runs all training stages in order. The shared L1 stays frozen; L2, AM, and PETAL start from newly initialized parameters. PETAL preparation learns action-grounded spatial support internally, then freezes that support throughout effect learning and execution-label collection. Teacher candidate pools are generated from the supplied trajectories with a bounded memory cache.

AM learns first-hit times from fixed-waypoint executions lasting up to 30 primitive steps. Completed-at-start pairs are excluded; unsuccessful attempts use right censoring. AM probabilities are calibrated on reserved training-pool layouts. The frozen AM then scores observed trajectory edges, and dynamic programming selects four segments of 6–14 steps within each 40-step window. L2 learns the resulting observed endpoint transitions and action chunks. Its macro sampling prior is computed from the newly learned action encoder.

Settings are in [`configs/training.json`](configs/training.json). Use `--config PATH`, `--device cuda:N`, or `--low-checkpoint PATH` to supply a configuration, select a GPU, or select another trained L1. Microbatch sizes control peak training memory. L2 features are cached under `work/features/`; allow about 60 GB of free disk space for a full run, including execution labels and checkpoints. Repeating a stage with the same output resumes the saved model and optimizer state; completed stages reuse their saved checkpoints.

The output contains:

```text
trained/
├── checkpoints/
│   ├── l1.pt
│   └── ExAbs_counter_flow.pt
├── assets/             # Matching calibration, prior, support, and evaluation inputs
├── training_config.json
└── work/               # Locally generated stage checkpoints and execution labels
```

Check the complete training/export/evaluation path with small workloads:

```bash
python -m counter_flow.train --data-root "$EXABS_DATA_DIR/counter_flow" \
  --output outputs/counter_flow/training_check --smoke-test
python -m counter_flow.evaluate --device cuda --method exabs --limit 4 \
  --checkpoints outputs/counter_flow/training_check/checkpoints \
  --assets outputs/counter_flow/training_check/assets \
  --output outputs/counter_flow/training_check/evaluation
```

The small-workload option reduces update counts, candidate populations and evaluation budget to verify installation. Use the default configuration for full training. Each run writes its own progress and intermediate files under the selected output directory.

Unit tests for the training utilities run from the repository root:

```bash
python -m pip install pytest
python -m pytest tests
```

## Controller settings

`assets/config.json` contains the released evaluation settings. The local horizon is 15, the execution prefix is 10, and AM scores the first predicted waypoint at a fixed 10-step budget. PETAL uses the same 15-step comparison scale for the target and every candidate endpoint, retains 5% of spatial locations through the frozen selector, and refines the lowest-anchor 128 of 500 action candidates. Its local cost blends a 0.9 endpoint-anchor contribution with a 0.1 learned-effect contribution.

ExAbs and HWM use the same visual position readout in their inherited planning costs, the same 2,000 high-level and 500 low-level candidates, and the same macro-code projection and final direct-goal phase. The high-level cost averages over four predicted waypoints. The prior supplies the macro sampling distribution; projection follows the macro encoder's LayerNorm geometry. ExAbs adds AM scoring and PETAL's local endpoint comparison.
