# Two-room

Hierarchical visual navigation with a learned world model, action-grounded support selection, PETAL effect matching, and Achievability Model (AM) scoring. The agent observes the current RGB image and the goal RGB image and replans while moving between two rooms.

## Quick start

Run commands from the repository root after installing the root requirements.

```bash
# Run ExAbs.
python -m two_room.evaluate --device cuda --method exabs

# Run the single-level world model.
python -m two_room.evaluate --device cuda --method single

# Run the hierarchical world-model baseline.
python -m two_room.evaluate --device cuda --method hwm

# Check model loading or save four episode videos.
python -m two_room.evaluate --device cpu --method hwm --check
python -m two_room.evaluate --device cuda --method exabs --limit 4 --video
```

CPU execution is also available with `--device cpu`; a CUDA GPU is recommended for the full planning workload. For a quick execution check, use `--limit 4 --max-steps 6`.

## Included files

| File | Purpose |
| --- | --- |
| `evaluate.py` | Evaluation command and closed-loop execution |
| `environment.py` | Two-room dynamics and RGB renderer |
| `planner.py` | ExAbs high-level search and low-level control |
| `baselines.py` | Single-level and HWM planners |
| `models.py` | World-model definitions and checkpoint loading |
| `training.py` | Main L2, AM, and PETAL objectives and optimizer helpers |
| `networks/` | Convolution blocks, the AM spatial adapter, and the Two-room PETAL score |
| `assets/tasks.json` | 64 initial-state/goal pairs and their 16 layouts |
| `assets/config.json` | ExAbs planning settings and random seed |
| `assets/baseline_config.json` | HWM planning settings |
| `assets/single_config.json` | Single-level planning settings |
| `assets/calibration.json` | AM probability maps and thresholds |
| `checkpoints/` | Shared L1, the complete ExAbs model, and HWM |

The package renders observations directly from the supplied task specifications. Evaluation requires no additional image dataset.

## Checkpoints

| File | Component |
| --- | --- |
| `l1.pt` | RGB encoder, low-level dynamics, and visual position readout |
| `ExAbs_two_room.pt` | Complete ExAbs: L2, macro-action encoder, AM, and PETAL |
| `hwm.pt` | HWM baseline trained with fixed-duration macro-actions |

## Evaluation settings

All methods use a default 300-step episode budget and the same task list. ExAbs uses five high-level waypoints with 20-step macro-actions, a 15-step low-level prediction horizon, and a six-step execution prefix. At step 210 the local controller follows the final goal directly. The selector retains 20% of spatial locations and shares that support across every candidate in a query. PETAL encodes both requested and predicted transitions at the fixed 15-step comparison scale. Its local score combines 90% predicted path cost with 10% scale-matched endpoint effect discrepancy. The path cost averages visual position error plus 0.01 times feature error over the predicted steps. High-level search scores 2,000 candidates; local control scores 500 candidates and a weighted-mean proposal. Online AM evaluates the first predicted waypoint at a fixed 20-step reference budget.

HWM and ExAbs use five 20-step macro-actions, a 15-step low-level prediction horizon, and a six-step execution prefix. Both follow the final goal directly from step 210. At each replanning step, both include the previous high-level plan shifted by six primitive steps alongside newly sampled candidates. All candidates are predicted and scored again from the latest observation to select the next waypoint. Both controllers also reuse their low-level action plan after shifting it by six steps. HWM scores predictions with the base world-model cost; ExAbs adds AM scoring and PETAL effect matching.

Single-level control predicts 30 primitive steps and executes ten before replanning.

Tasks run in batches of four. Planner seeds start at 4,300,000 for all three methods and increase by four per batch. Set `--seed` to use a chosen starting seed for any method. The saved summary records the method, budget, and planning configuration.

## Training framework

`training.py` provides tensor-batch objectives and single-step optimization for L2, AM, and PETAL. Prepare RGB observation/action trajectories, fixed candidate-pool records, and waypoint execution outcomes for training. The supplied `assets/tasks.json` contains evaluation start/goal specifications. Training trajectories are organized in the separate data directory under `$EXABS_DATA_DIR/two_room/`; see the [data guide](../datasets/README.md) for the layout.

Use the frozen low-level encoder to convert observations into `[B,16,43,43]` features. Primitive actions have two components. The training stages use these inputs:

| Stage | Function | Prepared inputs |
| --- | --- | --- |
| PETAL | `petal_loss` | Source/target features `[B,16,43,43]`, candidate endpoints `[B,K,16,43,43]`, actions `[B,K,15,2]`, teacher costs/probabilities `[B,K]`, nominal actions and teacher control update `[B,15,2]` |
| AM | `am_loss` | Source/target features `[B,16,43,43]`, first-hit times `[B]` in `1..25` or `-1` for attempts incomplete after 25 steps, and integer trajectory-group IDs `[B]` |
| Boundaries | `exact_partition` | Edge-utility array `[101,101]` for a 100-step window; invalid edges use negative infinity |
| L2 | `high_loss` | Boundary features `[B,6,16,43,43]`, padded action chunks `[B,5,25,2]`, valid-action masks `[B,5,25]`, and motion masks `[B,5,43,43]` |

1. **Prepare PETAL records.** Freeze the low-level model and selector. `Executor.teacher_query` collects candidate actions, predicted endpoints, path costs, and teacher probabilities for source/target pairs. Store the current nominal action sequence and the probability-weighted action sequence minus that nominal sequence as the teacher update. Optimize `petal_loss` using these records.
2. **Prepare AM records.** Freeze the PETAL-enabled controller. Execute fixed waypoint requests for up to 25 primitive steps and record the first completion time, or `-1` when the attempt reaches its full budget without completion. Keep repeated attempts from the same source trajectory under a common group ID. Optimize `am_loss`; the training helper updates the spatial residual (`networks/am_adapter.py`) and keeps the pretrained completion branch frozen. Fit the per-budget probability calibration and thresholds from labeled execution records, using the format in `assets/calibration.json`.
3. **Build high-level segments.** Freeze AM and score each possible edge at its observed duration. `networks.scoring.risk(..., seg=True)` returns the calibrated uncertainty-corrected score `q` and calibrated member disagreement `u`; set each valid edge utility to `log(q) - 0.25*u`. `exact_partition` selects five segments with durations from 10 to 25 while retaining both endpoints of the 100-step window.
4. **Train L2.** Gather boundary observations and the corresponding primitive-action chunks, pad each chunk to 25 actions, and construct its boolean validity mask. Derive motion masks from changes between consecutive boundary images. Optimize `high_loss`; its `weight` argument scales the visual position term alongside feature, motion-region, and autoregressive prediction terms.

For each stage, enable its component before creating the optimizer. For example, with a prepared high-level batch and chosen training settings:

```python
from two_room.models import load_models
from two_room.training import (
    configure_training,
    high_loss,
    make_optimizer,
    optimizer_step,
)

models = load_models("cuda")
high = configure_training(models, "high")
optimizer = make_optimizer(high, "high", learning_rate=learning_rate)
loss, _, _ = high_loss(high, models["low"], prepared_batch, weight=position_weight)
gradient_norm = optimizer_step(high, optimizer, loss)
```

`configure_training` also accepts `"petal"` and `"am"`. Their objectives include candidate-selection distillation and effect regularization for PETAL, and survival likelihood, integrated Brier loss, and grouped ensemble supervision for AM. The low-level model and selector remain frozen throughout these stages.

## Outputs

Outputs are written to `outputs/two_room/<method>/` by default, or to the path supplied with `--output`.

- `summary.json`: method, planning settings, episode count, success rate, successful-episode step count, and elapsed time.
- `episodes.json`: task identifiers, planner seeds, actions, positions, first-hit success times, and planning traces.
- `videos/`: episode videos when `--video` is enabled.

Success means reaching within 0.04 normalized distance of the goal. Results are calculated from the newly executed trajectories.

The convolutional world-model building blocks are adapted from [PLDM](https://github.com/vladisai/PLDM); see the repository's `licenses/PLDM.txt`.
