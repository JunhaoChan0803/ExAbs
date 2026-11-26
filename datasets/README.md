# Training data

The navigation data bundle, `ExAbs-Training-Data.zip` on the [releases page](https://github.com/JunhaoChan0803/ExAbs/releases), contains simulator-generated observation/action trajectories, map layouts, and episode metadata. Store it in one directory and set:

```bash
export EXABS_DATA_DIR=/path/to/datasets
```

The code archive and data directory are separate:

```text
release/
├── ExAbs/
└── datasets/
    ├── README.md
    ├── manifest.json
    ├── SHA256SUMS
    ├── two_room/
    │   ├── trajectories.tar.zst
    │   └── metadata/
    └── counter_flow/
        ├── trajectories/
        │   ├── balanced.tar.zst
        │   └── long.tar.zst
        └── metadata/
```

## Two-room

The bundle contains 5,520 trajectories. The base collection includes 800 training episodes, 80 development episodes, and 80 calibration episodes. The additional collections contain 4,560 training trajectories covering navigation, goal reaching, exploration, local motion, and paired passage transitions.

Extract the trajectories:

```bash
tar --zstd -xf "$EXABS_DATA_DIR/two_room/trajectories.tar.zst" \
    -C "$EXABS_DATA_DIR/two_room"
```

Episodes use names such as `trajectories/base/train/episode_000000.npz`. The arrays include RGB images and primitive actions, together with the stored simulator positions and contact records. Each trajectory keeps its original arrays and data types.

`metadata/trajectories.json` records episode paths, collection and split identities, layout IDs, lengths, and checksums. `metadata/maps.json` describes the layouts. `metadata/local_windows.json` and `metadata/paired_passages.json` provide the recorded window/source indices.

## Counter-flow

Two trajectory pools are supplied:

| Pool | Training | Development | Calibration | Steps per episode |
| --- | ---: | ---: | ---: | ---: |
| Balanced | 1,280 | 128 | 0 | 80 |
| Long | 2,560 | 128 | 128 | 160 |

Extract both pools:

```bash
tar --zstd -xf "$EXABS_DATA_DIR/counter_flow/trajectories/balanced.tar.zst" \
    -C "$EXABS_DATA_DIR/counter_flow/trajectories"
tar --zstd -xf "$EXABS_DATA_DIR/counter_flow/trajectories/long.tar.zst" \
    -C "$EXABS_DATA_DIR/counter_flow/trajectories"
```

Episode paths follow `trajectories/<pool>/<split>/episode_000000.npz`. Arrays include `images`, `actions`, and `positions`; the long pool also includes `collisions` and `current_active`. Images are stored as unsigned 8-bit RGB arrays. The original action and position data types are preserved.

`metadata/balanced.json` and `metadata/long.json` list the episodes and array checksums. The matching map files record each layout.

## Read a trajectory

After extraction, load an episode with NumPy:

```python
import os
from pathlib import Path
import numpy as np

root = Path(os.environ["EXABS_DATA_DIR"])
path = root / "two_room/trajectories/base/train/episode_000000.npz"
with np.load(path, allow_pickle=False) as episode:
    images = episode["images"]  # [T + 1, H, W, 3]
    actions = episode["actions"]  # [T, 2]
```

The neural-model training inputs are RGB observations and actions. Simulator states and event arrays accompany the trajectories for replay and execution-outcome labeling. PETAL and AM additionally use planner-candidate and controller-execution records. The [Counter-flow trainer](../counter_flow/README.md#training) generates these records from the raw trajectories during training.

## Cube: public dataset

Download the expert trajectories from the [official LeWorldModel Cube dataset](https://huggingface.co/datasets/quentinll/lewm-cube/tree/main). The archive is approximately 46.2 GB; the extracted HDF5 used by this project is 101,942,558,720 bytes.

```bash
mkdir -p "$EXABS_DATA_DIR/cube"
curl --fail --location \
    "https://huggingface.co/datasets/quentinll/lewm-cube/resolve/main/cube_single_expert.tar.zst" \
    --output "$EXABS_DATA_DIR/cube/cube_single_expert.tar.zst"
tar --zstd -xf "$EXABS_DATA_DIR/cube/cube_single_expert.tar.zst" \
    -C "$EXABS_DATA_DIR/cube"
```

The expected SHA-256 of `cube_single_expert.h5` is:

```text
0664d507c4ff12009010644c9ae950836f954e700c172ccf22e7423af1a55625
```

On Ubuntu/Debian, `sudo apt-get install zstd` installs the decompressor used by these commands. The packaged Cube evaluation tasks run directly from the scene assets.

Please cite the data and environment sources:

- **LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels.** [Paper](https://arxiv.org/abs/2603.19312) · [Code](https://github.com/lucas-maes/le-wm) · [Data](https://huggingface.co/datasets/quentinll/lewm-cube).
- **OGBench: Benchmarking Offline Goal-Conditioned RL.** ICLR 2025. [Paper](https://arxiv.org/abs/2410.20092) · [Project](https://seohong.me/projects/ogbench/) · [Code](https://github.com/seohongpark/ogbench).
- **Hi-LeWM: Mind the Gap: Promises and Pitfalls of Hierarchical Planning in LeWorldModel.** [Paper](https://arxiv.org/abs/2607.12547) · [Code and checkpoints](https://doi.org/10.5281/zenodo.21353240).
