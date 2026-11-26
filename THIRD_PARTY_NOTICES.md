# Third-party notices

The root MIT license applies to original ExAbs code. The components below retain their upstream licenses. Package dependencies installed separately retain the licenses included with those distributions.

| Component | Source | License | Use in this repository |
| --- | --- | --- | --- |
| PLDM | [vladisai/PLDM](https://github.com/vladisai/PLDM) | [MIT](licenses/PLDM.txt) | Convolutional encoder and predictor building blocks in the navigation packages |
| Hi-LeWM | [Code and checkpoints, Zenodo](https://zenodo.org/records/21353240) | [CC BY 4.0](licenses/Hi-LeWM.txt) | Cube hierarchical world model, macro-action encoder, policy, and deterministic execution helpers |
| LeWorldModel | [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm) | [MIT](licenses/LeWorldModel.txt) | Cube Transformer building blocks |

The Transformer building blocks in `cube/models/layers.py` derive from LeWorldModel [`module.py` at commit `8edfeb3`](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/module.py).

Hi-LeWM code-archive attribution: Niccolò Castelli, Francesco Massafra, Samuele Punzo, Salvatore Lo Sardo, and Ippokratis Pantelidis, *Hi-LeWM: Code and Checkpoints for Mind the Gap: Promises and Pitfalls of Hierarchical Planning in LeWorldModel*, Zenodo record 21353240. Author names and licensing follow the Zenodo archive metadata. The associated research paper has a separate entry in `references.bib`.

Adaptations in this repository include portable imports and model loading, inference-only tensor checkpoints, concise module names, and integration with AM and PETAL. Original notices are preserved alongside these components.

Simulation and planning dependencies include [OGBench](https://github.com/seohongpark/ogbench) and [StableWorldModel](https://github.com/galilai-group/stable-worldmodel). Their source and license information are available in the respective upstream projects and installed packages.
