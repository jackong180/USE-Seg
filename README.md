# USE-Seg-main

Based on the uploaded source code of Ultralytics 8.4.150, an instance segmentation project of YOLOv8s integrating UADConv, US2f, and EAA is implemented.

**Source of Implementation:** The uploaded package does not include UADConv and US2f. This project has been authorized by the user, and two modules have been supplemented based on the method section of T-ASE.docx; EAA has undergone numerical correction based on the original EntropyAttention. It is a trainable reconstruction implementation, not the original experimental code that has verified and reproducible paper metrics. The dataset and confirmed USE-Seg training weights are not provided with the package; the identity of the original package's best.pt is not confirmed, and it is not packaged as the USE-Seg weight.
|---|---|
| `ultralytics/nn/modules/uadconv.py` | UADConv (including outer BN, SiLU, i.e. UBS) |
| `ultralytics/nn/modules/us2f.py` | US2f, UADBottleneck, four groups of random permutations |
| `ultralytics/nn/modules/eaa.py` | EAA attention |
| `ultralytics/nn/tasks.py` | YAML module parsing, channel scaling and repetition count registration |
| `ultralytics/nn/modules/__init__.py` | Module export |
| `configs/use-seg.yaml` | Complete USE-Seg Small network |
| `configs/baseline.yaml` | Original YOLOv8s-Seg, 7 classes |
| `configs/uadconv.yaml`, `us2f.yaml`, `eaa.yaml` | Single-module ablation configuration |
| `configs/WSD.yaml` | Data configuration |
| `run.py` | Unified entry for training, validation, inference, and export |
| `verify.py` | No need for dataset structure, segmentation loss, backpropagation check |
| `IMPLEMENTATION.md` | Calculation formula, design choices, differences from the original package |
| `VALIDATION.md` | Actual validation scope and results of this experiment |

Keep the complete `ultralytics/` run package intact, avoiding indirect dependencies for training, validation, export, etc. Do not package the original repository website documentation, CI, IDE cache, run results, and irrelevant images. Keep the AGPL-3.0 LICENSE and upstream CITATION.cff.

```bash
cd USE-Seg-main
python -m pip install -e .
python -c "import ultralytics; print(ultralytics.__file__)"
python verify.py
```

The print path should point to this project. Do not overwrite the local modifications with `pip install -U ultralytics`. CPU compatibility check; actual training is recommended to use GPU. The dependency declaration retains the pyproject.toml for uploading the source code; the new modules in this project only depend on PyTorch.

## Dataset

Edit the `path` in `configs/WSD.yaml`, it is recommended to use the absolute path of the data root directory (using forward slashes on Windows). Place `images/train`, `images/val`, `images/test` and corresponding `labels/train`, `labels/val`, `labels/test` in this directory.

The labels must be in the YOLO polygon segmentation format: `class x1 y1 x2 y2 ...`, the coordinates should be normalized to [0,1], and cannot be replaced by bounding box labels with only the center and height.

This project uses the configuration order pointed to by the original training entry: 

| ID | Category |
|---|---|
| 0 | PARTICLE |
| 1 | PIQ PARTICLE |
| 2 | SCRATCH |
| 3 | COATING BAD |
| 4 | PO CONTAMINATION |
| 5 | BLOCK ETCH |
| 6 | SEZ BURNT |

The order of the files `datasets/WSD.yaml` and `ultralytics/cfg/datasets/WSD.yaml` is inconsistent. Here, the latter file is uniformly used, but **the actual labels are not modified**; before training, it is necessary to check the class IDs of the exported data. Changing the YAML name will not automatically re-map the labels.

```bash
python run.py train --device 0 --epochs 200 --batch 2 --workers 0
python run.py train --model configs/baseline.yaml --name baseline --device 0
python run.py train --model configs/uadconv.yaml --name UADConv --device 0
python run.py train --model configs/us2f.yaml --name US2f --device 0
python run.py train --model configs/eaa.yaml --name EAA --device 0
```

Default training from scratch, SGD, lr0 = 0.01, 640 input, random seed 0. Save additional at every 10 epochs. All configurations are fixed as Small with depth = 0.33 and width = 0.50. Do not guess the model type based on the file name. In the ablation study, US2f already includes UADConv; `us2f.yaml` indicates only replacing C2f, not the independent Conv of the backbone.

Optional `--weights /path/to/checkpoint.pt` to initialize compatible parameters. The new structure cannot be fully loaded from the original YOLO weights; please check the loading log. This is not equivalent to resuming training. If you need to resume from a breakpoint, you can use the native interface `YOLO('runs/segment/USE-Seg/weights/last.pt').train(resume=True)`.

## Validation, Inference, Export

```bash
python run.py val --weights runs/segment/USE-Seg/weights/best.pt --split test --device 0
python run.py predict --weights runs/segment/USE-Seg/weights/best.pt --source path/to/images --device 0
python run.py export --weights runs/segment/USE-Seg/weights/best.pt --format onnx --device cpu
```

The export entry follows the upstream implementation. The target format may require additional dependencies; ONNX/TensorRT/RKNN deployment has not been verified this time. Execute `python run.py --help` to view all parameters. When running on CPU, change to `--device cpu`.

## Experimental Boundaries

The real data of WSD and the trained USE-Seg weights are not included, which means that the Mask AP50 = 74.9%, Box mAP50 = 78.8%, or FPS in the paper cannot be guaranteed. The model must be re-trained and re-evaluated on a unified data partition and testing platform. The explicit engineering choice of the code in this paper can be found in IMPLEMENTATION.md.
