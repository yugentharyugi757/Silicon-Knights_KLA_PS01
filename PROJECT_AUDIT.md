# Project Audit: KLA Image Restoration

## Current final pipeline

| Item | Official artifact |
|---|---|
| Checkpoint | `checkpoints/best_noise_aware_dncnn.pth` |
| Model | `models/noise_aware_dncnn.py` |
| Training | `train/train_noise_aware.py` |
| Quantitative evaluation | `evaluate/evaluate_final.py` |
| Full-dataset inference | `evaluate/infer_full_dataset.py` |
| Benchmark reporting | `evaluate/build_master_benchmark.py` |

The model and evaluation pipeline are frozen. The checkpoint above is the only official final checkpoint for this documentation.

## Dataset and split

The active dataset consists of 3,200 paired `.npy` images:

- `train/GT/`: ground-truth images
- `train/NoisyLR/`: noisy low-resolution images

The seed-42 split used by the official evaluator is:

| Split | Images |
|---|---:|
| Train | 2,560 |
| Validation | 320 |
| Test | 320 |

The held-out 320-image test partition is the official quantitative benchmark. It must not be conflated with the full-dataset restoration run.

## Official final results

The following metrics were calculated on the held-out 320-image test set using the official checkpoint at epoch 48.

| Metric | Result |
|---|---:|
| MSE | 0.00271895 |
| MAE | 0.03212717 |
| PSNR | 27.8571 dB |
| SSIM | 0.746004 |
| LPIPS | 0.305770 |
| Mean inference time | 64.13 ms/image |
| Median inference time | 71.30 ms/image |

Best validation loss: `0.0025563989940565078`.

The reported timing was collected on an NVIDIA GeForce RTX 3050 Laptop GPU with CUDA.

## Evaluation versus full-dataset inference

`python evaluate/evaluate_final.py` deterministically reconstructs the 80%/10%/10% split and evaluates only its 320 held-out test images. It writes final metrics to `results/final_test_results.csv` and `results/final_test_summary.txt`, with restoration artifacts in `outputs/final_evaluation/`.

`python evaluate/infer_full_dataset.py` processes all 3,200 paired images for restoration/output generation. Its outputs are:

- `outputs/full_dataset_inference/restored/`
- `outputs/full_dataset_inference/comparisons/`

This 3,200-image run is not a test-set evaluation and must not be reported as one.

## Historical archive

`archive/experiments/` contains historical experimental models, scripts, outputs, and results. These artifacts are retained for provenance and comparison, but are not part of the official final pipeline or final benchmark.

## Current documentation status

The repository documentation identifies the active dataset, split, checkpoint, scripts, results, and output locations above. Historical references to a 456-image dataset/test set and obsolete final artifacts are not part of this current audit.
