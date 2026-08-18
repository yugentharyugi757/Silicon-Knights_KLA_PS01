# KLA Image Restoration

Noise-aware grayscale image restoration and 2× upsampling using the final Noise-Aware DnCNN checkpoint.

## Overview

This project restores paired noisy low-resolution grayscale `.npy` images to one-channel 2× outputs. The final pipeline uses the frozen checkpoint `checkpoints/best_noise_aware_dncnn.pth`, the model in `models/noise_aware_dncnn.py`, and the official evaluator in `evaluate/evaluate_final.py`.

## Objective

The objective is to reconstruct a clean, higher-resolution grayscale image from a noisy low-resolution input. The model receives both the noisy image and an estimated signal-dependent noise map.

## Methodology and architecture

For each noisy input, the pipeline derives a noise-sigma map from the signal-dependent variance model used during training. The model consumes the noisy image and this map as a two-channel tensor.

`NoiseAwareDnCNN` comprises:

- a 2-channel input: noisy grayscale LR image and signal-dependent noise sigma map;
- a 3×3 convolutional head with 64 features;
- 10 residual blocks, each using 3×3 convolutions, BatchNorm, and ReLU;
- a global feature residual connection followed by a 3×3 reconstruction convolution;
- 2× PixelShuffle upsampling; and
- a final 3×3 convolution producing a one-channel output.

## Dataset and split

The dataset contains 3,200 paired `.npy` images in `train/GT/` and `train/NoisyLR/`. With seed 42, the official split is 2,560 training images, 320 validation images, and 320 test images. **These metrics are calculated on the held-out 320-image test set.**

## Training configuration

Training is optional; the supplied checkpoint is the official final artifact. The frozen training script uses 50 epochs, batch size 16, Adam (`lr=1e-3`, `weight_decay=1e-5`), MSE loss, gradient clipping at 1.0, and a `ReduceLROnPlateau` scheduler (factor 0.5, patience 5). The final checkpoint was saved at epoch 48 with best validation loss `0.0025563989940565078`.

## Final benchmark

Hardware: NVIDIA GeForce RTX 3050 Laptop GPU with CUDA.

| Metric | Final Result |
|---|---:|
| MSE | 0.00271895 |
| MAE | 0.03212717 |
| PSNR | 27.8571 dB |
| SSIM | 0.746004 |
| LPIPS | 0.305770 |
| Mean inference | 64.13 ms/image |
| Median inference | 71.30 ms/image |

## Ablation results

Archived ablation results compare the final noise-aware model with a baseline DnCNN on 320 images. The noise-aware model improved PSNR by 0.3676 dB and SSIM by 0.009750, and reduced MSE relative to the baseline. LPIPS was 0.007085 higher than the baseline in this evaluation, so LPIPS did not improve. These historical results are retained in `archive/experiments/results/ablation_summary.txt` and are not the official final benchmark.

## Full-dataset inference

Run:

```powershell
python evaluate/infer_full_dataset.py
```

This script processes all 3,200 paired images to generate restorations and comparison images. It is output generation, **not** held-out test-set evaluation. Outputs are written to:

- `outputs/full_dataset_inference/restored/`
- `outputs/full_dataset_inference/comparisons/`

## Repository structure

```text
checkpoints/best_noise_aware_dncnn.pth  # official checkpoint
models/noise_aware_dncnn.py             # final architecture
train/train_noise_aware.py               # optional training script
evaluate/evaluate_final.py               # official 320-image benchmark
evaluate/infer_full_dataset.py           # full-dataset restoration
evaluate/build_master_benchmark.py       # benchmark report generation
train/GT/ and train/NoisyLR/             # paired dataset
results/                                 # final metrics and benchmark reports
outputs/                                 # generated restorations and comparisons
archive/experiments/                     # historical experiments, not final pipeline
```

## Installation

From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

Official quantitative evaluation:

```powershell
python evaluate/evaluate_final.py
```

Full-dataset restoration:

```powershell
python evaluate/infer_full_dataset.py
```

Benchmark reporting:

```powershell
python evaluate/build_master_benchmark.py
```

Optional retraining:

```powershell
python train/train_noise_aware.py
```

## Output locations

The official evaluator writes per-image metrics to `results/final_test_results.csv`, the summary to `results/final_test_summary.txt`, and its held-out-test restorations and comparisons under `outputs/final_evaluation/`. Benchmark reporting writes `results/master_benchmark.csv`, `results/master_benchmark.txt`, and plots in `results/benchmark_plots/`.

## Limitations

Results reflect this dataset, split, checkpoint, and hardware configuration. Restoration quality may vary for images or degradations outside this setting. Inference times are hardware-dependent.

## Reproducibility

The training and evaluator scripts fix seed 42 and use the same 80%/10%/10% split logic. Reproduce the reported benchmark with the official checkpoint and `python evaluate/evaluate_final.py`; do not substitute the 3,200-image inference run for quantitative evaluation.

## License

No license file is currently provided. All rights and reuse terms should be clarified by the repository owner before redistribution.

## Official KLA Submission Inference

Run the final restoration pipeline using:

    python run.py <input-dir> <output-dir>

Example:

    python run.py test_input test_output

The input directory must contain grayscale .npy files. The output directory is created automatically and contains one restored .npy file for every input, using the same filename.

The restored output is a grayscale (H, W) array with loat32 values in the range [0, 1]. The pipeline uses the supplied final checkpoint at checkpoints/best_noise_aware_dncnn.pth and automatically uses CUDA when an NVIDIA GPU is available.
