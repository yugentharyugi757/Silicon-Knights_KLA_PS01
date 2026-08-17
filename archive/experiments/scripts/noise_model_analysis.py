import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

BINS = np.linspace(0.0, 1.0, 21)


# ============================================================
# Utilities
# ============================================================

def load_npy(path):
    arr = np.load(path)

    arr = np.asarray(arr)

    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    if arr.ndim != 2:
        raise ValueError(
            f"Unsupported shape {arr.shape} for {path}"
        )

    return arr.astype(np.float64, copy=False)


def area_downsample(img, target_shape):
    """
    Area-average GT to the NoisyLR resolution.

    This is ONLY used as a diagnostic reference.
    It is NOT assumed to be the actual KLA degradation process.
    """

    target_h, target_w = target_shape

    h, w = img.shape

    if h % target_h != 0 or w % target_w != 0:
        raise ValueError(
            f"Cannot downsample {img.shape} -> {target_shape}"
        )

    scale_h = h // target_h
    scale_w = w // target_w

    return img.reshape(
        target_h,
        scale_h,
        target_w,
        scale_w
    ).mean(axis=(1, 3))


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=str,
        required=True
    )

    args = parser.parse_args()

    root = Path(args.root)

    gt_dir = root / "GT"
    noisy_dir = root / "NoisyLR"

    if not gt_dir.exists():
        raise FileNotFoundError(gt_dir)

    if not noisy_dir.exists():
        raise FileNotFoundError(noisy_dir)

    gt_files = {
        p.stem: p
        for p in gt_dir.glob("*.npy")
    }

    noisy_files = {
        p.stem: p
        for p in noisy_dir.glob("*.npy")
    }

    common = sorted(
        set(gt_files) & set(noisy_files)
    )

    print("=" * 70)
    print("KLA NOISE MODEL ANALYSIS")
    print("=" * 70)

    print(f"GT files       : {len(gt_files)}")
    print(f"NoisyLR files  : {len(noisy_files)}")
    print(f"Matched pairs  : {len(common)}")

    # --------------------------------------------------------
    # Collect intensity and residual values
    # --------------------------------------------------------

    intensity_values = []
    residual_values = []

    for i, stem in enumerate(common):

        gt = load_npy(gt_files[stem])
        noisy = load_npy(noisy_files[stem])

        clean_lr = area_downsample(
            gt,
            noisy.shape
        )

        residual = noisy - clean_lr

        intensity_values.append(
            clean_lr.ravel()
        )

        residual_values.append(
            residual.ravel()
        )

        if (i + 1) % 500 == 0:
            print(
                f"Processed {i + 1}/{len(common)} pairs..."
            )

    x = np.concatenate(intensity_values)
    r = np.concatenate(residual_values)

    print("\nTotal pixels analyzed:", len(x))

    # --------------------------------------------------------
    # Calculate variance for intensity bins
    # --------------------------------------------------------

    centers = []
    variances = []
    stds = []
    counts = []

    for low, high in zip(
        BINS[:-1],
        BINS[1:]
    ):

        mask = (
            (x >= low) &
            (x < high)
        )

        values = r[mask]

        if len(values) < 100:
            continue

        centers.append(
            (low + high) / 2
        )

        variances.append(
            np.var(values)
        )

        stds.append(
            np.std(values)
        )

        counts.append(
            len(values)
        )

    centers = np.asarray(centers)
    variances = np.asarray(variances)
    stds = np.asarray(stds)
    counts = np.asarray(counts)

    # ========================================================
    # Fit Model A: Constant variance
    # ========================================================

    constant_prediction = np.full_like(
        variances,
        np.mean(variances)
    )

    # ========================================================
    # Fit Model B: Linear variance
    #
    # variance = a*x + b
    # ========================================================

    linear_coeff = np.polyfit(
        centers,
        variances,
        deg=1
    )

    linear_prediction = np.polyval(
        linear_coeff,
        centers
    )

    # ========================================================
    # Fit Model C: Quadratic variance
    #
    # variance = a*x^2 + b*x + c
    # ========================================================

    quadratic_coeff = np.polyfit(
        centers,
        variances,
        deg=2
    )

    quadratic_prediction = np.polyval(
        quadratic_coeff,
        centers
    )

    # ========================================================
    # R2 calculation
    # ========================================================

    def r2_score(y_true, y_pred):

        ss_res = np.sum(
            (y_true - y_pred) ** 2
        )

        ss_tot = np.sum(
            (y_true - np.mean(y_true)) ** 2
        )

        return 1 - ss_res / ss_tot

    r2_constant = r2_score(
        variances,
        constant_prediction
    )

    r2_linear = r2_score(
        variances,
        linear_prediction
    )

    r2_quadratic = r2_score(
        variances,
        quadratic_prediction
    )

    # ========================================================
    # Print results
    # ========================================================

    print("\n" + "=" * 70)
    print("VARIANCE MODEL RESULTS")
    print("=" * 70)

    print(
        f"\nConstant model R² : {r2_constant:.6f}"
    )

    print(
        f"Linear model R²   : {r2_linear:.6f}"
    )

    print(
        f"Quadratic model R²: {r2_quadratic:.6f}"
    )

    print("\nLinear model:")
    print(
        f"variance = "
        f"{linear_coeff[0]:.8f} * x "
        f"+ {linear_coeff[1]:.8f}"
    )

    print("\nQuadratic model:")
    print(
        f"variance = "
        f"{quadratic_coeff[0]:.8f} * x² "
        f"+ {quadratic_coeff[1]:.8f} * x "
        f"+ {quadratic_coeff[2]:.8f}"
    )

    # ========================================================
    # Determine best model
    # ========================================================

    scores = {
        "constant": r2_constant,
        "linear": r2_linear,
        "quadratic": r2_quadratic
    }

    best_model = max(
        scores,
        key=scores.get
    )

    print(
        f"\nBest empirical variance model: "
        f"{best_model.upper()}"
    )

    # ========================================================
    # Save results
    # ========================================================

    output_dir = root.parent / "noise_analysis"

    output_dir.mkdir(
        exist_ok=True
    )

    np.savez(
        output_dir / "noise_model_results.npz",
        intensity_centers=centers,
        variances=variances,
        stds=stds,
        counts=counts,
        linear_coeff=linear_coeff,
        quadratic_coeff=quadratic_coeff,
        r2_constant=r2_constant,
        r2_linear=r2_linear,
        r2_quadratic=r2_quadratic
    )

    # ========================================================
    # Plot variance vs intensity
    # ========================================================

    plt.figure(figsize=(9, 6))

    plt.scatter(
        centers,
        variances,
        s=35,
        label="Observed variance"
    )

    plt.plot(
        centers,
        constant_prediction,
        linewidth=2,
        label="Constant"
    )

    plt.plot(
        centers,
        linear_prediction,
        linewidth=2,
        label="Linear"
    )

    plt.plot(
        centers,
        quadratic_prediction,
        linewidth=2,
        label="Quadratic"
    )

    plt.xlabel(
        "Clean-LR intensity"
    )

    plt.ylabel(
        "Residual variance"
    )

    plt.title(
        "Empirical Noise Variance vs Signal Intensity"
    )

    plt.legend()

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        output_dir /
        "variance_model_comparison.png",
        dpi=180
    )

    plt.close()

    # ========================================================
    # Plot standard deviation
    # ========================================================

    plt.figure(figsize=(9, 6))

    plt.scatter(
        centers,
        stds,
        s=35
    )

    plt.xlabel(
        "Clean-LR intensity"
    )

    plt.ylabel(
        "Residual standard deviation"
    )

    plt.title(
        "Noise Standard Deviation vs Signal Intensity"
    )

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        output_dir /
        "noise_std_vs_signal.png",
        dpi=180
    )

    plt.close()

    print(
        "\nResults saved to:"
    )

    print(
        output_dir.resolve()
    )

    print("\nDone.")


if __name__ == "__main__":
    main()