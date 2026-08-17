import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_npy(path):
    arr = np.load(path)
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"{path} has unsupported shape {arr.shape}; expected HxW.")
    return arr.astype(np.float32, copy=False)


def area_downsample(img, target_hw):
    """Area-average downsample for integer scale factors."""
    th, tw = target_hw
    h, w = img.shape
    if h % th != 0 or w % tw != 0:
        raise ValueError(f"Cannot integer-area-downsample {img.shape} -> {target_hw}.")
    sh, sw = h // th, w // tw
    return img.reshape(th, sh, tw, sw).mean(axis=(1, 3))


def safe_stats(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {}
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "p01": float(np.percentile(finite, 1)),
        "p05": float(np.percentile(finite, 5)),
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "fraction_below_0": float(np.mean(finite < 0)),
        "fraction_above_1": float(np.mean(finite > 1)),
    }


def pearson_corr(a, b):
    a = np.asarray(a).ravel().astype(np.float64)
    b = np.asarray(b).ravel().astype(np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def intensity_binned_residual_stats(clean_lr, noisy, bins):
    x = clean_lr.ravel()
    r = (noisy - clean_lr).ravel()
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (x >= lo) & (x < hi)
        if np.sum(mask) < 20:
            continue
        rr = r[mask]
        rows.append({
            "low": float(lo),
            "high": float(hi),
            "pixels": int(np.sum(mask)),
            "residual_mean": float(np.mean(rr)),
            "residual_std": float(np.std(rr)),
            "residual_abs_mean": float(np.mean(np.abs(rr))),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Path containing GT/ and NoisyLR/ folders, e.g. /content/train",
    )
    parser.add_argument("--max_pairs", type=int, default=0,
                        help="0 = all pairs; otherwise audit only first N pairs.")
    args = parser.parse_args()

    root = Path(args.root)
    gt_dir = root / "GT"
    noisy_dir = root / "NoisyLR"

    if not gt_dir.is_dir() or not noisy_dir.is_dir():
        raise FileNotFoundError(
            f"Expected {gt_dir} and {noisy_dir}. "
            "Pass the folder that directly contains GT and NoisyLR."
        )

    gt_files = {p.stem: p for p in gt_dir.glob("*.npy")}
    noisy_files = {p.stem: p for p in noisy_dir.glob("*.npy")}

    common = sorted(set(gt_files) & set(noisy_files))
    only_gt = sorted(set(gt_files) - set(noisy_files))
    only_noisy = sorted(set(noisy_files) - set(gt_files))

    if args.max_pairs > 0:
        common = common[:args.max_pairs]

    if not common:
        raise RuntimeError("No matching .npy pairs found.")

    output_dir = root.parent / "dataset_audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_rows = []
    all_gt = []
    all_noisy = []
    all_residual = []
    all_clean_lr = []
    correlations = []

    bins = np.linspace(0.0, 1.0, 11)

    for stem in common:
        gt = load_npy(gt_files[stem])
        noisy = load_npy(noisy_files[stem])

        if gt.shape[0] < noisy.shape[0] or gt.shape[1] < noisy.shape[1]:
            raise ValueError(f"{stem}: GT {gt.shape} is not larger than NoisyLR {noisy.shape}.")

        scale_h = gt.shape[0] / noisy.shape[0]
        scale_w = gt.shape[1] / noisy.shape[1]

        if scale_h != scale_w or scale_h != int(scale_h):
            raise ValueError(
                f"{stem}: unsupported scale GT={gt.shape}, NoisyLR={noisy.shape}."
            )

        scale = int(scale_h)
        clean_lr = area_downsample(gt, noisy.shape)
        residual = noisy - clean_lr

        all_gt.append(gt.ravel())
        all_noisy.append(noisy.ravel())
        all_clean_lr.append(clean_lr.ravel())
        all_residual.append(residual.ravel())

        corr = pearson_corr(clean_lr, noisy)
        correlations.append(corr)

        pair_rows.append({
            "id": stem,
            "gt_shape": list(gt.shape),
            "noisy_shape": list(noisy.shape),
            "scale_factor": scale,
            "gt_min": float(np.min(gt)),
            "gt_max": float(np.max(gt)),
            "noisy_min": float(np.min(noisy)),
            "noisy_max": float(np.max(noisy)),
            "gt_mean": float(np.mean(gt)),
            "gt_std": float(np.std(gt)),
            "noisy_mean": float(np.mean(noisy)),
            "noisy_std": float(np.std(noisy)),
            "residual_mean": float(np.mean(residual)),
            "residual_std": float(np.std(residual)),
            "residual_mae": float(np.mean(np.abs(residual))),
            "correlation_noisy_vs_downsampled_gt": corr,
            "intensity_binned_residual": intensity_binned_residual_stats(
                clean_lr, noisy, bins
            ),
        })

    gt_concat = np.concatenate(all_gt)
    noisy_concat = np.concatenate(all_noisy)
    clean_lr_concat = np.concatenate(all_clean_lr)
    residual_concat = np.concatenate(all_residual)

    report = {
        "dataset_root": str(root.resolve()),
        "num_common_pairs_audited": len(common),
        "num_gt_files": len(gt_files),
        "num_noisy_files": len(noisy_files),
        "missing_noisy_for_gt": only_gt,
        "missing_gt_for_noisy": only_noisy,
        "gt_global_stats": safe_stats(gt_concat),
        "noisy_global_stats": safe_stats(noisy_concat),
        "downsampled_gt_global_stats": safe_stats(clean_lr_concat),
        "residual_global_stats": safe_stats(residual_concat),
        "mean_pair_correlation": float(np.nanmean(correlations)),
        "median_pair_correlation": float(np.nanmedian(correlations)),
        "pairs": pair_rows,
    }

    with open(output_dir / "audit_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Plot 1: global histograms
    plt.figure(figsize=(10, 5))
    plt.hist(gt_concat, bins=100, alpha=0.5, label="GT")
    plt.hist(noisy_concat, bins=100, alpha=0.5, label="NoisyLR")
    plt.xlabel("Pixel value")
    plt.ylabel("Count")
    plt.title("Global Pixel-Value Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "global_histogram.png", dpi=160)
    plt.close()

    # Plot 2: residual vs clean intensity
    # Randomly subsample for plotting if very large.
    rng = np.random.default_rng(42)
    n = min(200_000, clean_lr_concat.size)
    idx = rng.choice(clean_lr_concat.size, size=n, replace=False)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        clean_lr_concat[idx],
        residual_concat[idx],
        s=2,
        alpha=0.15,
    )
    plt.axhline(0, linewidth=1)
    plt.xlabel("Downsampled GT intensity")
    plt.ylabel("NoisyLR - downsampled GT")
    plt.title("Residual vs. Signal Intensity")
    plt.tight_layout()
    plt.savefig(output_dir / "residual_vs_intensity.png", dpi=160)
    plt.close()

    # Plot 3: residual std by intensity bin
    bin_centers = []
    bin_stds = []
    for row in pair_rows:
        for b in row["intensity_binned_residual"]:
            bin_centers.append((b["low"] + b["high"]) / 2)
            bin_stds.append(b["residual_std"])

    plt.figure(figsize=(8, 5))
    plt.scatter(bin_centers, bin_stds, s=15, alpha=0.5)
    plt.xlabel("Clean-LR intensity bin center")
    plt.ylabel("Residual standard deviation")
    plt.title("Noise Strength vs. Signal Intensity")
    plt.tight_layout()
    plt.savefig(output_dir / "noise_std_vs_intensity.png", dpi=160)
    plt.close()

    print("=" * 70)
    print("KLA DATASET AUDIT COMPLETE")
    print("=" * 70)
    print(f"Pairs audited:       {len(common)}")
    print(f"GT files:            {len(gt_files)}")
    print(f"NoisyLR files:       {len(noisy_files)}")
    print(f"Missing noisy:       {len(only_gt)}")
    print(f"Missing GT:          {len(only_noisy)}")
    print(f"GT stats:            {report['gt_global_stats']}")
    print(f"NoisyLR stats:       {report['noisy_global_stats']}")
    print(f"Residual stats:      {report['residual_global_stats']}")
    print(f"Mean pair corr.:     {report['mean_pair_correlation']:.5f}")
    print(f"Median pair corr.:   {report['median_pair_correlation']:.5f}")
    print(f"\nResults saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
