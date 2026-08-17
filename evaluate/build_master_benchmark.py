from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
MASTER_CSV_PATH = RESULTS_DIR / "master_benchmark.csv"
MASTER_TXT_PATH = RESULTS_DIR / "master_benchmark.txt"
PLOTS_DIR = RESULTS_DIR / "benchmark_plots"

EXCLUDED_FILE_NAMES = {"master_benchmark.csv", "master_benchmark.txt"}
EXCLUDED_DIRECTORIES = {"benchmark_plots"}
FINAL_MODEL_LABELS = {
    "final_test_results.csv": "Noise-Aware DnCNN",
    "super_resolution_summary.csv": "Super-Resolution",
}


def normalize_column_name(value: object) -> str:
    text = str(value).strip().lower()
    text = text.replace("-", "_").replace(" ", "_").replace("/", "_")
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def parse_numeric(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "n/a", "na", "none"}:
        return np.nan
    text = text.replace(",", "")
    text = text.replace("dB", "").replace("db", "")
    text = text.replace("ms", "").replace("MS", "")
    text = text.replace("%", "")
    text = text.strip()
    try:
        return float(text)
    except ValueError:
        return np.nan


def discover_csv_files(results_root: Path) -> list[Path]:
    if not results_root.exists():
        return []
    files: list[Path] = []
    for path in sorted(results_root.rglob("*.csv")):
        if not path.is_file():
            continue
        if path.name in EXCLUDED_FILE_NAMES:
            continue
        if path.parent.name in EXCLUDED_DIRECTORIES:
            continue
        files.append(path)
    return files


def read_summary_txt_metrics(summary_path: Path) -> dict[str, float]:
    if not summary_path.exists():
        return {}

    text = summary_path.read_text(encoding="utf-8", errors="ignore")
    metrics: dict[str, float] = {}
    patterns = {
        "MSE": r"Mean MSE\s*[:=]\s*([0-9]+\.[0-9]+|[0-9]+)",
        "MAE": r"Mean MAE\s*[:=]\s*([0-9]+\.[0-9]+|[0-9]+)",
        "PSNR_dB": r"Mean PSNR\s*[:=]\s*([0-9]+\.[0-9]+|[0-9]+)",
        "SSIM": r"Mean SSIM\s*[:=]\s*([0-9]+\.[0-9]+|[0-9]+)",
        "LPIPS": r"Mean LPIPS\s*[:=]\s*([0-9]+\.[0-9]+|[0-9]+)",
        "Mean_Inference_ms": r"Mean inference\s*[:=]\s*([0-9]+\.[0-9]+|[0-9]+)",
        "Median_Inference_ms": r"Median inference\s*[:=]\s*([0-9]+\.[0-9]+|[0-9]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            try:
                metrics[key] = float(match.group(1))
            except ValueError:
                pass
    return metrics


def summarize_csv_metrics(csv_path: Path) -> dict[str, float]:
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {}
    if df.empty:
        return {}

    renamed = df.rename(columns={c: normalize_column_name(c) for c in df.columns})
    aliases = {
        "MSE": ["mse"],
        "MAE": ["mae"],
        "PSNR_dB": ["psnr", "psnr_db", "psnr_d_b"],
        "SSIM": ["ssim"],
        "LPIPS": ["lpips"],
        "Mean_Inference_ms": ["inference_ms", "mean_inference_ms", "mean_inference", "mean_time_ms", "time_ms"],
        "Median_Inference_ms": ["median_inference_ms", "median_time_ms"],
    }
    metrics: dict[str, float] = {}
    for metric_name, candidates in aliases.items():
        col_name = None
        for candidate in candidates:
            if candidate in renamed.columns:
                col_name = candidate
                break
        if col_name is None:
            continue
        values = pd.to_numeric(renamed[col_name].map(parse_numeric), errors="coerce").dropna()
        if values.empty:
            continue
        metrics[metric_name] = float(values.mean())
    return metrics


def row_for_final_model(csv_name: str, label: str) -> dict:
    csv_path = RESULTS_DIR / csv_name
    txt_name = "final_test_summary.txt" if label == "Noise-Aware DnCNN" else "super_resolution_summary.txt"
    summary_metrics = read_summary_txt_metrics(RESULTS_DIR / txt_name)
    metrics = summary_metrics if summary_metrics else summarize_csv_metrics(csv_path)

    row = {
        "Evaluation": label,
        "Degradation": "Final benchmark",
        "Gaussian_Sigma": np.nan,
        "Speckle_Sigma": np.nan,
        "MSE": np.nan,
        "MAE": np.nan,
        "PSNR_dB": np.nan,
        "SSIM": np.nan,
        "LPIPS": np.nan,
        "Mean_Inference_ms": np.nan,
        "Median_Inference_ms": np.nan,
        "Num_Images": 320,
        "Source_File": csv_name,
    }
    for metric_key, row_key in {
        "MSE": "MSE",
        "MAE": "MAE",
        "PSNR_dB": "PSNR_dB",
        "SSIM": "SSIM",
        "LPIPS": "LPIPS",
        "Mean_Inference_ms": "Mean_Inference_ms",
        "Median_Inference_ms": "Median_Inference_ms",
    }.items():
        if metric_key in metrics:
            row[row_key] = float(metrics[metric_key])
    return row


def extract_gaussian_sigma_from_filename(path: Path) -> float | None:
    match = re.search(r"gaussian[_\- ]?sigma[_\- ]?([0-9]+(?:\.[0-9]+)?)", path.name.lower())
    if match:
        return float(match.group(1))
    match = re.search(r"sigma[_\- ]?([0-9]+(?:\.[0-9]+)?)", path.name.lower())
    if match:
        return float(match.group(1))
    return None


def row_from_gaussian_csv(path: Path) -> dict | None:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None
    df = df.rename(columns={c: normalize_column_name(c) for c in df.columns})

    sigma = None
    if "sigma" in df.columns:
        series = pd.to_numeric(df["sigma"].map(parse_numeric), errors="coerce").dropna()
        if not series.empty:
            sigma = float(series.iloc[0])
    if sigma is None:
        sigma = extract_gaussian_sigma_from_filename(path)
    if sigma is None:
        return None

    row = {
        "Evaluation": f"Gaussian sigma {sigma:.2f}",
        "Degradation": "Gaussian Noise",
        "Gaussian_Sigma": sigma,
        "Speckle_Sigma": np.nan,
        "MSE": np.nan,
        "MAE": np.nan,
        "PSNR_dB": np.nan,
        "SSIM": np.nan,
        "LPIPS": np.nan,
        "Mean_Inference_ms": np.nan,
        "Median_Inference_ms": np.nan,
        "Num_Images": int(len(df)),
        "Source_File": path.name,
    }

    for metric_name, candidates in {
        "MSE": ["mse"],
        "MAE": ["mae"],
        "PSNR_dB": ["psnr", "psnr_db", "psnr_d_b"],
        "SSIM": ["ssim"],
        "LPIPS": ["lpips"],
        "Mean_Inference_ms": ["mean_time", "mean_time_ms", "mean_inference_ms", "mean_inference"],
        "Median_Inference_ms": ["median_time", "median_time_ms", "median_inference_ms", "median_inference"],
    }.items():
        for candidate in candidates:
            if candidate in df.columns:
                values = pd.to_numeric(df[candidate].map(parse_numeric), errors="coerce").dropna()
                if not values.empty:
                    row[metric_name] = float(values.mean())
                    break
    return row


def row_from_speckle_summary(path: Path) -> list[dict]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    if df.empty:
        return []
    df = df.rename(columns={c: normalize_column_name(c) for c in df.columns})
    if "speckle_strength" not in df.columns and "sigma" not in df.columns:
        return []

    rows: list[dict] = []
    sigma_col = "speckle_strength" if "speckle_strength" in df.columns else "sigma"
    for _, row in df.iterrows():
        sigma = row.get(sigma_col)
        if pd.isna(sigma):
            continue
        sigma_val = float(parse_numeric(sigma))
        values = {
            "Evaluation": f"Speckle sigma {sigma_val:.2f}",
            "Degradation": "Speckle",
            "Gaussian_Sigma": np.nan,
            "Speckle_Sigma": sigma_val,
            "MSE": np.nan,
            "MAE": np.nan,
            "PSNR_dB": np.nan,
            "SSIM": np.nan,
            "LPIPS": np.nan,
            "Mean_Inference_ms": np.nan,
            "Median_Inference_ms": np.nan,
            "Num_Images": int(row.get("images", len(df))),
            "Source_File": path.name,
        }
        for metric_name, candidates in {
            "MSE": ["mse"],
            "MAE": ["mae"],
            "PSNR_dB": ["psnr", "psnr_db", "psnr_d_b"],
            "SSIM": ["ssim"],
            "LPIPS": ["lpips"],
            "Mean_Inference_ms": ["mean_inference_ms", "mean_time_ms", "mean_time"],
            "Median_Inference_ms": ["median_inference_ms", "median_time_ms", "median_time"],
        }.items():
            for candidate in candidates:
                if candidate in df.columns and candidate in row and not pd.isna(row[candidate]):
                    values[metric_name] = float(row[candidate])
                    break
        rows.append(values)
    return rows


def row_from_combined_csv(path: Path) -> list[dict]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    if df.empty:
        return []
    df = df.rename(columns={c: normalize_column_name(c) for c in df.columns})

    rows: list[dict] = []
    for _, row in df.iterrows():
        sigma_g = row.get("gaussian_sigma")
        sigma_s = row.get("speckle_sigma")
        if pd.isna(sigma_g):
            sigma_g = row.get("sigma_gaussian")
        if pd.isna(sigma_s):
            sigma_s = row.get("sigma_speckle")
        if pd.isna(sigma_g) and pd.isna(sigma_s):
            continue

        label = "Gaussian + Speckle"
        if pd.notna(sigma_g) and pd.notna(sigma_s):
            label = f"Gaussian sigma {float(sigma_g):.2f} + Speckle sigma {float(sigma_s):.2f}"
        elif pd.notna(sigma_g):
            label = f"Gaussian sigma {float(sigma_g):.2f}"
        elif pd.notna(sigma_s):
            label = f"Speckle sigma {float(sigma_s):.2f}"

        record = {
            "Evaluation": label,
            "Degradation": "Gaussian + Speckle",
            "Gaussian_Sigma": np.nan if pd.isna(sigma_g) else float(sigma_g),
            "Speckle_Sigma": np.nan if pd.isna(sigma_s) else float(sigma_s),
            "MSE": np.nan,
            "MAE": np.nan,
            "PSNR_dB": np.nan,
            "SSIM": np.nan,
            "LPIPS": np.nan,
            "Mean_Inference_ms": np.nan,
            "Median_Inference_ms": np.nan,
            "Num_Images": int(row.get("images", len(df))),
            "Source_File": path.name,
        }
        for metric_name, candidates in {
            "MSE": ["mse"],
            "MAE": ["mae"],
            "PSNR_dB": ["psnr_db", "psnr", "psnr_d_b"],
            "SSIM": ["ssim"],
            "LPIPS": ["lpips"],
            "Mean_Inference_ms": ["mean_inference_ms", "mean_time_ms", "mean_time"],
            "Median_Inference_ms": ["median_inference_ms", "median_time_ms", "median_time"],
        }.items():
            for candidate in candidates:
                if candidate in df.columns and candidate in row and not pd.isna(row[candidate]):
                    record[metric_name] = float(row[candidate])
                    break
        rows.append(record)
    return rows


def build_master_dataframe(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=[
            "Evaluation",
            "Degradation",
            "Gaussian_Sigma",
            "Speckle_Sigma",
            "MSE",
            "MAE",
            "PSNR_dB",
            "SSIM",
            "LPIPS",
            "Mean_Inference_ms",
            "Median_Inference_ms",
            "Num_Images",
            "Source_File",
        ])

    df = pd.DataFrame(records)
    columns = [
        "Evaluation",
        "Degradation",
        "Gaussian_Sigma",
        "Speckle_Sigma",
        "MSE",
        "MAE",
        "PSNR_dB",
        "SSIM",
        "LPIPS",
        "Mean_Inference_ms",
        "Median_Inference_ms",
        "Num_Images",
        "Source_File",
    ]
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan
    df = df[columns].copy()

    order_map = {
        "Noise-Aware DnCNN": 0,
        "Super-Resolution": 1,
        "Gaussian sigma 0.01": 2,
        "Gaussian sigma 0.03": 3,
        "Gaussian sigma 0.05": 4,
        "Speckle sigma 0.05": 5,
        "Speckle sigma 0.10": 6,
        "Speckle sigma 0.20": 7,
    }
    df["_sort_order"] = df["Evaluation"].map(lambda v: order_map.get(str(v), 99))
    df = df.sort_values(["_sort_order", "Evaluation"]).drop(columns=["_sort_order"]).reset_index(drop=True)
    return df


def detect_duplicate_rows(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    duplicates = df[df.duplicated(subset=["Evaluation", "Gaussian_Sigma", "Speckle_Sigma"], keep=False)]
    if duplicates.empty:
        return []
    return duplicates[["Evaluation", "Gaussian_Sigma", "Speckle_Sigma", "Source_File"]].drop_duplicates().to_dict("records")


def save_master_csv(df: pd.DataFrame) -> None:
    columns = [
        "Evaluation",
        "Degradation",
        "Gaussian_Sigma",
        "Speckle_Sigma",
        "MSE",
        "MAE",
        "PSNR_dB",
        "SSIM",
        "LPIPS",
        "Mean_Inference_ms",
        "Median_Inference_ms",
        "Num_Images",
        "Source_File",
    ]
    if df.empty:
        pd.DataFrame(columns=columns).to_csv(MASTER_CSV_PATH, index=False)
        return
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = np.nan
    out[columns].to_csv(MASTER_CSV_PATH, index=False)


def save_master_txt(df: pd.DataFrame) -> None:
    with open(MASTER_TXT_PATH, "w", encoding="utf-8") as handle:
        handle.write("======================================================================\n")
        handle.write("KLA IMAGE RESTORATION - MASTER BENCHMARK\n")
        handle.write("======================================================================\n\n")

        final_rows = df[df["Evaluation"].isin(["Noise-Aware DnCNN", "Super-Resolution"])].copy()
        if not final_rows.empty:
            handle.write("FINAL MODEL BENCHMARKS\n")
            handle.write("----------------------\n")
            for _, row in final_rows.iterrows():
                handle.write(f"Evaluation: {row['Evaluation']}\n")
                handle.write(f"Source file: {row['Source_File']}\n")
                handle.write(f"MSE: {float(row['MSE']):.8f}\n" if pd.notna(row["MSE"]) else "MSE: N/A\n")
                handle.write(f"MAE: {float(row['MAE']):.8f}\n" if pd.notna(row["MAE"]) else "MAE: N/A\n")
                handle.write(f"PSNR: {float(row['PSNR_dB']):.4f} dB\n" if pd.notna(row["PSNR_dB"]) else "PSNR: N/A\n")
                handle.write(f"SSIM: {float(row['SSIM']):.6f}\n" if pd.notna(row["SSIM"]) else "SSIM: N/A\n")
                handle.write(f"LPIPS: {float(row['LPIPS']):.6f}\n" if pd.notna(row["LPIPS"]) else "LPIPS: N/A\n")
                handle.write(f"Mean inference: {float(row['Mean_Inference_ms']):.2f} ms\n" if pd.notna(row["Mean_Inference_ms"]) else "Mean inference: N/A\n")
                handle.write(f"Median inference: {float(row['Median_Inference_ms']):.2f} ms\n" if pd.notna(row["Median_Inference_ms"]) else "Median inference: N/A\n")
                handle.write("\n")

        robustness_rows = df[~df["Evaluation"].isin(["Noise-Aware DnCNN", "Super-Resolution"])].copy()
        if not robustness_rows.empty:
            handle.write("DEGRADATION ROBUSTNESS RESULTS\n")
            handle.write("-----------------------------\n")
            for _, row in robustness_rows.iterrows():
                handle.write(f"Evaluation: {row['Evaluation']}\n")
                handle.write(f"Degradation: {row['Degradation']}\n")
                if pd.notna(row["Gaussian_Sigma"]):
                    handle.write(f"Gaussian sigma: {float(row['Gaussian_Sigma']):.2f}\n")
                if pd.notna(row["Speckle_Sigma"]):
                    handle.write(f"Speckle sigma: {float(row['Speckle_Sigma']):.2f}\n")
                handle.write(f"MSE: {float(row['MSE']):.8f}\n" if pd.notna(row["MSE"]) else "MSE: N/A\n")
                handle.write(f"MAE: {float(row['MAE']):.8f}\n" if pd.notna(row["MAE"]) else "MAE: N/A\n")
                handle.write(f"PSNR: {float(row['PSNR_dB']):.4f} dB\n" if pd.notna(row["PSNR_dB"]) else "PSNR: N/A\n")
                handle.write(f"SSIM: {float(row['SSIM']):.6f}\n" if pd.notna(row["SSIM"]) else "SSIM: N/A\n")
                handle.write(f"LPIPS: {float(row['LPIPS']):.6f}\n" if pd.notna(row["LPIPS"]) else "LPIPS: N/A\n")
                handle.write(f"Mean inference: {float(row['Mean_Inference_ms']):.2f} ms\n" if pd.notna(row["Mean_Inference_ms"]) else "Mean inference: N/A\n")
                handle.write(f"Median inference: {float(row['Median_Inference_ms']):.2f} ms\n" if pd.notna(row["Median_Inference_ms"]) else "Median inference: N/A\n")
                handle.write("\n")


def create_plots(df: pd.DataFrame) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No benchmark results available", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "final_model_comparison.png", dpi=300)
        plt.close(fig)
        return

    final_rows = df[df["Evaluation"].isin(["Noise-Aware DnCNN", "Super-Resolution"])].copy()
    if not final_rows.empty:
        metrics = ["PSNR_dB", "SSIM", "LPIPS", "MSE", "MAE", "Mean_Inference_ms"]
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()
        for ax, metric in zip(axes, metrics):
            valid = final_rows[["Evaluation", metric]].dropna()
            if valid.empty:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center")
                ax.axis("off")
                continue
            values = valid[metric].astype(float)
            ax.bar(valid["Evaluation"].astype(str), values)
            ax.set_title(metric)
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "final_model_comparison.png", dpi=300)
        plt.close(fig)

    robustness_rows = df[~df["Evaluation"].isin(["Noise-Aware DnCNN", "Super-Resolution"])].copy()
    if not robustness_rows.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        psnr_values = robustness_rows["PSNR_dB"].astype(float)
        ax.bar(robustness_rows["Evaluation"].astype(str), psnr_values)
        ax.set_title("Robustness PSNR by condition")
        ax.set_ylabel("PSNR (dB)")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "robustness_psnr.png", dpi=300)
        plt.close(fig)


def print_master_table(df: pd.DataFrame) -> None:
    if df.empty:
        print("No benchmark data available.")
        return
    print("\n======================================================================")
    print("MASTER BENCHMARK")
    print("======================================================================")
    display = df[["Evaluation", "MSE", "MAE", "PSNR_dB", "SSIM", "LPIPS", "Mean_Inference_ms", "Median_Inference_ms"]].copy()
    print(display.to_string(index=False, formatters={
        "MSE": lambda v: "N/A" if pd.isna(v) else f"{float(v):.8f}",
        "MAE": lambda v: "N/A" if pd.isna(v) else f"{float(v):.8f}",
        "PSNR_dB": lambda v: "N/A" if pd.isna(v) else f"{float(v):.4f}",
        "SSIM": lambda v: "N/A" if pd.isna(v) else f"{float(v):.6f}",
        "LPIPS": lambda v: "N/A" if pd.isna(v) else f"{float(v):.6f}",
        "Mean_Inference_ms": lambda v: "N/A" if pd.isna(v) else f"{float(v):.2f}",
        "Median_Inference_ms": lambda v: "N/A" if pd.isna(v) else f"{float(v):.2f}",
    }))


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("======================================================================")
    print("KLA MASTER BENCHMARK GENERATION")
    print("======================================================================")
    print(f"Project:\n{PROJECT_ROOT}")
    print("\nMaster CSV excluded: YES")
    print("Searching results...")

    csv_files = discover_csv_files(RESULTS_DIR)
    print(f"Found: {len(csv_files)} CSV file(s)")
    for path in csv_files:
        print(f"    {path.relative_to(PROJECT_ROOT)}")

    records: list[dict] = []
    final_model_files = {"final_test_results.csv": "Noise-Aware DnCNN", "super_resolution_summary.csv": "Super-Resolution"}
    for csv_name, label in final_model_files.items():
        csv_path = RESULTS_DIR / csv_name
        if csv_path.exists():
            records.append(row_for_final_model(csv_name, label))

    for csv_path in csv_files:
        lower_name = csv_path.name.lower()
        if lower_name in {"final_test_results.csv", "super_resolution_summary.csv"}:
            continue
        if lower_name in {"noise_aware_test_results.csv", "noise_aware_lpips_results.csv"}:
            continue
        if lower_name == "gaussian_summary.csv":
            continue
        if lower_name == "speckle_summary.csv":
            records.extend(row_from_speckle_summary(csv_path))
            continue
        if lower_name.startswith("gaussian_sigma_"):
            row = row_from_gaussian_csv(csv_path)
            if row is not None:
                records.append(row)
            continue
        if lower_name.startswith("combined"):
            records.extend(row_from_combined_csv(csv_path))
            continue
        if lower_name.startswith("speckle") and lower_name.endswith(".csv"):
            records.extend(row_from_speckle_summary(csv_path))
            continue

    master_df = build_master_dataframe(records)

    duplicate_rows = detect_duplicate_rows(master_df)
    if duplicate_rows:
        print("\nWARNING: Duplicate evaluation rows detected:")
        for row in duplicate_rows:
            print(f"  - {row}")

    save_master_csv(master_df)
    save_master_txt(master_df)
    create_plots(master_df)
    print_master_table(master_df)

    print("\n======================================================================")
    print("OUTPUTS")
    print("======================================================================")
    print(f"Master CSV: {MASTER_CSV_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Master TXT: {MASTER_TXT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Plots: {PLOTS_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

