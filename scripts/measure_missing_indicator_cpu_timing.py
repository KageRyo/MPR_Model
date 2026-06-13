from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import joblib
import pandas as pd
import yaml
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_missing_indicator_robustness_experiments import (  # noqa: E402
    COMPLETE_SET,
    FULL_REFERENCE,
    INDICATOR_RECONSTRUCTION,
    INFERENCE_DROPOUT,
    REDUCED_RETRAINING,
    experiment_name,
    extract_external_set,
    predict_bundle,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_config_path(args_config: str | None, manifest: dict[str, Any]) -> Path:
    config_value = args_config or manifest["config_path"]
    config_path = Path(config_value)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    return config_path


def set_cpu_backend(bundle: dict[str, Any]) -> None:
    models = [bundle.get("wqi_model")]
    models.extend(bundle.get("indicator_models", {}).values())
    for estimator in models:
        if estimator is None:
            continue
        model = getattr(estimator, "named_steps", {}).get("model", estimator)
        if hasattr(model, "set_params"):
            params = model.get_params(deep=False) if hasattr(model, "get_params") else {}
            try:
                model.set_params(device="cpu")
            except (TypeError, ValueError):
                pass
            if "device_type" in params:
                try:
                    model.set_params(device_type="cpu")
                except (TypeError, ValueError):
                    pass


def model_path(output_dir: Path, seed: int, missing_set: str, mode: str, model_type: str) -> Path:
    if mode in {FULL_REFERENCE, INFERENCE_DROPOUT}:
        return output_dir / "models" / f"seed_{seed}" / FULL_REFERENCE / f"{model_type}.joblib"
    return output_dir / "models" / f"seed_{seed}" / mode / missing_set / f"{model_type}.joblib"


def main() -> None:
    args = parse_args()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    config_path = resolve_config_path(args.config, manifest)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    timing_config = config.get("cpu_timing", {})
    repeats = args.repeats if args.repeats is not None else int(timing_config.get("repeats", 5))
    warmup = args.warmup if args.warmup is not None else int(timing_config.get("warmup", 1))
    output_path = output_dir / "timing" / "cpu_only_inference_timing.csv"
    summary_path = output_dir / "timing" / "cpu_only_inference_timing_summary.csv"
    if (output_path.exists() or summary_path.exists()) and not args.overwrite:
        raise FileExistsError("CPU timing outputs already exist. Pass --overwrite to replace them.")

    subset = pd.read_csv(PROJECT_ROOT / config["dataset_50000"])
    full = pd.read_csv(PROJECT_ROOT / config["full_dataset"])
    external = extract_external_set(full, subset, config.get("expected_external_rows"), config.get("external_max_rows"))

    rows: list[dict[str, Any]] = []
    experiments: list[tuple[str, str]] = [(COMPLETE_SET, FULL_REFERENCE)]
    for missing_set in manifest["missing_sets"]:
        experiments.extend(
            [
                (missing_set, INFERENCE_DROPOUT),
                (missing_set, REDUCED_RETRAINING),
                (missing_set, INDICATOR_RECONSTRUCTION),
            ]
        )

    for seed in manifest["seeds"]:
        for model_type in manifest["models"]:
            for missing_set, mode in experiments:
                path = model_path(output_dir, int(seed), missing_set, mode, model_type)
                logger.info("timing seed={} model={} experiment={}", seed, model_type, experiment_name(mode, missing_set))
                bundle = joblib.load(path)
                set_cpu_backend(bundle)
                for _ in range(warmup):
                    predict_bundle(mode, bundle, external)
                for repeat in range(repeats):
                    started = time.perf_counter()
                    predictions = predict_bundle(mode, bundle, external)
                    elapsed = time.perf_counter() - started
                    rows.append(
                        {
                            "source": "external_10714",
                            "missing_set": missing_set,
                            "experiment_mode": mode,
                            "experiment": experiment_name(mode, missing_set),
                            "seed": int(seed),
                            "model_type": model_type,
                            "repeat": repeat,
                            "n_rows": len(predictions),
                            "latency_s": elapsed,
                            "latency_per_row_ms": elapsed * 1000.0 / len(predictions),
                            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
                            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", ""),
                            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS", ""),
                            "numexpr_num_threads": os.environ.get("NUMEXPR_NUM_THREADS", ""),
                        }
                    )

    frame = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    summary = (
        frame.groupby(["source", "missing_set", "experiment_mode", "experiment", "model_type"], as_index=False)
        .agg(
            n_repeats=("repeat", "count"),
            n_rows=("n_rows", "first"),
            latency_s_mean=("latency_s", "mean"),
            latency_s_std=("latency_s", "std"),
            latency_per_row_ms_mean=("latency_per_row_ms", "mean"),
            latency_per_row_ms_std=("latency_per_row_ms", "std"),
        )
        .sort_values(["source", "missing_set", "experiment_mode", "model_type"])
    )
    summary.to_csv(summary_path, index=False)
    write_csv(
        output_dir / "timing" / "cpu_only_environment.csv",
        [
            {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", ""),
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", ""),
                "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS", ""),
                "repeats": repeats,
                "warmup": warmup,
            }
        ],
    )
    logger.success(f"CPU-only timing written to: {output_path}")


if __name__ == "__main__":
    main()
