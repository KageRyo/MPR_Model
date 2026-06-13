from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml


def test_reproduce_results_tiny_config(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "results_tiny_test"
    config_path = tmp_path / "tiny_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset": "data/dataV1_1000.csv",
                "output_dir": str(output_dir.relative_to(project_root.parent))
                if output_dir.is_relative_to(project_root.parent)
                else str(output_dir),
                "test_size": 0.2,
                "seeds": [0],
                "models": ["direct_wqi5", "lr"],
            }
        ),
        encoding="utf-8",
    )

    process = subprocess.run(
        [sys.executable, "scripts/reproduce_results.py", "--config", str(config_path)],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert process.returncode == 0, process.stderr or process.stdout
    expected_files = {
        "metrics_summary.csv",
        "repeated_split_results.csv",
        "residual_statistics.csv",
        "category_metrics.csv",
    }
    assert expected_files.issubset({path.name for path in output_dir.glob("*.csv")})


def test_reproduce_results_refuses_to_overwrite_existing_results(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "existing_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics_summary.csv").write_text("already-exists\n", encoding="utf-8")

    config_path = tmp_path / "tiny_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset": "data/dataV1_1000.csv",
                "output_dir": str(output_dir),
                "test_size": 0.2,
                "seeds": [0],
                "models": ["direct_wqi5", "lr"],
            }
        ),
        encoding="utf-8",
    )

    process = subprocess.run(
        [sys.executable, "scripts/reproduce_results.py", "--config", str(config_path)],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert process.returncode != 0
    assert "Use --output-dir to write elsewhere or pass --overwrite explicitly" in (process.stderr or process.stdout)


def test_reproduce_reduced_indicators_tiny_config(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "reduced_tiny_test"
    config_path = tmp_path / "reduced_tiny_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset": "data/dataV1_1000.csv",
                "output_dir": str(output_dir),
                "test_size": 0.2,
                "compute_device": "cpu",
                "include_direct_wqi5_full_baseline": True,
                "seeds": [0],
                "scenarios": {
                    "full": {"features": ["DO", "BOD", "NH3N", "EC", "SS"]},
                    "low_cost_core": {"features": ["DO", "EC", "SS"]},
                },
                "models": ["lr"],
            }
        ),
        encoding="utf-8",
    )

    process = subprocess.run(
        [sys.executable, "scripts/reproduce_reduced_indicators.py", "--config", str(config_path)],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert process.returncode == 0, process.stderr or process.stdout
    expected_files = {
        "reduced_indicator_results.csv",
        "reduced_indicator_summary.csv",
        "reduced_indicator_category_metrics.csv",
        "best_surrogate_by_scenario.csv",
    }
    assert expected_files.issubset({path.name for path in output_dir.glob("*.csv")})

    with (output_dir / "reduced_indicator_results.csv").open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert any(row["scenario"] == "full" and row["model_type"] == "direct_wqi5" for row in rows)
    assert not any(row["scenario"] == "low_cost_core" and row["model_type"] == "direct_wqi5" for row in rows)


def test_reproduce_reduced_indicators_refuses_to_overwrite_existing_results(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "existing_reduced_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "reduced_indicator_results.csv").write_text("already-exists\n", encoding="utf-8")

    config_path = tmp_path / "reduced_tiny_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset": "data/dataV1_1000.csv",
                "output_dir": str(output_dir),
                "test_size": 0.2,
                "compute_device": "cpu",
                "seeds": [0],
                "scenarios": {
                    "full": {"features": ["DO", "BOD", "NH3N", "EC", "SS"]},
                },
                "models": ["lr"],
            }
        ),
        encoding="utf-8",
    )

    process = subprocess.run(
        [sys.executable, "scripts/reproduce_reduced_indicators.py", "--config", str(config_path)],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert process.returncode != 0
    assert "Use --output-dir to write elsewhere or pass --overwrite explicitly" in (process.stderr or process.stdout)


def test_revision_missing_indicator_experiments_tiny_config(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "revision_missing_tiny_test"
    config_path = tmp_path / "revision_missing_tiny_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset_50000": "data/dataV1_1000.csv",
                "full_dataset": "data/dataV1.csv",
                "output_dir": str(output_dir),
                "test_size": 0.2,
                "expected_external_rows": None,
                "external_max_rows": 40,
                "compute_device": "cpu",
                "gpu_id": 0,
                "lightgbm_gpu_backend": "gpu",
                "n_bootstrap": 3,
                "save_models": True,
                "seeds": [0],
                "models": ["lr"],
                "stress_test": {
                    "enabled": True,
                    "scenarios": {
                        "combined_pollution": {
                            "DO": 0.7,
                            "BOD": 2.0,
                            "NH3N": 2.0,
                            "EC": 1.2,
                            "SS": 2.0,
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    process = subprocess.run(
        [
            sys.executable,
            "scripts/run_revision_missing_indicator_experiments.py",
            "--config",
            str(config_path),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert process.returncode == 0, process.stderr or process.stdout

    expected_files = {
        "manifest.json",
        "hardware.json",
        "versions.json",
        "predictions/predictions_long.csv",
        "metrics/metrics_by_seed.csv",
        "metrics/metrics_summary.csv",
        "metrics/best_by_experiment_source.csv",
        "metrics/stage1_reconstruction_metrics.csv",
        "metrics/error_by_wqi_band.csv",
        "stats/bootstrap_ci.csv",
        "stats/paired_error_tests.csv",
        "stress_tests/stress_summary.csv",
        "splits/split_indices.csv",
    }
    assert expected_files.issubset(
        {str(path.relative_to(output_dir)) for path in output_dir.rglob("*") if path.is_file()}
    )
    assert (output_dir / "models" / "seed_0" / "full_model" / "lr.joblib").exists()
    assert (output_dir / "models" / "seed_0" / "reduced_retraining" / "lr.joblib").exists()
    assert (output_dir / "models" / "seed_0" / "two_stage_reconstruction" / "lr.joblib").exists()

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["external_rows"] == 40
    assert manifest["experiments"] == [
        "full_reference",
        "full_inference_dropout",
        "reduced_retraining",
        "two_stage_reconstruction",
        "stress_scenarios",
    ]

    with (output_dir / "predictions" / "predictions_long.csv").open("r", encoding="utf-8") as handle:
        prediction_rows = list(csv.DictReader(handle))
    assert len(prediction_rows) == 960
    assert {row["experiment"] for row in prediction_rows} == {
        "full_reference",
        "full_inference_dropout",
        "reduced_retraining",
        "two_stage_reconstruction",
    }
    assert {row["source"] for row in prediction_rows} == {"internal_test", "external_10714"}

    with (output_dir / "metrics" / "metrics_by_seed.csv").open("r", encoding="utf-8") as handle:
        metric_rows = list(csv.DictReader(handle))
    assert len(metric_rows) == 8
    assert {"r2", "mae", "rmse", "nmae", "accuracy", "macro_f1"}.issubset(metric_rows[0])


def test_revision_missing_indicator_experiments_refuses_to_overwrite_existing_results(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "existing_revision_missing_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text("already-exists\n", encoding="utf-8")

    config_path = tmp_path / "revision_missing_tiny_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset_50000": "data/dataV1_1000.csv",
                "full_dataset": "data/dataV1.csv",
                "output_dir": str(output_dir),
                "test_size": 0.2,
                "expected_external_rows": None,
                "external_max_rows": 40,
                "compute_device": "cpu",
                "gpu_id": 0,
                "lightgbm_gpu_backend": "gpu",
                "n_bootstrap": 3,
                "save_models": True,
                "seeds": [0],
                "models": ["lr"],
                "stress_test": {"enabled": False, "scenarios": {}},
            }
        ),
        encoding="utf-8",
    )

    process = subprocess.run(
        [
            sys.executable,
            "scripts/run_revision_missing_indicator_experiments.py",
            "--config",
            str(config_path),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert process.returncode != 0
    assert "Output directory already contains files" in (process.stderr or process.stdout)
