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


def test_sample_size_experiments_tiny_workflow(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "sample_size_results"
    model_dir = tmp_path / "sample_size_models"

    process = subprocess.run(
        [
            sys.executable,
            "scripts/run_sample_size_experiments.py",
            "--datasets",
            "data/dataV1_1000.csv",
            "--models",
            "lr",
            "--n-splits",
            "2",
            "--output-dir",
            str(output_dir),
            "--model-dir",
            str(model_dir),
            "--compute-device",
            "cpu",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert process.returncode == 0, process.stderr or process.stdout
    expected_files = {
        "manifest.json",
        "metrics/metrics_by_fold.csv",
        "metrics/metrics_summary.csv",
        "splits/split_indices.csv",
    }
    assert expected_files.issubset(
        {str(path.relative_to(output_dir)) for path in output_dir.rglob("*") if path.is_file()}
    )
    assert (model_dir / "LR" / "dataV1_1000" / "fold_1" / "lr.pkl").exists()
    assert (model_dir / "LR" / "dataV1_1000" / "fold_2" / "lr.pkl").exists()

    with (output_dir / "metrics" / "metrics_by_fold.csv").open("r", encoding="utf-8") as handle:
        metric_rows = list(csv.DictReader(handle))
    assert len(metric_rows) == 4
    assert {"r2", "mae", "rmse", "split", "artifact_path"}.issubset(metric_rows[0])
    assert {row["split"] for row in metric_rows} == {"train", "test"}


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


def test_missing_indicator_experiments_tiny_config(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "missing_indicator_tiny_test"
    config_path = tmp_path / "missing_indicator_tiny_config.yaml"
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
            "scripts/run_missing_indicator_experiments.py",
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

    for derived_path in [
        output_dir / "stats" / "bootstrap_ci.csv",
        output_dir / "stats" / "paired_error_tests.csv",
        output_dir / "stress_tests" / "stress_summary.csv",
    ]:
        derived_path.unlink()

    derived_process = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_missing_indicator_derived_outputs.py",
            "--output-dir",
            str(output_dir),
            "--config",
            str(config_path),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert derived_process.returncode == 0, derived_process.stderr or derived_process.stdout
    assert (output_dir / "stats" / "bootstrap_ci.csv").exists()
    assert (output_dir / "stats" / "paired_error_tests.csv").exists()
    assert (output_dir / "stress_tests" / "stress_summary.csv").exists()


def test_missing_indicator_experiments_refuses_to_overwrite_existing_results(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "existing_missing_indicator_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text("already-exists\n", encoding="utf-8")

    config_path = tmp_path / "missing_indicator_tiny_config.yaml"
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
            "scripts/run_missing_indicator_experiments.py",
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


def test_missing_indicator_robustness_tiny_workflow(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "robustness_tiny"
    config_path = tmp_path / "missing_indicator_robustness_tiny.yaml"
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
                "missing_sets": {
                    "missing_bod": {"missing_indicators": ["BOD"]},
                    "missing_nh3n": {"missing_indicators": ["NH3N"]},
                    "missing_bod_nh3n": {"missing_indicators": ["BOD", "NH3N"]},
                },
                "event_window_stress": {
                    "enabled": True,
                    "source": "external_10714",
                    "window_fraction": 0.1,
                    "window_position": "middle",
                    "context_multiplier": 1,
                    "scenarios": {
                        "combined_event_30pct": {
                            "DO": 0.7,
                            "BOD": 1.3,
                            "NH3N": 1.3,
                            "SS": 1.3,
                        }
                    },
                },
                "stress107_event_windows": {
                    "enabled": True,
                    "source": "external_10714",
                    "window_mode": "sequential_equal_blocks",
                    "n_windows": 4,
                    "severities": {
                        "low_30pct": {"perturbation_pct": 30},
                        "medium_100pct": {"perturbation_pct": 100},
                    },
                    "scenarios": {
                        "combined_pollution": {
                            "DO_decrease_factors": {
                                "low_30pct": 0.7,
                                "medium_100pct": 0.5,
                            },
                            "increase_indicators": {
                                "BOD": {
                                    "low_30pct": 1.3,
                                    "medium_100pct": 2.0,
                                },
                                "NH3N": {
                                    "low_30pct": 1.3,
                                    "medium_100pct": 2.0,
                                },
                                "SS": {
                                    "low_30pct": 1.3,
                                    "medium_100pct": 2.0,
                                },
                            },
                        }
                    },
                },
                "cpu_timing": {
                    "enabled": True,
                    "source": "external_10714",
                    "repeats": 1,
                    "warmup": 0,
                },
            }
        ),
        encoding="utf-8",
    )

    run_process = subprocess.run(
        [
            sys.executable,
            "scripts/run_missing_indicator_robustness_experiments.py",
            "--config",
            str(config_path),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_process.returncode == 0, run_process.stderr or run_process.stdout

    timing_process = subprocess.run(
        [
            sys.executable,
            "scripts/measure_missing_indicator_cpu_timing.py",
            "--output-dir",
            str(output_dir),
            "--config",
            str(config_path),
            "--repeats",
            "1",
            "--warmup",
            "0",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert timing_process.returncode == 0, timing_process.stderr or timing_process.stdout

    stress107_dir = tmp_path / "robustness_tiny_stress107"
    stress107_process = subprocess.run(
        [
            sys.executable,
            "scripts/run_stress107_event_windows.py",
            "--artifact-dir",
            str(output_dir),
            "--output-dir",
            str(stress107_dir),
            "--config",
            str(config_path),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert stress107_process.returncode == 0, stress107_process.stderr or stress107_process.stdout

    workbook_path = output_dir / "reports" / "robustness_tiny.xlsx"
    excel_process = subprocess.run(
        [
            sys.executable,
            "scripts/export_missing_indicator_robustness_excel.py",
            "--output-dir",
            str(output_dir),
            "--output-file",
            str(workbook_path),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert excel_process.returncode == 0, excel_process.stderr or excel_process.stdout
    assert workbook_path.exists()

    stress107_workbook_path = stress107_dir / "reports" / "stress107_tiny.xlsx"
    stress107_excel_process = subprocess.run(
        [
            sys.executable,
            "scripts/export_missing_indicator_robustness_excel.py",
            "--output-dir",
            str(stress107_dir),
            "--output-file",
            str(stress107_workbook_path),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert stress107_excel_process.returncode == 0, stress107_excel_process.stderr or stress107_excel_process.stdout
    assert stress107_workbook_path.exists()

    expected_files = {
        "manifest.json",
        "hardware.json",
        "versions.json",
        "predictions/predictions_long.csv",
        "metrics/metrics_by_seed.csv",
        "metrics/metrics_summary.csv",
        "metrics/best_by_experiment_source.csv",
        "metrics/indicator_reconstruction_metrics.csv",
        "metrics/error_by_wqi_band.csv",
        "stats/bootstrap_ci.csv",
        "stats/paired_error_tests.csv",
        "stress_tests/event_window_stress_summary.csv",
        "timing/cpu_only_inference_timing.csv",
        "timing/cpu_only_inference_timing_summary.csv",
        "reports/robustness_tiny.xlsx",
    }
    assert expected_files.issubset(
        {str(path.relative_to(output_dir)) for path in output_dir.rglob("*") if path.is_file()}
    )
    assert len(list((output_dir / "models").rglob("*.joblib"))) == 7

    row_expectations = {
        "predictions/predictions_long.csv": 2400,
        "metrics/metrics_by_seed.csv": 20,
        "metrics/metrics_summary.csv": 20,
        "metrics/best_by_experiment_source.csv": 20,
        "metrics/indicator_reconstruction_metrics.csv": 8,
        "stress_tests/event_window_stress_summary.csv": 30,
        "timing/cpu_only_inference_timing.csv": 10,
        "timing/cpu_only_inference_timing_summary.csv": 10,
    }
    for relative_path, expected_rows in row_expectations.items():
        with (output_dir / relative_path).open("r", encoding="utf-8") as handle:
            assert len(list(csv.DictReader(handle))) == expected_rows

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["missing_sets"] == {
        "missing_bod": ["BOD"],
        "missing_bod_nh3n": ["BOD", "NH3N"],
        "missing_nh3n": ["NH3N"],
    }

    stress107_expected_rows = {
        "stress_tests/stress107_setting.csv": 12,
        "stress_tests/stress107_window_summary.csv": 80,
        "stress_tests/stress107_detection_summary.csv": 20,
        "stress_tests/stress107_severity_monotonicity.csv": 10,
    }
    for relative_path, expected_rows in stress107_expected_rows.items():
        with (stress107_dir / relative_path).open("r", encoding="utf-8") as handle:
            assert len(list(csv.DictReader(handle))) == expected_rows
    stress107_manifest = json.loads((stress107_dir / "manifest.json").read_text(encoding="utf-8"))
    assert stress107_manifest["n_windows"] == 4
    assert stress107_manifest["severities"] == ["low_30pct", "medium_100pct"]
