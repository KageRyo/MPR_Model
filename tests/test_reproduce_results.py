from __future__ import annotations

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
