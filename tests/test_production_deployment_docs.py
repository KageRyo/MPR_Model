from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]


def test_production_deployment_guide_covers_operator_workflow():
    guide = (PROJECT_ROOT / "docs" / "production-backend-deployment.md").read_text(encoding="utf-8")

    for heading in (
        "## Choose a Runtime",
        "## 1. Prepare Configuration and Local Resources",
        "## 2. Run with Docker Compose",
        "## 3. Run Directly with `uv`",
        "## 4. Verify Health, Readiness, and Assessment",
        "## 5. Connect WaterMirror",
        "## Troubleshooting",
    ):
        assert heading in guide

    for required_value in (
        "LOCAL_MODEL_DIR",
        "LOCAL_DATA_DIR",
        "CORS_ALLOW_ORIGINS",
        "INSTALL_MODEL_EXTRAS=true",
        "/api/v2/health",
        "/api/v2/ready",
        "X-Request-ID",
    ):
        assert required_value in guide


def test_readme_links_to_production_deployment_guide():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "[Production Backend Deployment](docs/production-backend-deployment.md)" in readme
