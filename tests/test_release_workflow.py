from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]


def test_release_workflow_reads_the_canonical_package_version():
    workflow = (PROJECT_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    assert "from wqsurrogatemodels import __version__; print(__version__)" in workflow
    assert '"project"]["version"]' not in workflow
