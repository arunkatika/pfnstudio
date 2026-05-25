from pathlib import Path

from pfnstudio.scaffold import scaffold_project
from pfnstudio.validate import validate_project


def test_template_validates_clean(tmp_path: Path):
    target = tmp_path / "demo-fm"
    scaffold_project(
        target=target, project_name="demo-fm", description="A test project.", org="acme"
    )
    errors = validate_project(target)
    assert errors == [], f"template should validate cleanly, got: {errors}"


def test_validate_catches_missing_required_field(tmp_path: Path):
    target = tmp_path / "demo-fm"
    scaffold_project(
        target=target, project_name="demo-fm", description="A test project.", org="acme"
    )

    bad = target / "priors" / "example_linear_scm" / "prior.yaml"
    bad.write_text("name: still here\nversion: 0.1.0\n")

    errors = validate_project(target)
    assert any("required" in e.lower() or "id" in e for e in errors)
