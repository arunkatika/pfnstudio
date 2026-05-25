from pathlib import Path

from pfnstudio.lint import lint_project
from pfnstudio.scaffold import scaffold_project


def test_template_lints_clean(tmp_path: Path):
    target = tmp_path / "demo-fm"
    scaffold_project(
        target=target, project_name="demo-fm", description="A test project.", org="acme"
    )
    errors = lint_project(target)
    assert errors == [], f"template should lint clean, got: {errors}"


def test_lint_catches_unknown_prior_in_run(tmp_path: Path):
    target = tmp_path / "demo-fm"
    scaffold_project(
        target=target, project_name="demo-fm", description="A test project.", org="acme"
    )

    run_yaml = target / "runs" / "example_run.yaml"
    text = run_yaml.read_text().replace("example_linear_scm", "ghost_prior")
    run_yaml.write_text(text)

    errors = lint_project(target)
    assert any("ghost_prior" in e for e in errors)


def test_lint_catches_dir_id_mismatch(tmp_path: Path):
    target = tmp_path / "demo-fm"
    scaffold_project(
        target=target, project_name="demo-fm", description="A test project.", org="acme"
    )

    prior_dir = target / "priors" / "example_linear_scm"
    new_dir = target / "priors" / "renamed"
    prior_dir.rename(new_dir)

    errors = lint_project(target)
    assert any("does not match directory name" in e for e in errors)


def test_lint_catches_unknown_bibkey(tmp_path: Path):
    target = tmp_path / "demo-fm"
    scaffold_project(
        target=target, project_name="demo-fm", description="A test project.", org="acme"
    )

    prior_yaml = target / "priors" / "example_linear_scm" / "prior.yaml"
    text = prior_yaml.read_text()
    text = text.replace("- peters2017elements", "- ghost_paper_2099")
    prior_yaml.write_text(text)

    errors = lint_project(target)
    assert any("ghost_paper_2099" in e for e in errors)
