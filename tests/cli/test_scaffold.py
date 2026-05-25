from pathlib import Path

from pfnstudio.scaffold import scaffold_project


def test_scaffold_creates_expected_layout(tmp_path: Path):
    target = tmp_path / "demo-fm"
    scaffold_project(
        target=target, project_name="demo-fm", description="A test project.", org="acme"
    )

    assert (target / "README.md").exists()
    assert (target / "ROADMAP.md").exists()
    assert (target / "priors" / "example_linear_scm" / "prior.yaml").exists()
    assert (target / "priors" / "example_linear_scm" / "prior.py").exists()
    assert (target / "models" / "example_transformer.yaml").exists()
    assert (target / "evals" / "example_sachs.yaml").exists()
    assert (target / "runs" / "example_run.yaml").exists()
    assert (target / "literature" / "references.bib").exists()


def test_scaffold_substitutes_placeholders(tmp_path: Path):
    target = tmp_path / "demo-fm"
    scaffold_project(
        target=target, project_name="demo-fm", description="A test project.", org="acme"
    )

    readme = (target / "README.md").read_text()
    assert "demo-fm" in readme
    assert "acme" in readme
    assert "{{project_name}}" not in readme
    assert "{{org}}" not in readme
