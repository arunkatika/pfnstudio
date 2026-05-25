from pathlib import Path

from pfnstudio.scaffold import scaffold_project
from pfnstudio_core.loaders import load_eval, load_model, load_prior, load_project, load_run


def test_load_template_artifacts(tmp_path: Path):
    target = tmp_path / "demo-fm"
    scaffold_project(
        target=target, project_name="demo-fm", description="A test project.", org="acme"
    )

    prior = load_prior(target / "priors" / "example_linear_scm" / "prior.yaml")
    assert prior.id == "example_linear_scm"
    assert "num_variables" in prior.parameters

    model = load_model(target / "models" / "example_transformer.yaml")
    assert model.id == "example_transformer"
    assert any(b.type == "transformer_encoder" for b in model.blocks)

    ev = load_eval(target / "evals" / "example_sachs.yaml")
    assert ev.id == "example_sachs"
    assert ev.task == "discovery"

    run = load_run(target / "runs" / "example_run.yaml")
    assert run.prior.id == "example_linear_scm"
    assert run.compute.target == "vast"


def test_load_project_aggregates(tmp_path: Path):
    target = tmp_path / "demo-fm"
    scaffold_project(
        target=target, project_name="demo-fm", description="A test project.", org="acme"
    )

    project = load_project(target)
    assert len(project["priors"]) == 1
    assert len(project["models"]) == 1
    assert len(project["evals"]) == 1
    assert len(project["runs"]) == 1
