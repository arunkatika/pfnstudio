"""End-to-end: import the template's example prior via discover_in_project,
then sample from it deterministically."""

from pathlib import Path

import numpy as np
from pfnstudio.scaffold import scaffold_project
from pfnstudio_core.registry import _clear_for_tests, discover_in_project, get_prior


def test_example_prior_samples_reproducibly(tmp_path: Path):
    _clear_for_tests()
    target = tmp_path / "demo-fm"
    scaffold_project(
        target=target, project_name="demo-fm", description="A test project.", org="acme"
    )

    discover_in_project(target)

    prior_cls = get_prior("example_linear_scm")
    prior = prior_cls()

    s1 = prior.sample(seed=42, num_variables=5, num_samples=20)
    s2 = prior.sample(seed=42, num_variables=5, num_samples=20)

    assert s1["X"].shape == (20, 5)
    assert s1["A"].shape == (5, 5)
    assert np.array_equal(s1["X"], s2["X"]), "same seed must produce identical samples"
    assert np.array_equal(s1["A"], s2["A"])
