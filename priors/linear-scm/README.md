# Linear SCM Discovery

`Recover the DAG behind linear y = Ax + ε`

Random sparse linear structural causal models. The PFN outputs an adjacency matrix — pure structure discovery, no fitted edges.

Erdős–Rényi DAGs with random linear weights. The training target is the {0,1} adjacency. After training the model recovers structure from observational data in one pass — fast alternative to NOTEARS / GES.

## Try it

```bash
pfnstudio sample priors/linear-scm/prior.yaml --seeds 3
```

## Citations

- `zheng2018notears`

---

_Author: PFN Studio Team_
