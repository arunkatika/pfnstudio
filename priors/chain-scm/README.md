# Causal Chain Discovery

`Detect X → Y → Z chains from data alone`

Specialised version of the SCM prior with chain-shaped DAGs. Faster to learn than full ER-DAGs and matches a common scientific use case.

Each task is a random permutation of variables wired as a chain. Useful as a sanity check for any discovery PFN — if it can\

## Try it

```bash
pfnstudio sample priors/chain-scm/prior.yaml --seeds 3
```

## Citations

- `zheng2018notears`

---

_Author: Community_
