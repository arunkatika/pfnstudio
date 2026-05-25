# Hierarchical Normal

`μ_g ~ N(μ_0, τ); y ~ N(μ_g, σ)`

Two-level normal model. Groups share information through a population mean — the canonical multi-level / mixed-effects setup.

The PFN learns partial-pooling automatically: small groups borrow strength from the population, large groups stay near their group mean.

## Try it

```bash
pfnstudio sample priors/hierarchical-normal/prior.yaml --seeds 3
```

## Citations

- `gelman2013bda`

---

_Author: Community_
