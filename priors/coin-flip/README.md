# Bayesian Coin Flip

`p ~ Beta(α, β); flips ~ Bernoulli(p)`

Textbook conjugate prior. The PFN learns to approximate the closed-form posterior mean (α + heads) / (α + β + N).

Direct evidence that the model has learned Bayesian inference end-to-end — compare the predicted bias to the analytical posterior on held-out flips.

## Try it

```bash
pfnstudio sample priors/coin-flip/prior.yaml --seeds 3
```

## Citations

- `mueller2022pfn`

---

_Author: PFN Studio Team_
