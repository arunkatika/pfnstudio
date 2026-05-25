# Linear Regression

`y = a·x + b + noise`

The reference PFN prior. Random linear functions with Gaussian noise — the simplest demonstrable PFN training task.

Each task samples a fresh slope and intercept, then N noisy (x, y) pairs. After training, the model performs linear regression on any new dataset in a single forward pass.

## Try it

```bash
pfnstudio sample priors/linear-regression/prior.yaml --seeds 3
```

## Citations

- `mueller2022pfn`

---

_Author: PFN Studio Team_
