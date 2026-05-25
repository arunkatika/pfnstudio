# Logistic w/ Feature Interactions

`σ(w·x + interactions) with random weights`

Logistic regression with pairwise feature interactions baked in. Trains a PFN that picks up cross-feature signal automatically.

Random main effects + a sparse random interaction matrix. The model has to learn that two features only become predictive together — harder than linear logistic.

## Try it

```bash
pfnstudio sample priors/logistic-interactions/prior.yaml --seeds 3
```

## Citations

- `mueller2022pfn`

---

_Author: Community_
