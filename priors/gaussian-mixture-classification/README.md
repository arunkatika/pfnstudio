# Gaussian-Mixture Classification

`Per-task GMM with K components → K-class labels`

Random Gaussian mixtures in D dimensions. Trains a PFN that does Bayesian-optimal classification on any well-separated mixture.

Sample K random means + covariances per task, draw N points by component, assign labels by component index. Higher-D extension of two-moons; closer to real tabular classification.

## Try it

```bash
pfnstudio sample priors/gaussian-mixture-classification/prior.yaml --seeds 3
```

## Citations

- `mueller2022pfn`

---

_Author: PFN Studio Team_
