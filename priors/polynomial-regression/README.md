# Polynomial Regression

`y = Σ c_k · x^k + noise`

Random polynomial functions up to degree D. Generalises linear regression with curvature; the model learns to fit smooth nonlinear shapes in-context.

Sample polynomial coefficients up to a max degree, evaluate on N points with Gaussian noise. Good first step before kernel methods.

## Try it

```bash
pfnstudio sample priors/polynomial-regression/prior.yaml --seeds 3
```

## Citations

- `mueller2022pfn`

---

_Author: PFN Studio Team_
