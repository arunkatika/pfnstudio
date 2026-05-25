# GP Regression (RBF)

`y ~ GP(0, k_RBF(x, x\`

Functions sampled from a Gaussian Process with an RBF kernel. The model learns to do GP regression at inference without solving the kernel system.

Per-task random lengthscale and signal variance. The PFN approximates the GP posterior in one forward pass — orders-of-magnitude faster than the closed-form solve at long context.

## Try it

```bash
pfnstudio sample priors/gp-regression/prior.yaml --seeds 3
```

## Citations

- `mueller2022pfn`
- `rasmussen2006gp`

---

_Author: Müller et al._
