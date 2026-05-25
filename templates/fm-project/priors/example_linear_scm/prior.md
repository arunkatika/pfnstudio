# example_linear_scm

## Why this prior

A minimal, well-understood prior for smoke-testing the training pipeline. Linear Gaussian SCMs are identifiable in the observational case under faithfulness assumptions, so a model trained on this prior should at least learn to recover the structure.

Replace this with the real prior for your project — typically more complex and tailored to the domain (temporal, non-linear, latent confounders, etc.).

## Identifiability notes

- DAGs are sampled as upper-triangular adjacency, so topological order is fixed.
- Coefficients bounded away from zero (|w| ≥ 0.1) to ensure faithful edges.
- Noise scale is per-variable Gaussian; equal scales are *not* enforced.

## Known limitations

- No latent confounders.
- Only linear relationships.
- Observational only (no interventions).
