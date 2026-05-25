---
bibkey: mueller2022pfn
title: Transformers Can Do Bayesian Inference
year: 2022
relevance: foundational
---

# Transformers Can Do Bayesian Inference (Müller et al., 2022)

## TL;DR

Train a transformer on samples from a prior `p(D, y)`. At inference, the model approximates the posterior predictive `p(y* | x*, D)` for arbitrary held-out `(x*, y*)`. No fine-tuning needed.

## Why it matters here

This is the foundational PFN paper. Sets up the training-on-priors paradigm that PFN Studio is built around. Read first before contributing priors.

## Key ideas to carry forward

- **Synthetic prior is the dataset.** Real data only appears at inference.
- **Reproducibility via the prior spec.** Different prior = different model, by definition.
- **In-context learning.** Model conditions on `(D, x*)` and predicts `y*` directly.

## Open questions / where this project diverges

- Original work is tabular; we generalize to temporal/causal structure.
- Original priors are simple GP/BNN families; project priors are domain-specific.
