# Why this prior

The paper's central claim is that a PFN trained on a *family* of SCMs learns to amortize CID prediction — one forward pass replaces per-instance optimization, no causal graph required. v0.1 ships the simplest faithful version of that family: MLP-SCMs with confounded propensity + outcome, block-correlated X, random per-task confounding strength. The block correlation matters because real-world covariates are correlated; training across a distribution of correlation regimes is one of the paper's stated robustness levers.
