Faithful port of the label-construction pipeline from Javurek et al.,
*Amortizing Causal Sensitivity Analysis via PFNs* (arXiv:2605.10590).

Each sample() draws one synthetic SCM (DAG-structured covariates, MLP
propensity f_A, MLP outcome f_BNN with a hidden confounder U~N(0,1)) and
computes MSM sensitivity-bound labels θ* via the authors' Lagrangian
scalarization: a rational-quadratic spline flow over U, MSM divergence
γ = max(max r, 1/min r), warm-started descending λ-sweep, and cumulative
monotonicity repair. Because the SCM is synthetic, f_Y is known exactly,
so no first-stage density estimation is needed.

Partial identification: U is never returned — the model must recover the
[lower, upper] bounds on the causal query from (X, A, Y) context alone.
Numerics match the paper's Table 2 (350 steps/λ, k_train=128, k_eval=4096,
16 bins, tail 6.0, 50-point λ grid).