import numpy as np

try:
    from pfnstudio_core import Prior, register_prior
except Exception:
    # Fallback only for local testing outside PFN Studio.
    class Prior:
        pass

    def register_prior(name):
        def wrapper(cls):
            return cls
        return wrapper


def sigmoid(z):
    """Stable sigmoid."""
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


class RandomMLP:
    """
    Small random neural network used to create synthetic SCM mechanisms.

    Used for:
    1. Treatment assignment: T = f_T(X, U)
    2. Outcome generation: output = f_Y(X, T, U)
    """

    def __init__(self, rng, input_dim, hidden_dim=32, output_dim=1, num_hidden_layers=2):
        self.weights = []
        self.biases = []

        dims = [input_dim] + [hidden_dim] * num_hidden_layers + [output_dim]

        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            w = rng.normal(
                loc=0.0,
                scale=1.0 / np.sqrt(in_dim),
                size=(in_dim, out_dim),
            )
            b = rng.normal(
                loc=0.0,
                scale=0.2,
                size=(out_dim,),
            )

            self.weights.append(w)
            self.biases.append(b)

    def __call__(self, x):
        h = x

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ w + b

            if i < len(self.weights) - 1:
                h = np.tanh(h)

        return h


def empirical_msm_bound(values, gamma, upper=True):
    """
    Empirical MSM-style bound under a density-ratio constraint.

    We approximate:

        max/min E[r(U) * Y(U)]

    subject to:

        1 / gamma <= r(U) <= gamma
        E[r(U)] = 1

    For upper bound:
        give more weight to high outcome values.

    For lower bound:
        give more weight to low outcome values.

    gamma = 1 means no hidden-confounding shift, so lower = upper = mean.
    """

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    n = values.shape[0]

    if n == 0:
        return 0.0

    if gamma <= 1.000001:
        return float(np.mean(values))

    low_weight = 1.0 / gamma
    high_weight = gamma

    order = np.argsort(values)

    if upper:
        fill_order = order[::-1]  # highest values first
    else:
        fill_order = order        # lowest values first

    weights = np.full(n, low_weight, dtype=np.float64)

    remaining_mass = float(n) - float(weights.sum())
    capacity = high_weight - low_weight

    for idx in fill_order:
        if remaining_mass <= 1e-12:
            break

        add = min(capacity, remaining_mass)
        weights[idx] += add
        remaining_mass -= add

    weights = weights * (float(n) / float(weights.sum()))

    return float(np.mean(weights * values))


@register_prior("causal-sensitivity-msm")
class CausalSensitivityMSMPrior(Prior):
    """
    Paper-style synthetic SCM prior for causal sensitivity analysis.

    Returns:
        X
        T
        output
        x_query
        t_query
        gamma
        lower_bound
        higher_bound
    """

    def sample(
        self,
        seed=None,
        tag=None,
        num_points=256,
        d=10,
        num_query_points=8,
        num_gamma=5,
        gamma_max=3.0,
        mc_samples=256,
        **kwargs,
    ):
        rng = np.random.default_rng(seed)

        num_points = int(num_points)
        d = int(d)
        num_query_points = int(num_query_points)
        num_gamma = int(num_gamma)
        gamma_max = float(gamma_max)
        mc_samples = int(mc_samples)

        # -------------------------------------------------
        # 1. Generate observed covariates X
        # -------------------------------------------------

        X_raw = rng.normal(
            loc=0.0,
            scale=1.0,
            size=(num_points, d),
        )

        # -------------------------------------------------
        # 2. Generate hidden confounder U
        # -------------------------------------------------
        # U is internal only. We do not return it.

        U = rng.normal(
            loc=0.0,
            scale=1.0,
            size=(num_points, 1),
        )

        # -------------------------------------------------
        # 3. Generate binary treatment T
        # -------------------------------------------------
        # T depends on X and hidden U.
        # This creates hidden confounding.

        treatment_mlp = RandomMLP(
            rng=rng,
            input_dim=d + 1,
            hidden_dim=32,
            output_dim=1,
            num_hidden_layers=2,
        )

        treatment_input = np.concatenate([X_raw, U], axis=1)
        treatment_logits = treatment_mlp(treatment_input)
        propensity = sigmoid(treatment_logits)

        T = rng.binomial(
            n=1,
            p=propensity,
        ).astype(np.float64)

        # -------------------------------------------------
        # 4. Generate observed output
        # -------------------------------------------------
        # output depends on X, T, and hidden U.

        outcome_mlp = RandomMLP(
            rng=rng,
            input_dim=d + 2,
            hidden_dim=32,
            output_dim=1,
            num_hidden_layers=2,
        )

        outcome_input = np.concatenate([X_raw, T, U], axis=1)
        output_raw = outcome_mlp(outcome_input)

        output_raw = output_raw + rng.normal(
            loc=0.0,
            scale=0.1,
            size=output_raw.shape,
        )

        # -------------------------------------------------
        # 5. Normalize X and output
        # -------------------------------------------------

        X_mean = X_raw.mean(axis=0, keepdims=True)
        X_std = X_raw.std(axis=0, keepdims=True) + 1e-6
        X = (X_raw - X_mean) / X_std

        output_mean = output_raw.mean(axis=0, keepdims=True)
        output_std = output_raw.std(axis=0, keepdims=True) + 1e-6
        output = (output_raw - output_mean) / output_std

        # -------------------------------------------------
        # 6. Choose query base points from X
        # -------------------------------------------------

        num_query_points = min(num_query_points, num_points)

        query_indices = rng.choice(
            num_points,
            size=num_query_points,
            replace=False,
        )

        base_x_raw = X_raw[query_indices]
        base_x_norm = X[query_indices]

        # -------------------------------------------------
        # 7. Create gamma grid
        # -------------------------------------------------

        if num_gamma == 1:
            gamma_grid = np.array([1.0], dtype=np.float64)
        else:
            gamma_grid = np.linspace(
                1.0,
                gamma_max,
                num_gamma,
                dtype=np.float64,
            )

        # -------------------------------------------------
        # 8. Build query rows and compute labels
        # -------------------------------------------------

        x_query_rows = []
        t_query_rows = []
        gamma_rows = []
        lower_rows = []
        higher_rows = []

        for x_raw, x_norm in zip(base_x_raw, base_x_norm):
            x_raw_2d = x_raw.reshape(1, d)

            for t_val in [0.0, 1.0]:
                for g in gamma_grid:
                    x_query_rows.append(x_norm)
                    t_query_rows.append([t_val])
                    gamma_rows.append([g])

                    # Monte Carlo hidden U samples for this query.
                    U_mc = rng.normal(
                        loc=0.0,
                        scale=1.0,
                        size=(mc_samples, 1),
                    )

                    X_mc = np.repeat(
                        x_raw_2d,
                        mc_samples,
                        axis=0,
                    )

                    T_mc = np.full(
                        shape=(mc_samples, 1),
                        fill_value=t_val,
                        dtype=np.float64,
                    )

                    outcome_mc_input = np.concatenate(
                        [X_mc, T_mc, U_mc],
                        axis=1,
                    )

                    y_mc_raw = outcome_mlp(outcome_mc_input)

                    y_mc = (y_mc_raw - output_mean) / output_std
                    y_mc = y_mc.reshape(-1)

                    lower = empirical_msm_bound(
                        values=y_mc,
                        gamma=g,
                        upper=False,
                    )

                    higher = empirical_msm_bound(
                        values=y_mc,
                        gamma=g,
                        upper=True,
                    )

                    lower_rows.append([lower])
                    higher_rows.append([higher])

        x_query = np.asarray(x_query_rows, dtype=np.float32)
        t_query = np.asarray(t_query_rows, dtype=np.float32)
        gamma = np.asarray(gamma_rows, dtype=np.float32)
        lower_bound = np.asarray(lower_rows, dtype=np.float32)
        higher_bound = np.asarray(higher_rows, dtype=np.float32)

        # -------------------------------------------------
        # 9. Return variables declared in PFN Studio
        # -------------------------------------------------

        return {
            "X": X.astype(np.float32),
            "T": T.astype(np.float32),
            "output": output.astype(np.float32),
            "x_query": x_query.astype(np.float32),
            "t_query": t_query.astype(np.float32),
            "gamma": gamma.astype(np.float32),
            "lower_bound": lower_bound.astype(np.float32),
            "higher_bound": higher_bound.astype(np.float32),
        }
