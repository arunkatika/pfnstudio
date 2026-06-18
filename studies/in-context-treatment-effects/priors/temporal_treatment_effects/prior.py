"""Temporal treatment-effects prior — synthetic CATE generator with AR baseline.

Reference temporal-causal prior shipped with pfnstudio. Six effect shapes
(immediate / gradual / decay / permanent / delayed / oscillating) plus an
AR(p) baseline with unit-specific level + trend. Treatment is assigned via
a propensity model that depends on the static covariates, so the resulting
tasks are confounded by construction. Ground-truth potential outcomes
``E_y0_query`` and ``E_y1_query`` are emitted so a causal-judgment head
can be supervised on identifiability and effect direction.

Runs on CPU; the output dict is numpy at the wrapper boundary so the
default training step can ``torch.from_numpy(...)`` each field.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from pfnstudio_core import Prior, register_prior

EFFECT_TYPES = ["immediate", "gradual", "decay", "permanent", "delayed", "oscillating"]


def _generate_effect_shape(
    effect_idx: int,
    T_post: int,
    device: torch.device,
    rng: torch.Generator,
) -> torch.Tensor:
    """Generate effect shape on the given device."""
    t = torch.arange(T_post, device=device, dtype=torch.float32)

    if effect_idx == 0:  # immediate
        return torch.ones(T_post, device=device)
    if effect_idx == 1:  # gradual
        tau = (
            torch.empty(1, device=device).uniform_(2.0, max(3.0, T_post / 2), generator=rng).item()
        )
        return 1.0 - torch.exp(-t / tau)
    if effect_idx == 2:  # decay
        tau = (
            torch.empty(1, device=device).uniform_(2.0, max(3.0, T_post / 2), generator=rng).item()
        )
        return torch.exp(-t / tau)
    if effect_idx == 3:  # permanent
        ramp = max(1, T_post // 4)
        return torch.clamp(t / ramp, 0.0, 1.0)
    if effect_idx == 4:  # delayed
        delay = max(1, T_post // 3)
        tau = max(1.0, T_post / 3)
        return torch.where(t < delay, torch.zeros_like(t), 1.0 - torch.exp(-(t - delay) / tau))
    # oscillating
    freq = torch.empty(1, device=device).uniform_(0.3, 1.5, generator=rng).item()
    tau = max(3.0, T_post / 2)
    shape = torch.exp(-t / tau) * torch.cos(freq * t)
    max_abs = shape.abs().max().clamp(min=1e-8)
    return shape / max_abs


class _TemporalTreatmentEffectsCore:
    """Generator core. Single ``sample_one`` call yields one synthetic task."""

    def __init__(
        self,
        min_T_pre: int = 5,
        max_T_pre: int = 50,
        min_T_post: int = 3,
        max_T_post: int = 30,
        min_features: int = 3,
        max_features: int = 10,
        max_num_features: int = 100,
        noise_scale: float = 0.3,
        confounding_strength: float = 1.0,
        treatment_history_len: int = 3,
        max_treatment_dims: int = 1,
        max_n_outcomes: int = 1,
        seed: int = 42,
    ):
        self.min_T_pre = min_T_pre
        self.max_T_pre = max_T_pre
        self.min_T_post = min_T_post
        self.max_T_post = max_T_post
        self.min_features = min_features
        self.max_features = max_features
        self.max_num_features = max_num_features
        self.noise_scale = noise_scale
        self.confounding_strength = confounding_strength
        self.treatment_history_len = treatment_history_len
        self.max_treatment_dims = max_treatment_dims
        self.max_n_outcomes = max_n_outcomes
        self.seed = seed

    def sample_one(
        self,
        n_context_units: int,
        n_query_units: int,
        device: torch.device | str = "cpu",
        seed: int | None = None,
    ) -> dict[str, Any]:
        dev = torch.device(device) if isinstance(device, str) else device
        rng = torch.Generator(device=dev)
        rng.manual_seed(seed if seed is not None else self.seed)

        N = n_context_units + n_query_units
        D = torch.randint(
            self.min_features, self.max_features + 1, (1,), generator=rng, device=dev
        ).item()
        T_pre = torch.randint(
            self.min_T_pre, self.max_T_pre + 1, (1,), generator=rng, device=dev
        ).item()
        T_post = torch.randint(
            self.min_T_post, self.max_T_post + 1, (1,), generator=rng, device=dev
        ).item()
        T_total = T_pre + T_post
        ar_order = torch.randint(1, 6, (1,), generator=rng, device=dev).item()

        X = torch.randn(N, D, device=dev, generator=rng)

        w_level = torch.randn(D, device=dev, generator=rng) * 0.5
        w_trend = torch.randn(D, device=dev, generator=rng) * 0.02
        unit_level = X @ w_level
        unit_trend = X @ w_trend

        noise_scale = torch.empty(1, device=dev).uniform_(0.1, 0.5, generator=rng).item()
        noise = torch.randn(N, T_total, device=dev, generator=rng) * noise_scale

        t_range = torch.arange(T_total, device=dev, dtype=torch.float32)
        baseline = unit_level.unsqueeze(1) + unit_trend.unsqueeze(1) * t_range.unsqueeze(0)

        ar_coeffs = torch.randn(ar_order, device=dev, generator=rng) * 0.3
        ar_coeffs = ar_coeffs / (ar_coeffs.abs().sum().clamp(min=1.0) / 0.9)

        Y_cf0 = baseline + noise
        for t in range(ar_order, T_total):
            ar_sum = torch.zeros(N, device=dev)
            for j in range(ar_order):
                ar_sum = ar_sum + ar_coeffs[j] * (Y_cf0[:, t - j - 1] - unit_level)
            Y_cf0[:, t] = baseline[:, t] + ar_sum + noise[:, t]

        effect_idx = torch.randint(0, 6, (1,), generator=rng, device=dev).item()
        effect_type = EFFECT_TYPES[effect_idx]
        effect_shape = _generate_effect_shape(effect_idx, T_post, dev, rng)

        w_effect = torch.randn(D, device=dev, generator=rng)
        effect_magnitude = X @ w_effect
        y_std = Y_cf0[:, :T_pre].std().clamp(min=1e-8)
        effect_magnitude = (
            effect_magnitude / effect_magnitude.abs().std().clamp(min=1e-8) * y_std * 0.5
        )

        true_cate = effect_magnitude.unsqueeze(1) * effect_shape.unsqueeze(0)

        Y_cf1 = Y_cf0.clone()
        Y_cf1[:, T_pre:] = Y_cf1[:, T_pre:] + true_cate

        w_prop = torch.randn(D, device=dev, generator=rng) * self.confounding_strength
        prop_logit = X @ w_prop
        prop_logit = (prop_logit - prop_logit.mean()) / prop_logit.std().clamp(min=1e-8)
        propensity = torch.sigmoid(prop_logit).clamp(0.05, 0.95)
        treatments = torch.bernoulli(propensity, generator=rng)

        Y_obs = torch.where(treatments.unsqueeze(1) == 1, Y_cf1, Y_cf0)

        perm = torch.randperm(N, device=dev, generator=rng)
        ctx_idx = perm[:n_context_units]
        qry_idx = perm[n_context_units:]

        L = self.treatment_history_len
        M = 1
        treatment_slots = M * (1 + L)

        n_ctx = n_context_units
        X_ctx = X[ctx_idx]
        T_ctx = treatments[ctx_idx]
        Y_ctx = Y_obs[ctx_idx]

        ctx_features = torch.zeros(n_ctx, T_total, self.max_num_features, device=dev)
        phase_mask = (t_range >= T_pre).float()

        ctx_features[:, :, 0] = T_ctx.unsqueeze(1) * phase_mask.unsqueeze(0)

        for lag in range(1, L + 1):
            shifted_mask = torch.zeros(T_total, device=dev)
            if lag < T_total:
                shifted_mask[lag:] = phase_mask[:-lag]
            ctx_features[:, :, M * lag] = T_ctx.unsqueeze(1) * shifted_mask.unsqueeze(0)

        D_eff = min(D, self.max_num_features - treatment_slots)
        ctx_features[:, :, treatment_slots : treatment_slots + D_eff] = X_ctx[:, :D_eff].unsqueeze(
            1
        )

        ctx_features = ctx_features.reshape(n_ctx * T_total, self.max_num_features)
        ctx_outcomes = Y_ctx.reshape(n_ctx * T_total)
        ctx_delta_t = (t_range - T_pre).repeat(n_ctx)
        ctx_phase = (t_range >= T_pre).long().repeat(n_ctx)
        ctx_t = treatments[ctx_idx].repeat_interleave(T_total)

        n_qry = n_query_units
        X_qry = X[qry_idx]

        qry_features = torch.zeros(n_qry, T_post, self.max_num_features, device=dev)
        post_offsets = torch.arange(T_post, device=dev, dtype=torch.float32)

        qry_features[:, :, 0] = 1.0
        for lag in range(1, L + 1):
            if lag < T_post:
                qry_features[:, lag:, M * lag] = 1.0

        D_eff = min(D, self.max_num_features - treatment_slots)
        qry_features[:, :, treatment_slots : treatment_slots + D_eff] = X_qry[:, :D_eff].unsqueeze(
            1
        )

        qry_features = qry_features.reshape(n_qry * T_post, self.max_num_features)
        qry_delta_t = post_offsets.repeat(n_qry)
        qry_phase = torch.ones(n_qry * T_post, dtype=torch.long, device=dev)

        E_y0 = Y_cf0[qry_idx, T_pre:].reshape(n_qry * T_post)
        E_y1 = Y_cf1[qry_idx, T_pre:].reshape(n_qry * T_post)

        ctx_elapsed = torch.ones(n_ctx * T_total, device=dev)
        qry_elapsed = torch.ones(n_qry * T_post, device=dev)

        return {
            "X_context": ctx_features,
            "t_context": ctx_t,
            "y_context": ctx_outcomes,
            "delta_t_context": ctx_delta_t,
            "phase_context": ctx_phase,
            "X_query": qry_features,
            "delta_t_query": qry_delta_t,
            "phase_query": qry_phase,
            "E_y0_query": E_y0,
            "E_y1_query": E_y1,
            "elapsed_t_context": ctx_elapsed,
            "elapsed_t_query": qry_elapsed,
            "metadata": {
                "T_pre": int(T_pre),
                "T_post": int(T_post),
                "D": int(D),
                "K": 1,
                "M": M,
                "effect_type": effect_type,
                "treatment_type": "binary",
                "treatment_mode": "single",
                "n_context_units": n_context_units,
                "n_query_units": n_query_units,
                "has_missing_data": False,
            },
        }


def _to_numpy(v: Any) -> Any:
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().numpy()
    return v


@register_prior("temporal_treatment_effects")
class TemporalTreatmentEffectsPrior(Prior):
    """pfnstudio Prior wrapping the generator core.

    Each ``sample(seed=...)`` call constructs a fresh core instance with
    that seed and pulls one task. The trainer's default ``sample_batch``
    iterates seed=base+i over the batch.
    """

    def sample(
        self,
        *,
        seed: int,
        min_T_pre: int = 5,
        max_T_pre: int = 50,
        min_T_post: int = 3,
        max_T_post: int = 30,
        min_features: int = 3,
        max_features: int = 10,
        max_num_features: int = 100,
        noise_scale: float = 0.3,
        confounding_strength: float = 1.0,
        treatment_history_len: int = 3,
        n_context_units: int = 10,
        n_query_units: int = 5,
        **_: Any,
    ) -> dict[str, Any]:
        core = _TemporalTreatmentEffectsCore(
            min_T_pre=min_T_pre,
            max_T_pre=max_T_pre,
            min_T_post=min_T_post,
            max_T_post=max_T_post,
            min_features=min_features,
            max_features=max_features,
            max_num_features=max_num_features,
            noise_scale=noise_scale,
            confounding_strength=confounding_strength,
            treatment_history_len=treatment_history_len,
            seed=seed,
        )
        torch_batch = core.sample_one(
            n_context_units=n_context_units,
            n_query_units=n_query_units,
            device="cpu",
            seed=seed,
        )
        # Convert tensors → numpy at the wrapper boundary so the default
        # training step can `torch.from_numpy(...)` them when stacking.
        np_seed = np.random.default_rng(seed)  # noqa: F841 — placeholder if any np ops added later
        return {k: _to_numpy(v) for k, v in torch_batch.items()}
