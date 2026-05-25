"""Compute per-axis honoring scores for a trained tag-aware model.

For each axis, the score answers: *"when I hold context constant and
flip the tag, does the model's prediction actually change?"* A high
score means the tag is influencing the model output; near-zero means
the tag is being silently dropped — which would be the worst possible
failure mode (the chip ships in the UI but nothing happens).

Output shape per axis:
    {
        "divergence_mean": float,    # mean L2 distance across batch
        "divergence_max": float,     # worst case
        "values_compared": (a, b),   # which two tag values we flipped
        "n_samples": int,
    }

Limitations in v1:
- Only categorical axes with ≥2 values are scored. Range/boolean
  axes need their own divergence strategy (TBD when the first axis
  of each kind ships).
- We compare the first two values listed on the axis. Multi-pair
  comparison (e.g. positive↔negative AND positive↔mixed) can be
  added if the first-pair signal turns out to be weak.
- The synthetic context comes from ``prior.sample_batch(tag=None)``
  — the *unconstrained* distribution. That's the right baseline:
  the divergence then measures pure tag effect, not tag+data drift.
"""

from __future__ import annotations

from typing import Any

from .base import UNKNOWN, Axis
from .encoding import encode_tag


def compute_axis_honoring(
    *,
    model: Any,
    prior: Any,
    axes: list[Axis],
    batch_size: int = 16,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """Run two forward passes per axis (different tag values) and
    measure how much the model's prediction changes.

    Returns ``{axis_name: {divergence_mean, divergence_max,
    values_compared, n_samples}}`` for every categorical axis with at
    least two distinct values. Axes that aren't scoreable (boolean,
    range, or categorical with one value) are silently skipped — the
    runner sees an empty entry rather than a crash.

    Skips entirely when the model has no tag-aware blocks; returns an
    empty dict so callers can branch on truthiness.
    """
    try:
        import torch
    except ImportError:
        return {}

    if not axes:
        return {}

    # Find the first tag-aware encoder block in the model. If none
    # exists, honoring is meaningless — the model can't condition on
    # the tag regardless of what we feed it.
    has_tag_aware = any(
        getattr(mod, "tag_embedder", None) is not None for _, mod in getattr(model, "modules", [])
    )
    if not has_tag_aware:
        return {}

    # Sample one fixed-context batch. tag=None means UNKNOWN-or-skip
    # at the prior level, giving us the unconstrained data
    # distribution — the right baseline for measuring pure tag effect.
    batch = prior.sample_batch(batch_size=batch_size, seed=seed)
    if not batch:
        return {}

    X = torch.stack([torch.from_numpy(b["X"]).float() for b in batch])

    results: dict[str, dict[str, Any]] = {}

    # We forward through the same module chain step_fn uses, but
    # without computing a loss. Reuse the encoder/heads split helper
    # via a local import to avoid a circular dependency.
    from ..training.loop import _is_head_module

    modules = list(getattr(model, "modules", []))
    encoder: list[Any] = []
    heads: list[Any] = []
    in_heads = False
    for _, mod in modules:
        if in_heads or _is_head_module(mod):
            in_heads = True
            heads.append(mod)
        else:
            encoder.append(mod)

    def _forward(tag_dict: dict[str, Any]) -> Any:
        """Run the encoder + each head with the given tag dict.
        Returns the first head output (or encoder output if heads is empty)."""
        enc_np = encode_tag(tag_dict, axes)
        tag_tensor = torch.from_numpy(enc_np).float().unsqueeze(0).repeat(len(batch), 1)
        enc_out = X
        for mod in encoder:
            if getattr(mod, "tag_embedder", None) is not None:
                enc_out = mod(enc_out, tag=tag_tensor)
            else:
                enc_out = mod(enc_out)
        if not heads:
            return enc_out
        return heads[0](enc_out)

    for axis in axes:
        if axis.kind != "categorical" or len(axis.values) < 2:
            continue
        value_a, value_b = axis.values[0], axis.values[1]

        # Build tag dicts that are identical except for this one axis,
        # so the divergence isolates the effect of this axis alone.
        tag_a = {a.name: UNKNOWN for a in axes}
        tag_b = dict(tag_a)
        tag_a[axis.name] = value_a
        tag_b[axis.name] = value_b

        with torch.no_grad():
            pred_a = _forward(tag_a)
            pred_b = _forward(tag_b)

        # L2 distance per sample, then mean and max across the batch.
        diff = (pred_a - pred_b).reshape(pred_a.shape[0], -1)
        l2 = torch.linalg.norm(diff, dim=1)
        results[axis.name] = {
            "divergence_mean": float(l2.mean().item()),
            "divergence_max": float(l2.max().item()),
            "values_compared": (value_a, value_b),
            "n_samples": len(batch),
        }

    return results
