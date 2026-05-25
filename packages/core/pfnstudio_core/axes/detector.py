"""Auto-detector — supervised model that reads context and proposes
axis values.

The pitch in one line: instead of asking the user to fill out an empty
form of axis chips, the detector reads their data and pre-fills the
chips with its best guess (and a confidence). The user just confirms,
overrides, or adds the axes the detector can't see.

Training:
- Each batch, the prior samples a real (non-UNKNOWN) tag value and
  generates data from it. The tag value is the supervised label.
- The detector reads the data (no tag) and predicts a softmax over
  axis values.
- Cross-entropy loss.

Inference:
- Same encoder forward, output softmax probabilities.
- Highest-probability value becomes the proposed chip; that probability
  becomes the displayed confidence.

Limitations in v1:
- Only categorical axes are detected. Range axes need a regression
  head and a different loss; boolean axes need a 2-class head.
  Both are straightforward additions but defer until they're actually
  shipped on a prior.
- The detector is small (32-d, 2-layer) — same shape as the test
  brain. Scale up when a real prior with many axes ships.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import UNKNOWN, Axis


class AxisDetector:
    """A small encoder + per-axis classification head. Holds one
    head per categorical axis; non-categorical axes are silently
    skipped (an empty detector is legal — predicts nothing)."""

    def __init__(
        self,
        axes: list[Axis],
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.0,
    ):
        try:
            import torch.nn as nn
        except ImportError as e:
            raise ImportError(
                "AxisDetector requires torch. Install with: pip install pfnstudio-core[torch]"
            ) from e

        # Only categorical axes get heads in v1. Skipped axes are
        # tracked so callers can warn the user "we can't propose for
        # axis X yet."
        self._scored_axes: list[Axis] = [
            a for a in axes if a.kind == "categorical" and len(a.values) >= 2
        ]
        self._skipped_axes: list[Axis] = [a for a in axes if a not in self._scored_axes]
        self.d_model = d_model

        # Lazy embedder — input feature dim is inferred on first
        # forward. Same pattern the TabularEmbedder uses.
        self.embedder = nn.LazyLinear(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # One head per scored axis. Output dim = number of values
        # (no UNKNOWN class — the detector proposes a real value;
        # low confidence across all classes is how it expresses
        # "I'm not sure", not a separate class).
        self.heads = nn.ModuleDict(
            {a.name: nn.Linear(d_model, len(a.values)) for a in self._scored_axes}
        )

    @property
    def scored_axes(self) -> list[Axis]:
        """Axes the detector has heads for (categorical, ≥2 values)."""
        return self._scored_axes

    @property
    def skipped_axes(self) -> list[Axis]:
        """Axes the detector can't predict (boolean, range, single-value)."""
        return self._skipped_axes

    def parameters(self):
        # nn.ModuleDict + nn.Module attributes — yield everything.
        yield from self.embedder.parameters()
        yield from self.encoder.parameters()
        yield from self.heads.parameters()

    def state_dict(self) -> dict[str, Any]:
        return {
            "embedder": self.embedder.state_dict(),
            "encoder": self.encoder.state_dict(),
            "heads": self.heads.state_dict(),
        }

    def __call__(self, x: Any) -> dict[str, Any]:
        """Forward pass. Returns axis_name → logits tensor of shape
        (B, num_values_for_axis)."""
        emb = self.embedder(x)  # (B, N, d_model)
        encoded = self.encoder(emb)  # (B, N, d_model)
        # Mean-pool the context tokens to one vector per batch element.
        # (Could swap for an attention pool later; mean is fine for
        # the monotonicity demo and avoids extra parameters.)
        pooled = encoded.mean(dim=1)  # (B, d_model)
        return {name: head(pooled) for name, head in self.heads.items()}


def _value_to_index(axis: Axis, value: Any) -> int:
    """Index of `value` in axis.values. Raises if value is UNKNOWN
    or not a known value — UNKNOWN tags shouldn't be used as training
    labels because the detector is supposed to learn to assign a
    concrete value."""
    if value == UNKNOWN:
        raise ValueError(f"axis {axis.name!r}: cannot train detector on UNKNOWN label")
    return list(axis.values).index(value)


def train_detector(
    *,
    detector: AxisDetector,
    prior: Any,
    steps: int = 500,
    batch_size: int = 16,
    lr: float = 1e-3,
    seed: int = 0,
    on_step: Any = None,
) -> dict[str, Any]:
    """Train the detector on labeled batches sampled from the prior.

    Each batch picks a non-UNKNOWN value for each scored axis, samples
    data with that tag, and trains the detector to predict the tag
    from the data. Cross-entropy loss, summed across axes.

    Returns ``{status, final_loss, mean_acc}`` where ``mean_acc`` is
    the accuracy of the detector on the *last* batch (cheap sanity
    check that training did something)."""
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        return {"status": "skipped", "reason": "torch not installed"}

    if not detector.scored_axes:
        return {"status": "skipped", "reason": "detector has no scored axes"}

    rng = np.random.default_rng(seed)
    optim = torch.optim.AdamW(list(detector.parameters()), lr=lr)

    last_acc: dict[str, float] = {}
    final_loss = 0.0
    for step in range(steps):
        # Sample one tag value per axis (random, non-UNKNOWN — that's
        # the supervised label the detector learns to recover).
        tag = {a.name: rng.choice(list(a.values)) for a in detector.scored_axes}
        batch = prior.sample_batch(batch_size=batch_size, seed=seed + step * batch_size, tag=tag)
        X = torch.stack([torch.from_numpy(b["X"]).float() for b in batch])

        logits = detector(X)  # dict axis → (B, num_values)
        loss = torch.zeros(1)
        for axis in detector.scored_axes:
            label_idx = _value_to_index(axis, tag[axis.name])
            labels = torch.full((X.shape[0],), label_idx, dtype=torch.long)
            loss = loss + F.cross_entropy(logits[axis.name], labels)

        optim.zero_grad()
        loss.backward()
        optim.step()

        final_loss = float(loss.item())
        if on_step is not None:
            on_step(step, final_loss)

        # Final-batch accuracy (per axis) — cheap and informative.
        if step == steps - 1:
            with torch.no_grad():
                for axis in detector.scored_axes:
                    preds = logits[axis.name].argmax(dim=-1)
                    label_idx = _value_to_index(axis, tag[axis.name])
                    last_acc[axis.name] = float((preds == label_idx).float().mean().item())

    return {
        "status": "ok",
        "steps": steps,
        "final_loss": final_loss,
        "last_batch_accuracy": last_acc,
    }


def detect(
    *,
    detector: AxisDetector,
    context: Any,
) -> dict[str, dict[str, Any]]:
    """Run the detector on a context tensor and return a tag-shaped
    proposal with per-axis confidence.

    Returns ``{axis_name: {value, confidence, probs}}`` where:
      - ``value`` is the predicted axis value (argmax)
      - ``confidence`` is the softmax probability of that value
      - ``probs`` is the full {value: probability} dict, for callers
        that want to render uncertainty (e.g. show "60% positive,
        35% mixed, 5% negative" instead of just "positive 60%").
    """
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        return {}

    if not detector.scored_axes:
        return {}

    if not torch.is_tensor(context):
        context = torch.from_numpy(np.asarray(context)).float()
    if context.dim() == 2:
        context = context.unsqueeze(0)  # add batch dim

    with torch.no_grad():
        logits = detector(context)

    out: dict[str, dict[str, Any]] = {}
    for axis in detector.scored_axes:
        probs = F.softmax(logits[axis.name], dim=-1)
        # Mean across batch in case the caller passed multiple contexts —
        # the detector returns one proposal per axis, not per-sample.
        mean_probs = probs.mean(dim=0)
        best_idx = int(mean_probs.argmax().item())
        out[axis.name] = {
            "value": axis.values[best_idx],
            "confidence": float(mean_probs[best_idx].item()),
            "probs": {axis.values[i]: float(mean_probs[i].item()) for i in range(len(axis.values))},
        }
    return out
