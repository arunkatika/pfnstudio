# Cascade discovery (Tennessee-Eastman class)

A runnable causal **structure-discovery** study. It trains an in-context PFN to
recover the directed graph behind feed-forward, layer-partitioned SCMs that
mimic Tennessee-Eastman-class process topology (feeds → reactor → separator →
product, 0 cycles, V = 52) — the structure that Erdős–Rényi priors
systematically miss.

## What's inside

| Artifact | File |
|---|---|
| **Prior** | `priors/cascade_scm/` — feed-forward layered DAGs, gap-decayed inter-layer edges |
| **Model** | `models/cascade_discovery.yaml` — `tabular_embedder → transformer_encoder ×3 → causal_attention_pool → discovery_head` |
| **Run** | `runs/v0_1.yaml` — pins `d=52`, BCE on the adjacency, 8 000 steps, AdamW |
| **Eval** | `evals/cascade_recovery.yaml` — AUROC / SHD / F1 vs the prior's ground-truth `A` |

## How it wires together

The run sets `prior.overrides.d = 52` so every sampled task has a `(52, 52)`
adjacency. This matters: the trainer **skips** discovery batches whose `A`
shapes differ (`loop.py` variable-K guard), so a uniform `d` is what lets the
discovery head actually learn. The `discovery_head`'s `num_variables` is
auto-aligned to the prior's `d` at run time, so the model and prior stay in
sync without hand-editing.

## Run it

Scaffold/import this template into a project, then launch `v0_1`. A ~25-step
CPU smoke run already confirms the path end-to-end: discovery loss falls
(0.69 → 0.12) and `cascade_recovery` scores **AUROC ≈ 0.93** on 30 fresh draws.

## Scope note — what the eval measures today

`cascade_recovery` is scored by `TemporalPairwiseDiscovery`, whose **Phase 1**
edge-score source is a model-independent lagged-Pearson baseline. So the AUROC
reported today measures *whether the cascade structure is recoverable from the
data* — not yet whether this trained model recovers it. Swapping the score
source to the model's pairwise forward pass is the scorer's documented Phase 2.
The training above is real (BCE against `A`); it just isn't yet what produces
the eval number. SHD/F1/precision read low because the scorer thresholds at the
median score, which over-predicts edges on a sparse 52-variable graph — AUROC
is the threshold-free metric to read.
