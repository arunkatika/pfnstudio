# Refactor: regression PFNs → real in-context learning

## Why this is happening

The `/benchmarks` page is positioned as a reproducibility leaderboard,
but six of seven paper-backed templates **cannot actually reproduce
their papers' headline claims** today — not because of missing data,
but because the catalog's regression PFNs don't do in-context learning
at all.

The current state, in the trainer's own words ([predict.py:252-258](../packages/core/pfnstudio_core/training/predict.py#L252-L258)):

> "Today's `_default_step` trains the model on per-task batches of
> (X, y). For a PFN, inference is supposed to attend over the context
> and emit predictions for the query. Until the block library has a
> real ICL head, we approximate by running the model on the query x
> directly and returning its per-point output."

Every regression prior in the catalog (`bayesian_linear`,
`lc_pfn_curves`, `ifbo_curves`, `kinpfn_fpt`, `pfns4bo_gp`,
`tabpfn_ts`, plus the simpler `linear_regression`, `sine_wave`,
`ar2_process`) emits `(X, y)` without a context/query split. The
trainer treats each task as plain MSE regression over the whole
sequence. The model learns the **prior's average** function rather
than learning to *use observed (x, y) pairs to predict at new x*.

For the leaderboard to be honest, we need PFN-style ICL on the
regression path. The classification path already has it
([`pfns_classification`](../apps/web/src/app/projects/templates.ts) at
the bottom of `templates.ts`) — packed tokens, `n_ctx` field, trainer
slices query positions. We extend the same convention to regression.

## What "real ICL" looks like

The token layout `pfns_classification` already uses:

    each token = (features..., target_or_zero, is_context_flag)
    context tokens: real target value, is_context = 1
    query tokens:   target = 0,        is_context = 0
    per-token dim  = F + 2
    sequence shape = (N, F+2)

Plus a sibling `n_ctx` field on the task dict so the trainer knows
where to slice. Loss is computed only at query positions
(`pred[:, n_ctx:]`). Targets stored as `y[n_query]`, not `y[N]`.

This is the canonical PFN format (Müller 2022 §3). It's also what the
trainer already understands for the regression branch — see
[loop.py:191-200](../packages/core/pfnstudio_core/training/loop.py#L191-L200).
The infrastructure is there; the priors don't speak it.

## Scope — affected files

### Priors (templates.ts, embedded Python)

| Prior slug | Template | Current shape | Verdict |
|---|---|---|---|
| `linear_regression` | linear-regression | `X (N,1)` + `y (N,)` | refactor |
| `sine_wave` | sine-wave | `t (N,1)` + `y (N,)` | refactor |
| `ar2_process` | ar2-forecasting | `X (N,1)` + `y (N,)` | refactor |
| `bayesian_linear` | pfns-reference | `X (N,1)` + `y (N,)` | **canary** |
| `lc_pfn_curves` | lc-pfn | `X (N,1)` + `y (N,)` | refactor |
| `ifbo_curves` | ifbo | `X (N,F)` + `y (N,)` + extras | refactor |
| `kinpfn_fpt` | kinpfn | `X (N,1)` + `y (N,)` | refactor |
| `pfns4bo_gp` | pfns4bo | `X (N,F)` + `y (N,)` | refactor |
| `tabpfn_ts` | tabpfn-ts | `X (N,F)` + `y (N,)` | refactor |

### Inference

- [`predict.py:241-274`](../packages/core/pfnstudio_core/training/predict.py#L241-L274)
  — the regression branch that explicitly admits it doesn't use
  context today. Fix: pack `(x_ctx, y_ctx, is_ctx=1)` + `(x_qry, 0, 0)`
  into one sequence, run the encoder once, return predictions at query
  positions.

### Scorers

- [`synthetic_regression_mse.py`](../packages/core/pfnstudio_core/scorers/synthetic_regression_mse.py)
  — currently calls `prior.sample()` and assumes `(X, y)` without
  `n_ctx`. Update to consume the new packed-token shape.
- [`in_context_regression_ols.py`](../packages/core/pfnstudio_core/scorers/in_context_regression_ols.py)
  — same.
- [`m4_monthly_forecast.py`](../packages/core/pfnstudio_core/scorers/m4_monthly_forecast.py)
  — builds its own feature rows and runs the model directly. Update
  to pack (lag-features, observed_y, is_ctx) so the model actually
  attends to history.

### Trainer

- [`loop.py:191-200`](../packages/core/pfnstudio_core/training/loop.py#L191-L200)
  — already supports `n_ctx` slicing for regression. **No changes
  needed.** The infrastructure was waiting for priors to use it.

### Model specs

- TabularEmbedder uses `nn.LazyLinear` so it auto-infers the new
  input dim on first forward. **No model-spec changes needed** in
  principle — but the catalog `inputShape` annotations should be
  updated for honesty (e.g. `(B, N, 1)` → `(B, N, 3)` for
  bayesian_linear).

### Wizard

- [`wizard-submit.service.ts`](../apps/web/src/app/teacher/wizard-submit.service.ts)
  — may construct synthetic-prior runs whose prior code is generated
  inline rather than pulled from the catalog. Audit for shape
  assumptions during the rollout.

### Templates page

- For each refactored prior, bump its catalog version (e.g.
  `bayesian_linear@0.1.0` → `0.2.0`). Old runs keep working with
  their pinned version; new runs get the ICL version.

## Sequencing

### Phase 1 — Trainer verification (no code changes) ✅ DONE

Canary test at [`tests/core/test_regression_icl_canary.py`](../tests/core/test_regression_icl_canary.py)
defines an inline packed-token bayesian-linear prior, trains for 400
steps, verifies:
  1. Training ran to completion.
  2. Loss decreased substantially (final < 0.5 × initial).
  3. The trained model beats the context-mean baseline by 30%+ on a
     fresh task — proving it actually attends to context.

**All three pass.** The trainer's regression branch
([loop.py:191-200](../packages/core/pfnstudio_core/training/loop.py#L191-L200))
already supports packed-token ICL — no trainer surgery needed.

**Phase 1 finding: fixed n_ctx within a batch.** Variable n_ctx
across tasks breaks the trainer's `torch.stack([b["y"] for b in batch])`
call because query tensors have different shapes. `pfns_classification`
uses fixed 75% n_ctx for the same reason. Phase 2 will match that
convention; revisiting variable n_ctx (via padding + masked loss)
is left as a future enhancement that's not load-bearing for the
leaderboard claim.

### Phase 2 — bayesian_linear canary (~1-2 days)
- Refactor `bayesian_linear` prior in `templates.ts` to emit packed
  tokens. Bump to `bayesian_linear@0.2.0`.
- Update `closed_form_baseline` scorer (or `synthetic_regression_mse`)
  to construct context+query input properly.
- Fix the regression branch of `predict.py` end-to-end.
- Train a new pfns-reference brain. Verify it matches the closed-form
  posterior to within tolerance (RMSE-vs-analytic ≈ 0). **This is the
  cleanest possible canary** because the verdict is analytical.

### Phase 3 — Roll out to other regression priors (~2-3 days each)
Order by leaderboard value:
1. **kinpfn_fpt** — unblocks the deferred parquet + Table 7 verdict.
2. **lc_pfn_curves** — unblocks log-likelihood metric work.
3. **ifbo_curves** — already has Table 1 baselines populated.
4. **pfns4bo_gp** — analytical GP-mean check works now; HPO-B comes later.
5. **tabpfn_ts** — model now respects context; M4 scorer becomes meaningful.
6. **linear_regression, sine_wave, ar2_process** — simpler priors, get
   them along the way.

Each prior bump:
- Update embedded Python in `templates.ts`
- Bump `seedPriors[0].version` → `0.2.0`
- Bump `seedRuns[0].priorRef.version` to match
- Update audit-notes if relevant

### Phase 4 — Migration shim (~1 day)
- Old brains on disk reference `bayesian_linear@0.1.0`. They keep
  working — versions are immutable.
- New brains use `@0.2.0`. The catalog's wizard always pins the latest
  version, so new flow is automatic.
- Document the version split in CONTRIBUTING.md or a migration note.

### Phase 5 — Re-enable the KinPFN pilot (~1 day)
- Wire `kinpfn-real-fpt` registry entry to the parquet (URL + sha256
  from `manifest.json`).
- Implement `kinpfn_real_fpt_ks` scorer using the now-functional ICL
  context.
- Update `KINPFN_TEMPLATE` eval slug + dataset reference.
- Verify trained brain produces KS distance comparable to paper's 0.0632.

### Phase 6 — Brain page verdict UI (~1 day)
- Surface per-metric verdict ("matches Table N within X%" / "below
  paper claim by Y%") on the brain page.
- Update `/benchmarks` "Latest community reproduction" column once
  data accumulates.

## Risks

1. **Training convergence shift.** The loss now ignores context-token
   positions. Paper-pinned hyperparameters tuned for the old loss may
   under- or over-train. Plan: keep the old `0.1.0` versions live so
   we can A/B; tune `0.2.0` against the closed-form check.

2. **Existing brains break under predict.** Models trained with
   `(N, 1)` inputs can't run on `(N, F+2)` packed sequences. The
   version bump isolates this — old runs stay on `@0.1.0` and use
   the old predict path; new runs use `@0.2.0` and the new path.
   **Risk: predict.py needs to dispatch on the prior version it sees.**

3. **Wizard-generated synthetic priors.** Some wizard flows construct
   priors inline (not via the catalog). Need to find and update those.

4. **Scorer assumptions.** Multiple scorers iterate `prior.sample()`
   and unpack `X` / `y`. Updating them is mechanical but easy to miss
   one — testing matters.

5. **Two-moons / ar2 / coin-flip etc. unintended drift.** These
   simpler templates aren't on `/benchmarks` but they're in the
   catalog. Either refactor them at the same time (consistent) or
   leave them on `@0.1.0` and only refactor the paper-backed ones.
   Recommendation: refactor all to keep one mental model.

## Acceptance for the whole refactor

- All catalog priors emit packed-token shape (or document why not).
- The trainer regression branch produces models that **demonstrably
  use observed (x, y) pairs at inference** (verifiable on
  `bayesian_linear` because the closed-form posterior gives a
  ground-truth check).
- The KinPFN pilot from today's deferred work can be re-enabled and
  produces a real KS-vs-paper-Table-7 verdict.
- No existing trained brain is silently broken.
