# Paper audit notes

One file per paper-backed prior in the catalog, documenting:
- Which tables / sections each catalog value came from
- Open questions for the original authors
- Hand-off notes for whoever picks up the next audit pass

This directory exists because `/benchmarks` claims to be a falsifiable
reproducibility leaderboard. The claim is only as credible as the
provenance of the numbers in the catalog. Hand-waved baselines or
mis-cited numbers compromise the *whole* page, so we keep an explicit
paper trail.

## Current state

| Paper | Status | File |
|---|---|---|
| **Müller 2022 (pfns-reference regression)** | ✅ **Reproducible end-to-end** — trained brain matches closed-form posterior; `scripts/reproduce-pfns-reference.py` | [muller-2022-pfns.md](./muller-2022-pfns.md) |
| **KinPFN (ICLR 2025)** | ✅ **Reproducible end-to-end** — trained brain produces real-RNA KS verdict vs Table 7's 0.0632; `scripts/reproduce-kinpfn.py` | [kinpfn-2025.md](./kinpfn-2025.md) |
| **Rakotoarison 2024 (ifBO)** | ✅ Table 1 LCBench-1000 MSE baselines populated; needs Phase 3 packed-token refactor | [rakotoarison-2024-ifbo.md](./rakotoarison-2024-ifbo.md) |
| **Müller 2022 (pfns-classification)** | ⚠️ Audited — paper's OpenML table doesn't include breast-cancer; catalog eval is analytical-style | [muller-2022-pfns.md](./muller-2022-pfns.md) |
| **Adriaensen 2023 (LC-PFN)** | ⚠️ Audited — paper reports log-likelihood, catalog measures MSE (different unit); needs new metric to cite | [adriaensen-2023-lcpfn.md](./adriaensen-2023-lcpfn.md) |
| **Müller 2023 (PFNs4BO)** | ⚠️ Audited — paper tables are HPO-B regret; catalog is analytical GP-mean check (different eval suite) | [muller-2023-pfns4bo.md](./muller-2023-pfns4bo.md) |
| **Hoo 2024 (TabPFN-TS)** | ⚠️ Audited — paper reports GIFT-Eval aggregate, no per-M4 number; inference-only paper | [hoo-2024-tabpfn-ts.md](./hoo-2024-tabpfn-ts.md) |

**Legend:** ✅ paper-cited baselines live in `templates.ts`. ⚠️ audited
but the catalog's eval surface doesn't match the paper's tabulated
metrics — the audit-note documents what's needed to close the gap
(typically a new metric, new eval, or new dataset wiring).

## How to audit a new paper

1. **Open the paper.** PDF or arxiv HTML (newer papers only have HTML).
2. **Find the headline results table.** Usually Experiments / §4 / Table 1.
3. **Identify the "PFN row"** — the paper's own claimed performance.
4. **Identify the baseline rows** — alternative methods compared against.
5. **Edit `apps/web/src/app/projects/templates.ts`**. Find the relevant
   template (e.g. `PFNS_REFERENCE_TEMPLATE` for Müller 2022) and update
   `seedEvals[*].baselines[*]`:
   ```ts
   { name: '<method-name>', score: <number>, source: '<Author Year Table N — dataset, context size, metric>' },
   ```
   Don't put the paper's own claimed PFN value in `baselines` —
   instead put the *bar to beat*, plus the paper's claimed number
   as a `target` row labelled "X target (paper)".
6. **Cross-reference `seedRuns[0].hyperparams`** with the paper's
   Implementation Details. If they differ, update — and add a note in
   the audit file naming the section the values came from.
7. **Update this audit file.** Mention the table, the open questions,
   what's not yet captured.
8. **Build + commit**. The wizard's step-5 paper-pinned banner will
   surface the new values; the brain page's Paper reproduction card
   will compare the trained brain's measured score against them.

## Why we keep these notes

- **Future contributors** don't have to re-discover the source — every
  catalog value points back to a paper section.
- **Disputes** about reproduction quality are resolvable: "we trained
  on X but you say Y" can be settled by re-reading the cited section
  together.
- **The audit work itself becomes citable** — *"PFN Studio reproduces
  Müller 2022 per their Table 2 (see audit-notes/muller-2022-pfns.md)"*
  is a stronger claim than *"PFN Studio reproduces Müller 2022"*.
