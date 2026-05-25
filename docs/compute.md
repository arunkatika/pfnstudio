# Compute adapters

Each compute adapter takes a `run.yaml` and a project root and returns a results dict. The CLI dispatches based on `run.compute.target`.

## Local

In-process. Imports the project, samples from the prior, runs the PFN training loop in `pfnstudio_core.training`. Requires `pfnstudio-core[torch]`.

```yaml
compute:
  target: local
```

Used in tests + for quick iteration. Works on CPU; uses CUDA if available.

## Vast.ai

```yaml
compute:
  target: vast
  gpu: A100-80GB
  num_gpus: 1
```

Requires `VAST_API_KEY` env var and the `vastai` CLI. Per-org config (image, on-start command, instance selector) goes in `.priorstudio/vast.yaml` (not yet enforced — v0.5).

The intended flow:

1. Tar the project root
2. `vastai create instance` matching `gpu` / `num_gpus`
3. Wait for SSH
4. scp the tarball, untar, `pip install pfnstudio`, `pfnstudio run <run.yaml> --target local`
5. scp results.json back

The current implementation submits a stub indicating "configure your vast template" — the full SSH orchestration is not in v0.4. Use `target: local` for now; adapter stubs return the shape/fields you'd see when fully wired.

## Modal

```yaml
compute:
  target: modal
```

Requires `modal>=0.62`. Modal apps are defined in Python; the adapter wraps `modal.App` + a `@app.function(image=...)` that re-runs `target: local` inside the container. Per-org config (image base, secrets) in `.priorstudio/modal.yaml`.

## RunPod

```yaml
compute:
  target: runpod
```

Requires `RUNPOD_API_KEY` and `runpod>=1.6`. Submits to a RunPod template; pulls results when finished.

## HuggingFace Spaces

```yaml
compute:
  target: hf_spaces
```

Requires `HF_TOKEN`. Pushes the project + a Dockerfile to a Space, streams logs. Best for short fine-tunes, not full pretraining.

## Adding your own

Subclass `priorstudio.compute.base.ComputeAdapter`, register in `priorstudio.compute.__init__.ADAPTERS`. The adapter must:

1. Read the run YAML
2. Run the experiment (locally, remotely, or by submitting elsewhere)
3. Return `{"status": ..., **results}`

Trackers (W&B, MLflow, local JSON) wrap the adapter in `cli.run`, so you don't need to log metrics yourself — just return the final results dict.
