# priorstudio — CLI

The command-line interface for [PFN Studio](https://github.com/profitopsai/pfnstudio),
the toolkit for training [prior-fitted foundation models](https://arxiv.org/abs/2112.10510).

## Install

```bash
pip install pfnstudio
```

For training (requires PyTorch):

```bash
pip install "pfnstudio[torch]"
```

## Commands

```text
pfnstudio init <dir>            # scaffold a new FM project
pfnstudio validate <path>       # check artifacts against JSON Schema
pfnstudio lint <project>        # cross-reference + style checks
pfnstudio sample <prior.yaml>   # draw N tasks from a prior
pfnstudio run <run.yaml>        # execute a training run end-to-end
pfnstudio predict <run-dir>     # inference against a trained checkpoint
pfnstudio export <project>     # tar-gzipped project archive
```

Run `priorstudio --help` for the full list and `<cmd> --help` for each
subcommand's flags.

## What this CLI is for

PFN Studio organises every PFN project around five first-class
artifacts: **priors** (synthetic data generators), **models** (block
compositions), **evals** (benchmarks + metrics), **runs** (training
manifests), and **initiatives** (research workstreams). This CLI
operates on the file layout those artifacts produce — scaffolding new
projects, validating them, running training, and exporting them for
sharing.

The full story (concepts, architecture, examples, marketplace catalog)
lives at the main repo:
**[github.com/profitopsai/pfnstudio](https://github.com/profitopsai/pfnstudio)**

## License

Apache-2.0.
