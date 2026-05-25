# pfnstudio-core

The Python contract for PFN Studio FM projects.

```python
from pfnstudio_core import Prior, Model, Eval, Run, register_block, register_prior

@register_prior("my_prior")
class MyPrior(Prior):
    def sample(self, seed: int): ...

@register_block("my_attention")
class MyAttention:
    def __init__(self, d_model: int, n_heads: int): ...
```

The CLI discovers anything registered via these decorators and validates `models/*.yaml` references against the registry.

## Layout

- `prior.py` — `Prior` ABC and built-in prior loader
- `model.py` — `Model` config + block-composition
- `eval.py` — `Eval` config + result schema
- `run.py` — `Run` manifest + executor protocol
- `registry.py` — `@register_prior`, `@register_block`, `@register_eval` and discovery
- `loaders.py` — load YAML artifacts into typed objects
- `blocks/` — built-in architecture blocks (transformer encoder, causal attention, heads)
- `training/` — minimal in-process training loop for the `local` compute adapter
