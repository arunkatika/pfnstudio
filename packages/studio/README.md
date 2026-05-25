# pfnstudio-studio

Static-site renderer for an FM project. Reads `ROADMAP.md`, `initiatives/`, `priors/`, `models/`, `evals/`, `runs/`, and `literature/` and produces a browsable HTML site.

```bash
pfnstudio studio build       # output in _studio/
pfnstudio studio serve       # build + serve at http://localhost:8000
```

The site is the same artifacts you'd review in a PR, just rendered. No database, no server, no auth — git is still the source of truth.
