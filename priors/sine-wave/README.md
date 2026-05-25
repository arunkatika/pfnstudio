# Sine Wave

`y_t = a·sin(ω·t + φ) + noise`

The simplest temporal PFN. Random amplitude, frequency, and phase per task; the model forecasts the continuation from a noisy prefix.

A clean, well-understood time-series prior. Good warm-up before AR / ARMA / seasonal priors.

## Try it

```bash
pfnstudio sample priors/sine-wave/prior.yaml --seeds 3
```

## Citations

- `mueller2022pfn`

---

_Author: PFN Studio Team_
