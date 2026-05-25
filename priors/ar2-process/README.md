# AR(2) Time Series

`y_t = φ₁·y_{t-1} + φ₂·y_{t-2} + ε_t`

Real-world-shaped time series. Random stationary AR(2) coefficients per task; the PFN learns to forecast any well-behaved autoregressive series.

Stationarity-triangle rejection sampling keeps training bounded. Closer to real industrial data (sensors, demand, traffic) than a sine wave.

## Try it

```bash
pfnstudio sample priors/ar2-process/prior.yaml --seeds 3
```

## Citations

- `mueller2022pfn`

---

_Author: PFN Studio Team_
