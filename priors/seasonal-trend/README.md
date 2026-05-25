# Seasonal + Trend

`y_t = trend(t) + seasonal(t) + noise`

Classic decomposition: a slow trend plus a periodic component plus noise. Demo-worthy for retail / energy / web traffic.

Random slope for the trend, random period and amplitude for the seasonal. The model forecasts each component implicitly from a context window.

## Try it

```bash
pfnstudio sample priors/seasonal-trend/prior.yaml --seeds 3
```

---

_Author: Community_
