"""Built-in dataset scorers.

A "scorer" owns the full pipeline from raw dataset bytes to metric value:
load the dataset via RegistryDatasetLoader, run the trained model on the
relevant slice, compute the metrics declared on the eval spec.

The CLI's local adapter looks up scorers by eval slug after training
completes. Templates that want a real-data eval ship an eval YAML whose
slug matches a key in BUILTIN_SCORERS; any unmatched slug is treated as
synthetic-only (no real-data scoring step).

To add a new scorer:
  1. Implement a subclass of DatasetScorer in this package.
  2. Register it in BUILTIN_SCORERS below by eval slug.
"""

from __future__ import annotations

from .base import DatasetScorer, ScorerResult
from .breast_cancer_vs_logreg import BreastCancerVsLogReg
from .closed_form_baseline import ClosedFormBaseline
from .in_context_regression_ols import InContextRegressionVsOLS
from .kinpfn_real_fpt_ks import KinPFNRealFPTKS
from .m4_monthly_forecast import M4MonthlyForecast
from .synthetic_classification_bce import SyntheticClassificationBCE
from .synthetic_regression_mse import SyntheticRegressionMSE

BUILTIN_SCORERS: dict[str, DatasetScorer] = {
    # Real-dataset scorers — keyed on the eval slug shipped by the
    # template that uses them.
    "m4_monthly_mse": M4MonthlyForecast(),
    "in_context_regression_ols": InContextRegressionVsOLS(),
    "breast_cancer_vs_logreg": BreastCancerVsLogReg(),
    # Closed-form Bayesian baseline for pfns-reference. Verifies the PFN
    # matches the analytic posterior mean on bayesian_linear tasks.
    "closed_form_baseline": ClosedFormBaseline(),
    # KinPFN real-RNA FPT KS distance — compares trained brain to the
    # paper's Table 7 (KS = 0.0632 at N=100 context).
    "kinpfn_real_fpt_ks": KinPFNRealFPTKS(),
    # Generic synthetic scorers — work on any prior emitting the
    # standard X + y (or X + labels) shape. The wizard's buildEval
    # routes here for non-paper-backed runs so the run-detail Evals
    # scorecard always has data to show.
    "synthetic_regression_mse": SyntheticRegressionMSE(),
    "synthetic_classification_bce": SyntheticClassificationBCE(),
}

__all__ = ["BUILTIN_SCORERS", "DatasetScorer", "ScorerResult"]
