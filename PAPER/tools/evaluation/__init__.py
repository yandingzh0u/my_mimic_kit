"""Method-independent evaluation utilities for paper experiments."""

from .metrics import (  # noqa: F401
    CompletionThresholds,
    canonical_motion_name,
    compute_completion,
    compute_tracking_errors,
    projected_motion_metrics,
    signed_winding_ratio,
)
