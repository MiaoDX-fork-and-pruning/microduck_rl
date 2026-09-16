"""Pure evaluation contracts used by adaptive curriculum experiments."""

from .capability import (
    BUCKETS,
    CapabilityReport,
    build_capability_report,
    resolve_enabled_axes,
    canonical_sha256,
    raw_metrics_from_trace,
)

__all__ = [
    "BUCKETS",
    "CapabilityReport",
    "build_capability_report",
    "resolve_enabled_axes",
    "canonical_sha256",
    "raw_metrics_from_trace",
]
