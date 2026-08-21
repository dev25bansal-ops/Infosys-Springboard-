"""Cascade convenience re-export.

`from flash_crash_watchdog.cascade import DetectionCascade`
is equivalent to
`from flash_crash_watchdog.models.cascade import DetectionCascade`.
"""
from flash_crash_watchdog.models.cascade import (
    Alert,
    CascadeStats,
    DetectionCascade,
)

__all__ = ["Alert", "CascadeStats", "DetectionCascade"]
