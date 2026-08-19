"""Reusable optic disc/cup segmentation and CDR measurement utilities."""

from .cdr import CDRMeasurement, estimate_vertical_cdr, vertical_diameter

__all__ = ["CDRMeasurement", "estimate_vertical_cdr", "vertical_diameter"]
