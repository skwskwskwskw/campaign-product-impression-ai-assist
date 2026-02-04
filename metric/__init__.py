"""Utilities for computing campaign × product metrics from coalesced product data."""

from .processing import compute_metrics
from .schema import infer_column_map
from .io import load_dataframe, write_outputs

__all__ = [
    "compute_metrics",
    "infer_column_map",
    "load_dataframe",
    "write_outputs",
]
