"""Data pipeline for the spatiotemporal MJO V-JEPA.

Reuses the *exact* anomaly definition of ``0-compute_rmm.ipynb`` (seasonal cycle =
mean + 3 annual harmonics, then trailing-120-day mean removed; no bandpass), but
keeps the full 2-D lat-lon field instead of the meridional average.
"""

from jepa.data.anomaly import (
    BASE_PERIOD,
    LAT_BAND,
    N_HARMONICS,
    ROLLING_DAYS,
    compute_anomalies_2d,
    load_raw_fields_2d,
    to_anomaly,
)
from jepa.data.dataset import MJOSequence, build_dataset

__all__ = [
    "BASE_PERIOD",
    "LAT_BAND",
    "N_HARMONICS",
    "ROLLING_DAYS",
    "to_anomaly",
    "load_raw_fields_2d",
    "compute_anomalies_2d",
    "MJOSequence",
    "build_dataset",
]
