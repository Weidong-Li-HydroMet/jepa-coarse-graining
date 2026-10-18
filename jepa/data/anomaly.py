"""Tropical 2-D anomaly fields (OLR, U850, U200) for the MJO V-JEPA.

The anomaly definition is taken **verbatim** from ``0-compute_rmm.ipynb`` (the WH04
RMM reproduction notebook): for each grid point,

  1. remove the seasonal cycle = time mean + first ``N_HARMONICS`` annual harmonics,
     least-squares fit over ``BASE_PERIOD`` (``fit_seasonal_climatology``);
  2. remove the trailing-``ROLLING_DAYS``-day running mean (``remove_prev_mean``),
     which also strips ENSO / low-frequency variability. No bandpass filter.

The RMM notebook then meridionally averages over the equatorial band to feed the
combined EOFs.  Here we deliberately **keep the full 2-D field** (no meridional
mean) so the V-JEPA encoder sees lat-lon maps — exactly what ``0-compute_rmm.ipynb``
§9 does for OLR (``to_anomaly(olr)``), extended to U850 and U200.

All three fields live on the same 2.5° NCEP-2 / NOAA-OLR grid; we restrict latitude
to ``LAT_BAND`` and keep the full 0-357.5°E longitude circle (periodic).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

# --- configuration, mirroring 0-compute_rmm.ipynb -------------------------- #
DATA_ROOT = Path(
    "/glade/campaign/univ/uchi0018/weidong/SSL_project/Self_supervised_learning/MJO/data"
)
YEAR_START = 1979  # first year of the U-wind files
YEAR_END = 2020  # last year of the U-wind files
BASE_PERIOD = slice("1979-01-01", "2020-12-31")  # seasonal-fit base period
LAT_BAND = (15.0, -15.0)  # 15N -> 15S (latitude stored descending); RMM band
N_HARMONICS = 3  # annual harmonics removed with the mean
ROLLING_DAYS = 120  # trailing-N-day mean removed (WH04 real-time convention)

FIELD_ORDER = ("olr", "u850", "u200")


# --------------------------------------------------------------------------- #
# Anomaly operators — copied verbatim from 0-compute_rmm.ipynb (§4).
# --------------------------------------------------------------------------- #
def fit_seasonal_climatology(da, base_period=BASE_PERIOD, n_harm=N_HARMONICS):
    """Mean + first ``n_harm`` annual harmonics, least-squares over ``base_period``.

    Works on any DataArray with a leading 'time' dimension; the fit is done
    independently at each remaining grid point.  Returns the climatology
    evaluated at every time in ``da``.
    """
    t = ((da.time - da.time[0]) / np.timedelta64(1, "D")).values.astype("float64")
    w = 2.0 * np.pi / 365.25
    cols = [np.ones_like(t)]
    for k in range(1, n_harm + 1):
        cols += [np.cos(k * w * t), np.sin(k * w * t)]
    X = np.column_stack(cols)  # (nt, 2*n_harm+1)

    in_base = (
        (da.time >= np.datetime64(base_period.start))
        & (da.time <= np.datetime64(base_period.stop))
    ).values
    y = da.values.reshape(da.time.size, -1)  # (nt, nspace)
    coef, *_ = np.linalg.lstsq(X[in_base], y[in_base], rcond=None)
    return da.copy(data=(X @ coef).reshape(da.shape))


def remove_prev_mean(da, n=ROLLING_DAYS):
    """Subtract the mean of the most recent ``n`` days (including the current day).

    Identical result to ``da - da.rolling(time=n, min_periods=n).mean()`` but computed
    with a cumulative sum so memory stays O(field) instead of O(field x window).  The
    xarray rolling path materializes the full (time, window, space) array — fine for the
    1-D meridional-mean series in ``0-compute_rmm.ipynb``, but ~270 GB (and an OOM) on the
    full 2-D field when ``bottleneck`` is too old to enable the streaming reduction.
    """
    v = da.values.astype("float64")
    T = v.shape[0]
    if T < n:
        return da.copy(data=np.full_like(v, np.nan))
    csum0 = np.concatenate([np.zeros((1,) + v.shape[1:]), np.cumsum(v, axis=0)], axis=0)
    trailing = (csum0[n:] - csum0[: T - n + 1]) / n  # trailing-n mean for t in [n-1, T)
    out = np.full_like(v, np.nan)
    out[n - 1 :] = v[n - 1 :] - trailing
    return da.copy(data=out)


def to_anomaly(da):
    """Full WH04 anomaly: seasonal cycle removed, then trailing-120-day mean removed."""
    return remove_prev_mean(da - fit_seasonal_climatology(da))


# --------------------------------------------------------------------------- #
# Raw field loading — mirrors 0-compute_rmm.ipynb §1-2 (grid check, no regrid).
# --------------------------------------------------------------------------- #
def load_raw_fields_2d(
    data_root: Path = DATA_ROOT,
    year_start: int = YEAR_START,
    year_end: int = YEAR_END,
    lat_band: tuple[float, float] = LAT_BAND,
) -> dict[str, xr.DataArray]:
    """Load OLR, U850, U200 as 2-D (time, lat, lon) DataArrays on the common grid.

    Returns a dict keyed by ``FIELD_ORDER``.  Latitude is restricted to ``lat_band``
    (descending), longitude kept full (0-357.5°E).  OLR is regridded onto the wind
    grid only if the grids differ (they do not, for the staged 2.5° data).
    """
    data_root = Path(data_root)

    uwnd_files = sorted((data_root / "ncep2_uwnd").glob("uwnd.*.nc"))
    uwnd_files = [
        f for f in uwnd_files if year_start <= int(f.stem.split(".")[1]) <= year_end
    ]
    if not uwnd_files:
        raise FileNotFoundError(f"no uwnd.YYYY.nc files in {data_root/'ncep2_uwnd'}")

    def _prep(ds):
        return ds[["uwnd"]].sel(level=[850.0, 200.0], lat=slice(*lat_band))

    ds_u = xr.open_mfdataset(uwnd_files, preprocess=_prep, combine="by_coords")
    u850 = ds_u.uwnd.sel(level=850.0, drop=True).load().astype("float64")
    u200 = ds_u.uwnd.sel(level=200.0, drop=True).load().astype("float64")

    dt = np.unique(np.diff(u850.time.values).astype("timedelta64[D]").astype(int))
    assert (dt == 1).all(), f"wind time axis is not continuous daily: gaps {dt}"

    ds_olr = xr.open_dataset(data_root / "olr" / "olr.day.mean.nc")
    olr_full = ds_olr.olr.sel(time=slice(f"{year_start}-01-01", f"{year_end}-12-31"))
    same_grid = np.array_equal(ds_olr.lon.values, ds_u.lon.values) and np.isin(
        ds_u.lat.values, ds_olr.lat.values
    ).all()
    if not same_grid:
        olr_full = olr_full.interp(lat=ds_u.lat, lon=ds_u.lon)
    olr = olr_full.sel(lat=slice(*lat_band)).load().astype("float64")

    # inner-join on time -> the common daily period
    olr, u850, u200 = xr.align(olr, u850, u200, join="inner")
    for name, f in [("olr", olr), ("u850", u850), ("u200", u200)]:
        assert not np.isnan(f).any(), f"missing values in {name}"
    return {"olr": olr, "u850": u850, "u200": u200}


def compute_anomalies_2d(
    fields: dict[str, xr.DataArray],
) -> tuple[dict[str, xr.DataArray], np.ndarray]:
    """Apply ``to_anomaly`` to each 2-D field; drop the leading undefined days.

    Returns ``(anoms, times)`` where ``anoms[var]`` is (time, lat, lon) and ``times``
    is the common valid time axis (after the trailing-mean warm-up is dropped).
    """
    anoms = {k: to_anomaly(fields[k]) for k in FIELD_ORDER}
    valid_t = ~np.isnan(anoms["olr"]).any(("lat", "lon"))
    anoms = {k: v.sel(time=valid_t) for k, v in anoms.items()}
    times = anoms["olr"].time.values
    return anoms, times
