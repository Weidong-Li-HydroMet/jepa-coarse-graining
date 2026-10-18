"""Build standardized 2-D MJO anomaly arrays once, then memory-map them at train time.

Pipeline (reuses ``jepa.data.anomaly``):
  1. load raw OLR / U850 / U200 on the 2.5° grid, 15°N-15°S band, full longitude;
  2. compute WH04 anomalies (seasonal harmonics + trailing-120-day mean), keeping the
     full 2-D field;
  3. standardize each variable by its *training-period* global mean/std;
  4. save a single contiguous array + the date index + the stats used.

Outputs (under ``data.cache_dir``):
    anom.npy    float32  [N, C=3, n_lat, n_lon]  (channel order = olr, u850, u200)
    dates.npy   datetime64[D]  [N]
    stats.npz   mean[C], std[C], lat[n_lat], lon[n_lon], variables, train_years

Run (once, on a node with xarray; ~2-3 GB peak):
    python -m jepa.data.preprocess --config jepa/config_vjepa.yaml
    python -m jepa.data.preprocess --config jepa/config_vjepa.yaml --subset 1998 2000   # quick test
"""

from __future__ import annotations

import argparse
import gc
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from jepa.data.anomaly import FIELD_ORDER, load_raw_fields_2d, to_anomaly  # noqa: E402


def build(cfg_data: dict, subset: tuple[int, int] | None = None) -> None:
    y0 = subset[0] if subset else cfg_data.get("year_start", 1979)
    y1 = subset[1] if subset else cfg_data.get("year_end", 2020)
    lat_band = tuple(cfg_data.get("lat_band", (15.0, -15.0)))
    cache_dir = cfg_data["cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)

    print(f"[preprocess] loading raw fields {y0}-{y1}, lat band {lat_band} ...", flush=True)
    fields = load_raw_fields_2d(
        data_root=cfg_data["data_root"], year_start=y0, year_end=y1, lat_band=lat_band
    )
    lat = fields["olr"].lat.values.astype("float32")
    lon = fields["olr"].lon.values.astype("float32")

    # Memory-frugal: compute anomalies one variable at a time into a float32 buffer,
    # freeing each float64 intermediate (the 10 GB session cgroup is shared across procs).
    print("[preprocess] computing WH04 2-D anomalies (seasonal + 120-day mean) ...", flush=True)
    nlat, nlon = lat.size, lon.size
    arr = None
    valid = None
    times = None
    for c, v in enumerate(FIELD_ORDER):
        a = to_anomaly(fields[v])
        if valid is None:
            valid = (~np.isnan(a).any(("lat", "lon"))).values  # warm-up days dropped
            times = fields[v].time.values[valid]
            N = int(valid.sum())
            arr = np.empty((N, len(FIELD_ORDER), nlat, nlon), dtype="float32")
        arr[:, c] = a.values[valid].astype("float32")
        del a
        gc.collect()
        print(f"[preprocess]   {v} anomaly done", flush=True)
    del fields
    gc.collect()
    dates = times.astype("datetime64[D]")
    C = len(FIELD_ORDER)
    print(f"[preprocess] anomaly array {arr.shape}  {dates[0]} -> {dates[-1]}", flush=True)

    # standardize each variable by its TRAINING-period global mean/std
    tr0, tr1 = cfg_data["train_years"]
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    train_mask = (years >= tr0) & (years <= tr1)
    assert train_mask.any(), f"no days in train years {tr0}-{tr1}"
    mean = arr[train_mask].mean(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)  # [C]
    std = arr[train_mask].std(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)  # [C]
    arr -= mean[None, :, None, None]  # in-place to avoid a full extra copy
    arr /= std[None, :, None, None]

    np.save(os.path.join(cache_dir, "anom.npy"), arr)
    np.save(os.path.join(cache_dir, "dates.npy"), dates)
    np.savez(
        os.path.join(cache_dir, "stats.npz"),
        mean=mean.astype("float32"),
        std=std.astype("float32"),
        lat=lat,
        lon=lon,
        variables=np.array(FIELD_ORDER),
        train_years=np.array([tr0, tr1]),
        lat_band=np.array(lat_band, dtype="float32"),
    )
    print(
        f"[preprocess] saved -> {cache_dir}\n"
        f"  per-variable train mean = {np.round(mean, 3)}\n"
        f"  per-variable train std  = {np.round(std, 3)}\n"
        f"  grid: {nlat} lat x {nlon} lon  ({lat[0]:.1f}..{lat[-1]:.1f}°, "
        f"{lon[0]:.1f}..{lon[-1]:.1f}°E)"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--subset", nargs=2, type=int, default=None, metavar=("Y0", "Y1"),
        help="restrict to a short year range for a quick test build",
    )
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    build(cfg["data"], subset=tuple(args.subset) if args.subset else None)


if __name__ == "__main__":
    main()
