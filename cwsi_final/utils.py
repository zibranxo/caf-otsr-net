import numpy as np
from scipy import stats
import warnings
from skimage import measure
from shapely.geometry import Polygon
import geopandas as gpd


# -------------------------------
# NPZ LOADING
# -------------------------------
def load_npz(path_or_buffer):
    data = np.load(path_or_buffer, allow_pickle=True)

    if "arr" in data:
        arr = data["arr"]
    else:
        # auto-detect 3D array if arr missing
        arr = None
        for k in data.files:
            if k in ("transform", "crs", "meta"):
                continue
            v = data[k]
            if isinstance(v, np.ndarray) and v.ndim == 3:
                arr = v
                break

        # try constructing from multiple 2D arrays
        if arr is None:
            chans = []
            for k in sorted(data.files):
                v = data[k]
                if isinstance(v, np.ndarray) and v.ndim == 2:
                    chans.append(v)
            if chans:
                arr = np.stack(chans, axis=0)
            else:
                raise ValueError("NPZ missing 'arr' or any 3D structure.")

    transform = data["transform"] if "transform" in data else None
    crs = data["crs"] if "crs" in data else None
    meta = data["meta"].item() if "meta" in data else {}

    return {
        "arr": arr.astype("float32"),
        "transform": transform,
        "crs": crs,
        "meta": meta
    }


# -------------------------------
# CHANNEL MAPPING
# -------------------------------
def map_channels(arr):
    names = {
        0: "tir_lr_upsampled",
        1: "tir_hr",
        2: "opt_blue",
        3: "opt_green",
        4: "opt_red",
        5: "opt_nir",
        6: "opt_swir1",
        7: "opt_swir2",
        8: "opt_swir1_dup",
        9: "opt_swir2_dup",
        10: "ndvi",
        11: "emissivity",
        12: "ndbi",
        13: "tex_red",
        14: "tex_nir",
        15: "lst_normalized",
    }

    out = {}
    C = arr.shape[0]
    for i in range(C):
        out[names.get(i, f"c{i}")] = arr[i]

    return out


# -------------------------------
# EMISSIVITY
# -------------------------------
def emissivity_from_ndvi(ndvi):
    ndvi_min = np.nanmin(ndvi)
    ndvi_max = np.nanmax(ndvi)
    denom = (ndvi_max - ndvi_min) or 1.0

    fv = np.clip((ndvi - ndvi_min) / denom, 0, 1)
    return 0.97 + 0.02 * fv


def apply_emissivity_correction(lst, emissivity, ref_emiss=0.99):
    safe_emiss = np.where(np.isfinite(emissivity), emissivity, ref_emiss)
    return lst + (1 - safe_emiss) * 0.5


# -------------------------------
# CWSI ENVELOPES
# -------------------------------
def compute_envelopes(lst, ndvi, nbins=20, hot_pct=95, cold_pct=5, min_count=20):
    mask = np.isfinite(lst) & np.isfinite(ndvi)
    if np.count_nonzero(mask) < min_count:
        raise ValueError("Not enough valid pixels for envelope estimation.")

    nd = ndvi[mask]
    ls = lst[mask]

    bins = np.linspace(np.nanmin(nd), np.nanmax(nd), nbins + 1)

    centers, Th, Tc = [], [], []

    for i in range(nbins):
        lo, hi = bins[i], bins[i + 1]
        sel = (ndvi >= lo) & (ndvi < hi) & mask

        if np.count_nonzero(sel) < 10:
            continue

        vals = lst[sel]
        centers.append((lo + hi) / 2)
        Th.append(np.nanpercentile(vals, hot_pct))
        Tc.append(np.nanpercentile(vals, cold_pct))

    centers = np.array(centers)
    Th = np.array(Th)
    Tc = np.array(Tc)

    # fallback if insufficient bins
    if centers.size < 2:
        Thv = np.nanpercentile(ls, hot_pct)
        Tcv = np.nanpercentile(ls, cold_pct)
        return (
            lambda x: Thv * np.ones_like(x),
            lambda x: Tcv * np.ones_like(x),
            centers,
            Th,
            Tc
        )

    slope_T, inter_T, *_ = stats.theilslopes(Th, centers, 0.95)
    slope_C, inter_C, *_ = stats.theilslopes(Tc, centers, 0.95)

    Th_fn = lambda x: slope_T * x + inter_T
    Tc_fn = lambda x: slope_C * x + inter_C

    return Th_fn, Tc_fn, centers, Th, Tc


# -------------------------------
# Compute CWSI
# -------------------------------
def compute_cwsi(lst, ndvi, Th_func, Tc_func, min_den=0.5):
    Th = Th_func(ndvi)
    Tc = Tc_func(ndvi)
    denom = Th - Tc

    cwsi = np.full_like(lst, np.nan, dtype="float32")

    valid = np.isfinite(lst) & np.isfinite(ndvi) & (denom >= min_den)
    cwsi[valid] = (lst[valid] - Tc[valid]) / denom[valid]

    warnings.warn("Some pixels have Th-Tc < min_den; assigned NaN.")

    return np.clip(cwsi, 0, 1), Th, Tc


# -------------------------------
# CLASS MAP
# -------------------------------
def class_map(cwsi):
    out = np.zeros_like(cwsi, dtype="uint8")

    out[(cwsi >= 0) & (cwsi <= 0.2)] = 1
    out[(cwsi > 0.2) & (cwsi <= 0.4)] = 2
    out[(cwsi > 0.4) & (cwsi <= 0.7)] = 3
    out[(cwsi > 0.7)] = 4

    return out


# -------------------------------
# GLOBAL SUMMARY
# -------------------------------
def global_summary(cwsi):
    v = cwsi[np.isfinite(cwsi)]
    if v.size == 0:
        return {}

    return {
        "mean_cwsi": float(np.nanmean(v)),
        "median_cwsi": float(np.nanmedian(v)),
        "p90_cwsi": float(np.nanpercentile(v, 90)),
        "pct_area_stressed": float(np.mean(v > 0.5)),
        "std_cwsi": float(np.nanstd(v)),
        "n_pixels": int(v.size)
    }


# -------------------------------
# PATCH-GRID STATS
# -------------------------------
def grid_patches_stats(cwsi, ndvi, patch_size=32):
    H, W = cwsi.shape

    rows = []
    patch_id = 0

    for r in range(0, H, patch_size):
        for c in range(0, W, patch_size):
            cw = cwsi[r:r + patch_size, c:c + patch_size]
            nd = ndvi[r:r + patch_size, c:c + patch_size]

            vals = cw[np.isfinite(cw)]

            if vals.size == 0:
                row = {
                    "patch_id": patch_id,
                    "row": r,
                    "col": c,
                    "mean_cwsi": None,
                    "median_cwsi": None,
                    "p90_cwsi": None,
                    "pct_area_stressed": None,
                    "std_cwsi": None,
                    "n_pixels": 0,
                    "mean_ndvi": None
                }
            else:
                row = {
                    "patch_id": patch_id,
                    "row": r,
                    "col": c,
                    "mean_cwsi": float(np.nanmean(vals)),
                    "median_cwsi": float(np.nanmedian(vals)),
                    "p90_cwsi": float(np.nanpercentile(vals, 90)),
                    "pct_area_stressed": float(np.mean(vals > 0.5)),
                    "std_cwsi": float(np.nanstd(vals)),
                    "n_pixels": int(vals.size),
                    "mean_ndvi": float(np.nanmean(nd))
                }

            rows.append(row)
            patch_id += 1

    import pandas as pd
    return pd.DataFrame(rows)


# -------------------------------
# HOTSPOT DETECTION
# -------------------------------
def detect_hotspots(cwsi, transform=None, crs=None, threshold=0.7, min_area_pixels=10):
    mask = (cwsi > threshold).astype("uint8")
    labels = measure.label(mask, connectivity=1)
    regions = measure.regionprops(labels)

    geoms = []

    for reg in regions:
        if reg.area < min_area_pixels:
            continue

        minr, minc, maxr, maxc = reg.bbox

        # Convert to polygon
        if transform is not None:
            x0, xres, _, y0, _, yres_neg = transform
            x_min = x0 + minc * xres
            x_max = x0 + maxc * xres
            y_max = y0 + minr * yres_neg
            y_min = y0 + maxr * yres_neg

            poly = Polygon([(x_min, y_min), (x_min, y_max),
                            (x_max, y_max), (x_max, y_min)])
        else:
            poly = Polygon([(minc, minr), (minc, maxr),
                            (maxc, maxr), (maxc, minr)])

        geoms.append({
            "geometry": poly,
            "area_pixels": int(reg.area),
            "label": int(reg.label)
        })

    return gpd.GeoDataFrame(geoms, crs=crs)


# -------------------------------
# DATE EXTRACTION
# -------------------------------
def get_date_from_meta(meta, fallback_idx):
    for k in ("date", "acq_date", "timestamp", "time"):
        if k in meta:
            return str(meta[k])
    return f"t{fallback_idx}"
