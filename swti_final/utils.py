
import numpy as np
from scipy import ndimage
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.morphology import disk, remove_small_objects, binary_dilation
from skimage import measure, morphology
from shapely.geometry import Polygon
from shapely.ops import unary_union
import warnings

def load_npz_any(path_or_buffer):
    """Robust loader: returns (arr3d, meta_dict). Scans keys for a 3D numpy array if 'arr' not present."""
    data = np.load(path_or_buffer, allow_pickle=True)
    # direct keys
    if 'arr' in data:
        return data['arr'], data.get('meta', {})
    # handle common alternatives
    for key in ('patch','features','stack','arr_0','arr0','data'):
        if key in data and isinstance(data[key], np.ndarray) and data[key].ndim == 3:
            return data[key], data.get('meta', {})
    # otherwise find first 3D array
    for k in data.files:
        v = data[k]
        if isinstance(v, np.ndarray) and v.ndim == 3:
            return v, data.get('meta', {})
    raise ValueError("No 3D array found in NPZ file. Available keys: {}".format(list(data.files)))

# LST cleaning & smoothing helpers (same as before)
def fill_holes_and_mask(arr, mask):
    a = arr.copy().astype('float32')
    valid = np.isfinite(a) & (mask==True)
    if valid.sum() == 0:
        return a
    inv = ~valid
    d, inds = ndimage.distance_transform_edt(inv, return_distances=True, return_indices=True)
    filled = a.copy()
    filled[inv] = a[tuple(inds[:, inv])]
    return filled

from skimage.restoration import denoise_nl_means, estimate_sigma as _est_sigma
def denoise_nl_means_masked(arr, mask=None, patch_size=5, patch_distance=6, h_scale=1.0):
    a = arr.copy().astype('float32')
    valid = np.isfinite(a)
    if mask is not None:
        valid = valid & (mask==True)
    if valid.sum() == 0:
        return a
    med = np.nanmedian(a[valid])
    a_f = a.copy()
    a_f[~np.isfinite(a_f)] = med
    sigma_est = float(_est_sigma(a_f, multichannel=False))
    h = h_scale * max(0.5, sigma_est)
    den = denoise_nl_means(a_f, h=h, sigma=sigma_est, patch_size=patch_size, patch_distance=patch_distance, fast_mode=True)
    out = a.copy()
    out[valid] = den[valid]
    return out

def gaussian_smooth(arr, sigma=1.5):
    return ndimage.gaussian_filter(arr, sigma=sigma, mode='nearest')

def smooth_then_gradient(lst, water_mask, sigma=1.5, use_nlme=False, nlme_params=None):
    filled = fill_holes_and_mask(lst, water_mask)
    if use_nlme:
        params = nlme_params or {}
        den = denoise_nl_means_masked(filled, mask=water_mask, **params)
    else:
        den = filled
    sm = gaussian_smooth(den, sigma=sigma)
    gx = ndimage.sobel(sm, axis=1, mode='nearest')
    gy = ndimage.sobel(sm, axis=0, mode='nearest')
    grad = np.hypot(gx, gy)
    grad[~water_mask] = np.nan
    return grad, sm

def multi_scale_gradient(lst, water_mask, sigmas=(1.0,2.5,4.0), weights=None, use_nlme=False):
    if weights is None:
        weights = [1.0/len(sigmas)]*len(sigmas)
    grads = {}
    last_sm = None
    combined = np.zeros_like(lst, dtype='float32')
    for s,w in zip(sigmas, weights):
        g, sm = smooth_then_gradient(lst, water_mask, sigma=s, use_nlme=use_nlme)
        grads[s] = g
        last_sm = sm
        gw = np.nan_to_num(g) * w
        combined += gw
    maxv = np.nanmax(combined)
    if maxv > 0:
        combined = combined / (maxv + 1e-9)
    combined[~water_mask] = np.nan
    return combined, grads, last_sm

def adaptive_plume_detection(tai, water_mask, tile_size=64, z_thresh=1.5, min_area=50, dilation=2, simplify_tol=2.0, transform=None, crs=None):
    H,W = tai.shape
    mask = np.isfinite(tai) & water_mask
    thresh_map = np.full_like(tai, np.nan)
    for r in range(0, H, tile_size):
        for c in range(0, W, tile_size):
            r1, r2 = r, min(H, r+tile_size)
            c1, c2 = c, min(W, c+tile_size)
            tile_mask = mask[r1:r2, c1:c2]
            if tile_mask.sum() < 5:
                continue
            vals = tai[r1:r2, c1:c2][tile_mask]
            med = np.nanmedian(vals); std = np.nanstd(vals)
            thr = med + z_thresh * (std if std>0 else 0.01)
            thresh_map[r1:r2, c1:c2] = thr
    plume_bool = (tai > thresh_map) & water_mask & np.isfinite(thresh_map)
    clean = remove_small_objects(plume_bool.astype(bool), min_size=8)
    if dilation>0:
        selem = disk(dilation)
        clean = binary_dilation(clean, selem)
    labels = measure.label(clean, connectivity=1)
    regions = measure.regionprops(labels)
    features = []
    for reg in regions:
        if reg.area < min_area:
            continue
        coords_list = measure.find_contours(labels==reg.label, 0.5)
        polys = []
        for cont in coords_list:
            pts = [(float(p[1]), float(p[0])) for p in cont]
            if len(pts) >= 3:
                poly = Polygon(pts)
                polys.append(poly)
        if len(polys) == 0:
            minr,minc,maxr,maxc = reg.bbox
            poly = Polygon([(minc,minr),(minc,maxr),(maxc,maxr),(maxc,minr)])
        else:
            poly = unary_union(polys)
        try:
            poly = poly.simplify(simplify_tol, preserve_topology=True)
        except Exception:
            pass
        max_val = float(np.nanmax(tai[labels==reg.label]))
        mean_val = float(np.nanmean(tai[labels==reg.label]))
        area_px = int(reg.area)
        features.append({'geometry': poly, 'area_pixels': area_px, 'max_val': max_val, 'mean_val': mean_val, 'label': int(reg.label)})
    return features

def normalize_for_display(arr, vmin=None, vmax=None):
    a = arr.copy().astype('float32')
    if vmin is None: vmin = np.nanpercentile(a, 2)
    if vmax is None: vmax = np.nanpercentile(a, 98)
    return (a - vmin) / (vmax - vmin + 1e-9)
