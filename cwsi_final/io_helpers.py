import numpy as np
import rasterio
from rasterio.transform import Affine
import imageio
from PIL import Image
import os


# -------------------------------
# Save GeoTIFF
# -------------------------------
def save_tif(path, arr, transform, crs, nodata=-9999.0, dtype="float32"):
    arr_out = np.where(np.isfinite(arr), arr, nodata).astype(dtype)

    meta = {
        "driver": "GTiff",
        "height": arr_out.shape[0],
        "width": arr_out.shape[1],
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": Affine(*transform),
        "nodata": nodata,
        "compress": "lzw"
    }

    with rasterio.open(path, "w", **meta) as dst:
        dst.write(arr_out, 1)


# -------------------------------
# Save PNG
# -------------------------------
def save_png(path, fig):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return path


# -------------------------------
# Adaptive resizing for animation
# -------------------------------
def resize_for_animation(img, max_size=512):
    H, W = img.shape

    if max(H, W) <= max_size:
        return img

    scale = max_size / max(H, W)
    newH = int(H * scale)
    newW = int(W * scale)

    return np.array(Image.fromarray(img).resize((newW, newH)))


# -------------------------------
# GIF / MP4 animation
# -------------------------------
def save_gif(path, frames, fps=2):
    imageio.mimsave(path, frames, fps=fps)
    return path


def save_mp4(path, frames, fps=2):
    imageio.mimsave(path, frames, fps=fps, format="FFMPEG")
    return path
