import numpy as np
import os
import argparse

def simulate_ndvi_curve(t, total):
    """
    Realistic crop NDVI seasonal curve:
    0 → vegetative growth → peak → senescence → harvest
    """
    t_norm = t / (total - 1)
    growth = np.exp(-((t_norm - 0.3) ** 2) / 0.01)
    decline = np.exp(-((t_norm - 0.8) ** 2) / 0.02)
    ndvi_peak = np.exp(-((t_norm - 0.5) ** 2) / 0.05)

    return 0.15 + 0.65 * (growth + ndvi_peak) / 2 - 0.4 * decline


def generate_snapshot(H, W, t, total_snapshots, base_temp=305, stress_event=False):
    """
    Generate a 16-channel snapshot according to your model structure.
    """
    C = 16
    arr = np.zeros((C, H, W), dtype="float32")

    # --- NDVI evolution ---
    ndvi_mean = simulate_ndvi_curve(t, total_snapshots)
    ndvi = np.clip(
        ndvi_mean + 0.05 * np.random.randn(H, W),
        0.05,
        0.9
    )

    # Save NDVI in channel 10 for compatibility
    arr[10] = ndvi

    # --- Emissivity ---
    emissivity = 0.97 + 0.02 * ((ndvi - ndvi.min()) / (ndvi.max() - ndvi.min() + 1e-6))
    arr[11] = emissivity

    # --- LST (thermal infrared HR: channel 1) ---
    stress_factor = 0

    # Stress event simulation (irrigation failure, heatwave)
    if stress_event:
        stress_mask = np.zeros((H, W))
        cx, cy = np.random.randint(W//4, 3*W//4), np.random.randint(H//4, 3*H//4)
        rx, ry = np.random.randint(20, 40), np.random.randint(20, 40)
        yy, xx = np.indices((H, W))
        stress_mask[((xx - cx)**2)/rx**2 + ((yy - cy)**2)/ry**2 < 1] = 1
        stress_factor = 4.5 * stress_mask  # hotter in stressed area

    lst_hr = (
        base_temp
        - 8 * ndvi               # colder for high NDVI
        + 3 * np.random.randn(H, W)     # noise
        + stress_factor                  # added stress heat
    )
    arr[1] = lst_hr

    # --- LST LR (upsampled): channel 0 ---
    arr[0] = lst_hr + np.random.randn(H, W) * 1.5

    # --- Optical channels (blue, green, red, nir, swir1, swir2) ---
    # Simulated reflectance tied loosely to NDVI
    arr[2] = 0.1 + 0.02 * np.random.randn(H, W)          # Blue
    arr[3] = 0.12 + 0.02 * np.random.randn(H, W)         # Green
    arr[4] = (0.2 + 0.1*(1-ndvi)) + 0.03*np.random.randn(H,W)   # Red
    arr[5] = (0.3 + 0.2*ndvi) + 0.04*np.random.randn(H,W)        # NIR
    arr[6] = 0.15 + 0.03*np.random.randn(H, W)           # SWIR1
    arr[7] = 0.18 + 0.03*np.random.randn(H, W)           # SWIR2
    arr[8] = arr[6].copy()
    arr[9] = arr[7].copy()

    # --- NDBI (simple synthetic urban proxy) ---
    arr[12] = (arr[6] - arr[5]) / (arr[6] + arr[5] + 1e-6)

    # --- Texture channels ---
    arr[13] = np.abs(np.random.randn(H, W)) * 0.1
    arr[14] = np.abs(np.random.randn(H, W)) * 0.1

    # --- LST normalized ---
    arr[15] = (lst_hr - lst_hr.min()) / (lst_hr.max() - lst_hr.min() + 1e-6)

    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", type=int, default=6, help="Number of time-series NPZ files")
    parser.add_argument("--H", type=int, default=256)
    parser.add_argument("--W", type=int, default=256)
    parser.add_argument("--out_dir", type=str, default="timeseries_npz")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for t in range(args.snapshots):
        stress_event = (np.random.rand() < 0.3)  # ~30% chance of stress event in each snapshot

        arr = generate_snapshot(
            args.H, args.W,
            t,
            args.snapshots,
            base_temp=305,
            stress_event=stress_event
        )

        # metadata
        meta = {
            "date": f"2025-SeasonA-Day{t}",
            "stress_event": bool(stress_event)
        }

        transform = (444720.0, 30.0, 0.0, 3751320.0, 0.0, -30.0)
        crs = 32643

        out_path = os.path.join(args.out_dir, f"field_t{t}.npz")
        np.savez_compressed(out_path, arr=arr, transform=transform, crs=crs, meta=meta)
        print(f"Saved {out_path}")

    print("DONE: Generated realistic time-series NPZ files.")


if __name__ == "__main__":
    main()
