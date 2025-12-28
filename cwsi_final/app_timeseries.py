import streamlit as st
import numpy as np
import pandas as pd
import os
import tempfile
from matplotlib import pyplot as plt

from utils import (
    load_npz, map_channels, emissivity_from_ndvi, apply_emissivity_correction,
    compute_envelopes, compute_cwsi, class_map, global_summary,
    grid_patches_stats, detect_hotspots, get_date_from_meta
)

from io_helpers import save_tif, save_png, resize_for_animation, save_gif, save_mp4
from ml import export_ml_features_from_field_stats, train_xgboost


# ---------------------------------------
# STREAMLIT PAGE CONFIG
# ---------------------------------------
st.set_page_config(
    layout="wide",
    page_title="CWSI Timeseries Analyzer",
    page_icon="🌾"
)


# ---------------------------------------
# SESSION STATE INITIALIZATION
# ---------------------------------------
if "npz_files" not in st.session_state:
    st.session_state.npz_files = []            # uploaded raw files
if "dates" not in st.session_state:
    st.session_state.dates = []                # extracted date labels
if "data_cache" not in st.session_state:
    st.session_state.data_cache = {}           # per-date precomputed arrays
if "patch_stats" not in st.session_state:
    st.session_state.patch_stats = None
if "global_stats" not in st.session_state:
    st.session_state.global_stats = None
if "outputs_dir" not in st.session_state:
    st.session_state.outputs_dir = os.path.join(os.getcwd(), "outputs")
if not os.path.exists(st.session_state.outputs_dir):
    os.makedirs(st.session_state.outputs_dir, exist_ok=True)


# ---------------------------------------
# SIDEBAR MENU
# ---------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    [
        "Load Data",
        "Map Viewer",
        "Compare Two Dates",
        "Animate Time Series",
        "Time-Series Metrics",
        "Downloads"
    ]
)

# Pcommom paramter controls
st.sidebar.subheader("Processing Parameters")
nbins = st.sidebar.number_input("NDVI bins", min_value=5, max_value=80, value=20)
hot_pct = st.sidebar.slider("Hot percentile", 90, 99, 95)
cold_pct = st.sidebar.slider("Cold percentile", 1, 20, 5)
min_den = st.sidebar.number_input("Min Th-Tc (K)", 0.1, 5.0, 0.5)
patch_size = st.sidebar.number_input("Patch size", 16, 256, 32)
hotspot_thresh = st.sidebar.slider("Hotspot threshold (CWSI)", 0.5, 0.95, 0.7)
min_hotspot_pixels = st.sidebar.number_input("Min hotspot pixels", 5, 200, 10)


# ---------------------------------------
# GLOBAL HELPER: PROCESS ONE NPZ FILE
# ---------------------------------------
def process_npz_file(uploaded_file, date_label):
    """
    Load NPZ → compute NDVI, emissivity, LST, Th/Tc, CWSI, class map, summaries.
    The output dictionary is cached in session_state.data_cache[date].
    """

    tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    tmp.write(uploaded_file.getbuffer())
    tmp.flush()
    tmp.close()

    # load data
    data = load_npz(tmp.name)
    arr = data["arr"]
    transform = data["transform"]
    crs = data["crs"]
    meta = data["meta"]

    ch = map_channels(arr)

    # NDVI
    ndvi = ch.get("ndvi")
    if ndvi is None:
        nir = ch.get("opt_nir")
        red = ch.get("opt_red")
        ndvi = (nir - red) / (nir + red + 1e-6)

    # emissivity
    emiss = ch.get("emissivity")
    if emiss is None:
        emiss = emissivity_from_ndvi(ndvi)

    # lst corrected
    lst = ch.get("tir_hr", ch.get("tir_lr_upsampled"))
    lst_corr = apply_emissivity_correction(lst, emiss)

    # envelope functions
    Thf, Tcf, _, _, _ = compute_envelopes(
        lst_corr, ndvi,
        nbins=nbins, hot_pct=hot_pct, cold_pct=cold_pct
    )

    # CWSI
    cwsi, Th_map, Tc_map = compute_cwsi(
        lst_corr, ndvi, Thf, Tcf, min_den=min_den
    )
    classes = class_map(cwsi)

    # Global stats
    gstats = global_summary(cwsi)
    gstats["date"] = date_label

    # Patch grid stats
    pstats = grid_patches_stats(cwsi, ndvi, patch_size=patch_size)
    pstats["date"] = date_label

    # Hotspots
    hotspots = detect_hotspots(
        cwsi,
        transform=transform,
        crs=crs,
        threshold=hotspot_thresh,
        min_area_pixels=min_hotspot_pixels
    )

    # Save everything in cache
    st.session_state.data_cache[date_label] = {
        "arr": arr,
        "lst_corr": lst_corr,
        "ndvi": ndvi,
        "cwsi": cwsi,
        "Th": Th_map,
        "Tc": Tc_map,
        "classes": classes,
        "global_stats": gstats,
        "patch_stats": pstats,
        "transform": transform,
        "crs": crs,
        "meta": meta,
        "hotspots": hotspots
    }


# ---------------------------------------
# NAVIGATION ROUTING
# ---------------------------------------
if page == "Load Data":
    st.header("Load NPZ Time-Series Data")
    st.info("Upload multiple NPZ files representing the same field over time.")

    uploaded = st.file_uploader(
        "Upload NPZ snapshots",
        type=["npz"],
        accept_multiple_files=True
    )

    if uploaded:
        st.session_state.npz_files = uploaded
        st.session_state.dates = []
        st.session_state.global_stats = []
        st.session_state.patch_stats = []

        # Process all NPZ files
        with st.spinner("Processing NPZ files..."):
            for i, f in enumerate(uploaded):
                # extract date
                d = load_npz(f)["meta"]
                dlabel = get_date_from_meta(d, i)
                st.session_state.dates.append(dlabel)

                process_npz_file(f, dlabel)

                # aggregate global stats
                st.session_state.global_stats.append(
                    st.session_state.data_cache[dlabel]["global_stats"]
                )

                # aggregate patch stats
                st.session_state.patch_stats.append(
                    st.session_state.data_cache[dlabel]["patch_stats"]
                )

        st.success("Data loaded and processed successfully!")

        st.write("### Loaded Dates:", st.session_state.dates)

    st.stop()  # end Load Data section


# ---------------------------------------
# BELOW THIS: SUBPAGES (2B, 2C, 2D WILL FILL HERE)
# ---------------------------------------
# ---------------------------------------
# Step 2B: Map Viewer Page (per-date visualization + PNG download)
# ---------------------------------------
if page == "Map Viewer":
    st.header("Map Viewer — browse maps per date")

    if not st.session_state.data_cache:
        st.info("No data loaded. Go to 'Load Data' and upload NPZ snapshots first.")
        st.stop()

    # list of date labels available
    dates = st.session_state.dates
    n_dates = len(dates)

    # slider or dropdown to pick a date
    pick_mode = st.radio("Date selector mode", ["Slider", "Dropdown"], horizontal=True)

    if pick_mode == "Slider":
        date_idx = st.slider("Select date index", 0, max(0, n_dates - 1), 0)
        date_label = dates[date_idx]
    else:
        date_label = st.selectbox("Select date", options=dates, index=0)
        date_idx = dates.index(date_label)

    st.markdown(f"**Viewing snapshot:** `{date_label}`")

    # Layer selector
    layer = st.selectbox("Select layer", [
        "CWSI",
        "NDVI",
        "LST_corrected",
        "Class map",
        "Th",
        "Tc"
    ])

    # visualization parameters
    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader("Display options")
        cmap_choice = st.selectbox("Colormap", ["inferno", "viridis", "RdYlGn", "magma", "coolwarm", "Greens"])
        show_colorbar = st.checkbox("Show colorbar", value=True)
        stretch = st.checkbox("Auto stretch (vmin/vmax)", value=True)

        # optional manual vmin/vmax for more control
        manual_vmin = st.number_input("vmin (leave 0 for auto)", value=0.0, format="%.3f")
        manual_vmax = st.number_input("vmax (leave 0 for auto)", value=0.0, format="%.3f")

    with col2:
        st.subheader("Map display")
        # fetch cached arrays
        cache = st.session_state.data_cache[date_label]
        arr_display = None
        cmap = cmap_choice

        if layer == "CWSI":
            arr_display = cache["cwsi"]
            if stretch and (manual_vmin == 0 and manual_vmax == 0):
                vmin, vmax = 0.0, 1.0
            elif manual_vmin != 0 or manual_vmax != 0:
                vmin = manual_vmin if manual_vmin != 0 else np.nanmin(arr_display)
                vmax = manual_vmax if manual_vmax != 0 else np.nanmax(arr_display)
            else:
                vmin, vmax = np.nanmin(arr_display), np.nanmax(arr_display)
        elif layer == "NDVI":
            arr_display = cache["ndvi"]
            if stretch and (manual_vmin == 0 and manual_vmax == 0):
                vmin, vmax = -1.0, 1.0
            elif manual_vmin != 0 or manual_vmax != 0:
                vmin = manual_vmin if manual_vmin != 0 else np.nanmin(arr_display)
                vmax = manual_vmax if manual_vmax != 0 else np.nanmax(arr_display)
            else:
                vmin, vmax = np.nanmin(arr_display), np.nanmax(arr_display)
        elif layer == "LST_corrected":
            arr_display = cache["lst_corr"]
            if stretch and (manual_vmin == 0 and manual_vmax == 0):
                vmin, vmax = np.nanpercentile(arr_display, 2), np.nanpercentile(arr_display, 98)
            elif manual_vmin != 0 or manual_vmax != 0:
                vmin = manual_vmin if manual_vmin != 0 else np.nanmin(arr_display)
                vmax = manual_vmax if manual_vmax != 0 else np.nanmax(arr_display)
            else:
                vmin, vmax = np.nanmin(arr_display), np.nanmax(arr_display)
        elif layer == "Class map":
            arr_display = cache["classes"]
            vmin, vmax = np.nanmin(arr_display), np.nanmax(arr_display)
            cmap = "tab10"
        elif layer == "Th":
            arr_display = cache["Th"]
            if stretch and (manual_vmin == 0 and manual_vmax == 0):
                vmin, vmax = np.nanpercentile(arr_display, 2), np.nanpercentile(arr_display, 98)
            else:
                vmin, vmax = (manual_vmin if manual_vmin != 0 else np.nanmin(arr_display),
                              manual_vmax if manual_vmax != 0 else np.nanmax(arr_display))
        elif layer == "Tc":
            arr_display = cache["Tc"]
            if stretch and (manual_vmin == 0 and manual_vmax == 0):
                vmin, vmax = np.nanpercentile(arr_display, 2), np.nanpercentile(arr_display, 98)
            else:
                vmin, vmax = (manual_vmin if manual_vmin != 0 else np.nanmin(arr_display),
                              manual_vmax if manual_vmax != 0 else np.nanmax(arr_display))

        # plotting
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(arr_display, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"{layer} — {date_label}")
        ax.axis("off")
        if show_colorbar:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)

        st.pyplot(fig)

        # --- Save PNG & download ---
        out_dir = st.session_state.outputs_dir
        png_name = f"{layer.replace(' ','_')}_{date_label}.png"
        png_path = os.path.join(out_dir, png_name)
        try:
            save_png(png_path, fig)
            with open(png_path, "rb") as f:
                png_bytes = f.read()
            st.download_button(
                label="Download PNG",
                data=png_bytes,
                file_name=png_name,
                mime="image/png"
            )
            st.success(f"Saved PNG to outputs/ as {png_name}")
        except Exception as e:
            st.error(f"Failed to save/download PNG: {e}")

    # show some quick stats for selected date and layer
    st.markdown("---")
    st.subheader("Quick stats")
    gs = st.session_state.data_cache[date_label]["global_stats"]
    st.write(gs)


# ---------------------------------------
# Step 2C: Compare Two Dates Side-by-Side
# ---------------------------------------
if page == "Compare Two Dates":
    st.header("Compare Two Dates — side-by-side")

    if not st.session_state.data_cache or len(st.session_state.dates) < 2:
        st.info("Load at least two NPZ snapshots in 'Load Data' to use comparison.")
        st.stop()

    dates = st.session_state.dates

    col_a, col_b = st.columns(2)
    with col_a:
        date_a = st.selectbox("Select Date A (left)", options=dates, index=0)
    with col_b:
        date_b = st.selectbox("Select Date B (right)", options=dates, index=min(1, len(dates)-1))

    if date_a == date_b:
        st.warning("Date A and Date B are the same — select two different dates for comparison.")

    layer = st.selectbox("Layer to compare", ["CWSI", "NDVI", "LST_corrected", "Class map", "Th", "Tc"])

    st.markdown("**Options:**")
    sync_cbar = st.checkbox("Synchronize color scale between A and B", value=True)
    show_diff = st.checkbox("Also show difference (B - A)", value=False)
    cmap_choice = st.selectbox("Colormap", ["inferno", "viridis", "RdYlGn", "magma", "coolwarm", "Greens"])

    cache_a = st.session_state.data_cache[date_a]
    cache_b = st.session_state.data_cache[date_b]

    def pick_layer(cache, layer_name):
        if layer_name == "CWSI":
            return cache["cwsi"]
        if layer_name == "NDVI":
            return cache["ndvi"]
        if layer_name == "LST_corrected":
            return cache["lst_corr"]
        if layer_name == "Class map":
            return cache["classes"]
        if layer_name == "Th":
            return cache["Th"]
        if layer_name == "Tc":
            return cache["Tc"]
        return None

    img_a = pick_layer(cache_a, layer)
    img_b = pick_layer(cache_b, layer)

    # compute vmin/vmax
    if sync_cbar:
        combo = np.concatenate([np.ravel(img_a[np.isfinite(img_a)]), np.ravel(img_b[np.isfinite(img_b)])])
        if combo.size == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = float(np.nanpercentile(combo, 2)), float(np.nanpercentile(combo, 98))
    else:
        def get_vrange(img):
            if img is None or not np.isfinite(img).any():
                return 0.0, 1.0
            return float(np.nanpercentile(img, 2)), float(np.nanpercentile(img, 98))
        vmin_a, vmax_a = get_vrange(img_a)
        vmin_b, vmax_b = get_vrange(img_b)

    # plotting
    ncols = 3 if show_diff else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5*ncols, 5))

    # left
    ax = axes[0]
    im0 = ax.imshow(img_a, cmap=cmap_choice, vmin=(vmin if sync_cbar else vmin_a), vmax=(vmax if sync_cbar else vmax_a))
    ax.set_title(f"A: {date_a}")
    ax.axis("off")

    # right
    ax = axes[1]
    im1 = ax.imshow(img_b, cmap=cmap_choice, vmin=(vmin if sync_cbar else vmin_b), vmax=(vmax if sync_cbar else vmax_b))
    ax.set_title(f"B: {date_b}")
    ax.axis("off")

    # colorbar (single)
    if sync_cbar:
        fig.colorbar(im1, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
    else:
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # difference
    if show_diff:
        diff = None
        try:
            diff = img_b - img_a
        except Exception:
            diff = np.full_like(img_a, np.nan)
        ax = axes[2]
        im2 = ax.imshow(diff, cmap="bwr", vmin=-np.nanpercentile(np.abs(diff[np.isfinite(diff)]), 98),
                        vmax=np.nanpercentile(np.abs(diff[np.isfinite(diff)]), 98))
        ax.set_title("Difference (B - A)")
        ax.axis("off")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    st.pyplot(fig)

    # combined PNG export
    out_dir = st.session_state.outputs_dir
    os.makedirs(out_dir, exist_ok=True)
    combo_name = f"compare_{layer}_{date_a}_vs_{date_b}.png".replace(" ", "_")
    combo_path = os.path.join(out_dir, combo_name)
    try:
        save_png(combo_path, fig)
        with open(combo_path, "rb") as fh:
            btn = st.download_button(
                label="Download comparison PNG",
                data=fh,
                file_name=combo_name,
                mime="image/png"
            )
        st.success(f"Saved comparison PNG: outputs/{combo_name}")
    except Exception as e:
        st.error(f"Failed to save comparison PNG: {e}")

    # small numeric comparison table
    st.markdown("---")
    st.subheader("Numeric comparison — quick stats")
    def quick_stats(arr):
        if arr is None or not np.isfinite(arr).any():
            return {"mean": None, "median": None, "p90": None}
        return {
            "mean": float(np.nanmean(arr)),
            "median": float(np.nanmedian(arr)),
            "p90": float(np.nanpercentile(arr, 90))
        }

    stats_a = quick_stats(img_a)
    stats_b = quick_stats(img_b)

    comp_df = pd.DataFrame({"metric": ["mean", "median", "p90"], "A": [stats_a["mean"], stats_a["median"], stats_a["p90"]],
                            "B": [stats_b["mean"], stats_b["median"], stats_b["p90"]]})
    st.table(comp_df)


# ---------------------------------------
# Step 2D: Animation, Metrics, Downloads pages
# ---------------------------------------

# small helper: convert matplotlib fig to RGB ndarray
def fig_to_rgb_array(fig):
    """
    Convert a Matplotlib figure to an (H,W,3) uint8 RGB numpy array.
    """
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = fig.canvas.tostring_rgb()
    arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 3))
    return arr


# --------------------------
# Animation Page
# --------------------------
if page == "Animate Time Series":
    st.header("Animate CWSI / NDVI / LST over time (GIF/MP4)")

    if not st.session_state.data_cache or len(st.session_state.dates) < 2:
        st.info("Load at least two NPZ snapshots to use animation.")
        st.stop()

    dates = st.session_state.dates

    # parameters
    layer_choice = st.selectbox("Layer to animate", ["CWSI", "NDVI", "LST_corrected"])
    out_type = st.selectbox("Output type", ["GIF", "MP4"])
    fps = st.slider("Frames per second (fps)", 1, 6, 2)
    max_anim_size = st.number_input("Max animation size (px) for longest side (adaptive)", min_value=128, max_value=1024, value=512, step=64)

    st.markdown("**Preview frames** (first, middle, last):")
    preview_cols = st.columns(3)
    # preview images for 0, mid, last
    idxs = [0, len(dates)//2, len(dates)-1]
    previews = []
    for i, idx in enumerate(idxs):
        lab = dates[idx]
        cache = st.session_state.data_cache[lab]
        if layer_choice == "CWSI":
            img = cache["cwsi"]
        elif layer_choice == "NDVI":
            img = cache["ndvi"]
        else:
            img = cache["lst_corr"]
        figp, axp = plt.subplots(figsize=(3,3))
        im0 = axp.imshow(img, cmap="inferno" if layer_choice=="CWSI" else ("Greens" if layer_choice=="NDVI" else "magma"))
        axp.set_title(lab); axp.axis("off")
        preview_cols[i].pyplot(figp)
        previews.append((figp, img))

    if st.button("Generate animation"):
        st.info("Generating animation — this may take a few seconds depending on number of frames and size.")
        frames = []
        try:
            for idx in range(len(dates)):
                lab = dates[idx]
                cache = st.session_state.data_cache[lab]
                if layer_choice == "CWSI":
                    img = cache["cwsi"]
                    cmap = "inferno"
                elif layer_choice == "NDVI":
                    img = cache["ndvi"]
                    cmap = "RdYlGn"
                else:
                    img = cache["lst_corr"]
                    cmap = "magma"

                # create figure for this frame
                figf, axf = plt.subplots(figsize=(6,6))
                # autoscale vmin/vmax across all frames to keep animation stable
                # compute vmin/vmax using percentiles across finite pixels
                all_vals = np.ravel(img[np.isfinite(img)])
                if all_vals.size == 0:
                    vmin, vmax = 0.0, 1.0
                else:
                    vmin, vmax = float(np.nanpercentile(all_vals, 2)), float(np.nanpercentile(all_vals, 98))

                axf.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
                axf.set_title(f"{lab}")
                axf.axis("off")

                # convert to RGB array
                arr_rgb = fig_to_rgb_array(figf)
                plt.close(figf)

                # adaptively resize for animation
                arr_rgb_small = resize_for_animation(arr_rgb, max_size=max_anim_size)
                frames.append(arr_rgb_small)

            out_dir = st.session_state.outputs_dir
            os.makedirs(out_dir, exist_ok=True)
            anim_basename = f"anim_{layer_choice}.{'gif' if out_type=='GIF' else 'mp4'}"
            anim_path = os.path.join(out_dir, anim_basename)

            if out_type == "GIF":
                save_gif(anim_path, frames, fps=fps)
            else:
                save_mp4(anim_path, frames, fps=fps)

            st.success(f"Animation saved: outputs/{anim_basename}")
            with open(anim_path, "rb") as fh:
                st.download_button("Download animation", data=fh, file_name=anim_basename, mime=("image/gif" if out_type=="GIF" else "video/mp4"))

        except Exception as e:
            st.error(f"Animation generation failed: {e}")


# --------------------------
# Time-Series Metrics Page
# --------------------------
if page == "Time-Series Metrics":
    st.header("Time-Series Metrics")

    if not st.session_state.global_stats:
        st.info("No global stats computed — load data first.")
        st.stop()

    # global time series
    gst = pd.DataFrame(st.session_state.global_stats).set_index("date").sort_index()
    st.subheader("Global scene-level timeseries")
    st.line_chart(gst[["mean_cwsi", "median_cwsi", "p90_cwsi"]])

    st.write("Global stats table")
    st.dataframe(gst)

    # patch-level exploration
    st.subheader("Patch-level trends (grid patches)")
    if not st.session_state.patch_stats:
        st.info("No patch stats available.")
    else:
        # patch_stats is a list of per-date dataframes; combine
        patches_long = pd.concat(st.session_state.patch_stats, ignore_index=True)
        patches_long = patches_long.sort_values(["patch_id", "date"])
        patch_ids = sorted(patches_long["patch_id"].unique())

        sel_patch = st.selectbox("Select patch_id", options=patch_ids, index=0)
        sel_patch_df = patches_long[patches_long["patch_id"] == sel_patch].set_index("date").sort_index()

        if sel_patch_df.empty:
            st.info("No data for this patch.")
        else:
            st.line_chart(sel_patch_df[["mean_cwsi", "p90_cwsi"]])
            st.dataframe(sel_patch_df)


# --------------------------
# Downloads Page
# --------------------------
if page == "Downloads":
    st.header("Available output files")
    out_dir = st.session_state.outputs_dir
    os.makedirs(out_dir, exist_ok=True)

    files = []
    for root, _, filenames in os.walk(out_dir):
        for fn in filenames:
            files.append(os.path.join(root, fn))

    if not files:
        st.info("No outputs found yet. Run the processing and exports in Load Data / Map Viewer / Compare / Animation pages.")
    else:
        df_files = []
        for fpath in sorted(files):
            size = os.path.getsize(fpath)
            df_files.append({"path": fpath, "name": os.path.basename(fpath), "size_kb": round(size / 1024, 1)})

        st.dataframe(pd.DataFrame(df_files))

        for entry in df_files:
            p = entry["path"]
            name = entry["name"]
            try:
                with open(p, "rb") as fh:
                    st.download_button(label=f"Download {name}", data=fh, file_name=name)
            except Exception as e:
                st.error(f"Failed to open {name}: {e}")

    st.markdown("---")
    st.write("You can also save outputs directly from the Map Viewer / Compare / Animation pages when generating images and animations.")
