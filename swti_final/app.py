
import streamlit as st
import numpy as np, tempfile, os, json
from utils import load_npz_any, smooth_then_gradient, multi_scale_gradient, adaptive_plume_detection, denoise_nl_means_masked, fill_holes_and_mask, normalize_for_display
from io_helpers import features_to_geojson
from matplotlib import pyplot as plt

st.set_page_config(layout='wide', page_title='SWTI)')
st.title('SWTI — Surface Water Thermal Intelligence')

menu = st.sidebar.selectbox('Mode', ['Single Date', 'Time-series', 'Plume Dashboard', 'Settings'])

sigma = st.sidebar.slider('Gradient smoothing sigma (px)', 1.0, 4.0, 2.0, 0.5)
use_nlme = st.sidebar.checkbox('Use NL-Means denoising (slower)', value=False)
nlme_hscale = st.sidebar.slider('NL-means h-scale', 0.2, 2.0, 0.8, 0.1)
tile_size = st.sidebar.number_input('Plume tile size (px)', 32, 256, 64)
z_thresh = st.sidebar.slider('Tile z-threshold', 0.5, 3.0, 1.5, 0.1)
min_area = st.sidebar.number_input('Min plume area (px)', 10, 500, 50)
dilation = st.sidebar.number_input('Morph dilation (px)', 0, 8, 2)

outdir = os.path.join(os.getcwd(),'outputs'); os.makedirs(outdir, exist_ok=True)

def display_img(arr, title='Image', cmap='viridis', vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(6,6)); im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax); ax.set_title(title); ax.axis('off'); fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); st.pyplot(fig)

if menu == 'Single Date':
    st.header('Single-date analysis')
    npz_file = st.file_uploader('Upload NPZ (single date)', type=['npz'])
    if npz_file is not None:
        tmp = tempfile.NamedTemporaryFile(suffix='.npz', delete=False); tmp.write(npz_file.getbuffer()); tmp.flush(); tmp.close()
        arr, meta = load_npz_any(tmp.name)
        st.sidebar.write('meta:', meta)
        # arr expected shape (C,H,W)
        if arr.ndim != 3:
            st.error('Loaded array is not 3D. Found shape: {}'.format(arr.shape))
        else:
            lst = arr[1]
            ndvi = arr[10] if arr.shape[0] > 10 else None
            water_mask = (ndvi < 0) if ndvi is not None else (np.ones_like(lst, dtype=bool))
            lst_filled = fill_holes_and_mask(lst, water_mask)
            if use_nlme:
                lst_dn = denoise_nl_means_masked(lst_filled, mask=water_mask, h_scale=nlme_hscale)
            else:
                lst_dn = lst_filled
            display_img(lst, title='LST raw')
            display_img(lst_dn, title='LST cleaned & denoised')
            comb_grad, grads, sm = multi_scale_gradient(lst_dn, water_mask, sigmas=(1.0,2.5,4.0), weights=None, use_nlme=use_nlme)
            display_img(sm, title='Smoothed LST (last scale)')
            display_img(comb_grad, title='Combined multi-scale gradient', cmap='viridis')
            ambient = np.nanmedian(lst_dn[water_mask])
            tai = lst_dn - ambient
            display_img(tai, title='TAI (LST - ambient)')
            features = adaptive_plume_detection(tai, water_mask, tile_size=tile_size, z_thresh=z_thresh, min_area=min_area, dilation=dilation)
            st.subheader(f'Detected plumes: {len(features)}')
            if len(features) > 0:
                import pandas as pd
                tab = pd.DataFrame([{'label':f['label'],'area_px':f['area_pixels'],'max_val':f['max_val'],'mean_val':f['mean_val']} for f in features])
                st.dataframe(tab)
                gj_path = os.path.join(outdir,'plumes_single.geojson'); features_to_geojson(features, gj_path)
                with open(gj_path,'rb') as fh: st.download_button('Download plumes (GeoJSON)', fh, file_name='plumes_single.geojson', mime='application/geo+json')
            else:
                st.info('No plumes found')

elif menu == 'Time-series':
    st.header('Multi-date analysis (time-series)')
    npz_files = st.file_uploader('Upload multiple NPZ files (2+)', type=['npz'], accept_multiple_files=True)
    if npz_files and len(npz_files) > 1:
        records = []; plume_records = []
        for uploaded in npz_files:
            tmp = tempfile.NamedTemporaryFile(suffix='.npz', delete=False); tmp.write(uploaded.getbuffer()); tmp.flush(); tmp.close()
            arr, meta = load_npz_any(tmp.name)
            date = meta.get('date', uploaded.name)
            lst = arr[1]; ndvi = arr[10] if arr.shape[0] > 10 else None
            water_mask = (ndvi < 0) if ndvi is not None else (np.ones_like(lst, dtype=bool))
            lst_filled = fill_holes_and_mask(lst, water_mask)
            lst_dn = denoise_nl_means_masked(lst_filled, mask=water_mask, h_scale=nlme_hscale) if use_nlme else lst_filled
            ambient = np.nanmedian(lst_dn[water_mask])
            tai = lst_dn - ambient
            features = adaptive_plume_detection(tai, water_mask, tile_size=tile_size, z_thresh=z_thresh, min_area=min_area, dilation=dilation)
            records.append({'date':date, 'mean_tai': float(np.nanmean(tai[water_mask])) if np.isfinite(tai).any() else None, 'p90_tai': float(np.nanpercentile(tai[water_mask],90)) if np.isfinite(tai).any() else None})
            for f in features:
                poly = f['geometry']
                try:
                    cent = poly.centroid.coords[0]
                except Exception:
                    cent = (None,None)
                plume_records.append({'date':date, 'label':f['label'], 'centroid_x': float(cent[0]) if cent[0] is not None else None, 'centroid_y': float(cent[1]) if cent[1] is not None else None, 'max_val': f['max_val']})
        import pandas as pd
        gts = pd.DataFrame(records).set_index('date').sort_index()
        st.subheader('Global time series'); st.line_chart(gts[['mean_tai','p90_tai']]); st.dataframe(gts)
        pr = pd.DataFrame(plume_records)
        st.subheader('Detected plume centroids (sample)'); st.dataframe(pr.head(50))
        csvp = os.path.join(outdir,'time_series_global.csv'); gts.to_csv(csvp)
        csvp2 = os.path.join(outdir,'plume_centroids.csv'); pr.to_csv(csvp2, index=False)
        with open(csvp,'rb') as fh: st.download_button('Download global time-series CSV', fh, file_name='time_series_global.csv', mime='text/csv')
        with open(csvp2,'rb') as fh: st.download_button('Download plume centroids CSV', fh, file_name='plume_centroids.csv', mime='text/csv')

elif menu == 'Plume Dashboard':
    st.header('Plume dashboard & exports')
    geojson_file = st.file_uploader('Upload plume GeoJSON (from Single Date output) or skip to use demo', type=['geojson','json'])
    if geojson_file is not None:
        feats = json.load(geojson_file)
        st.write('Feature count:', len(feats.get('features',[])))
        st.json(feats.get('features',[])[0:5])
        with open(os.path.join(outdir,'plumes_uploaded.geojson'),'wb') as f: f.write(geojson_file.getbuffer())
        with open(os.path.join(outdir,'plumes_uploaded.geojson'),'rb') as fh: st.download_button('Re-download uploaded', fh, file_name='plumes_uploaded.geojson', mime='application/geo+json')

elif menu == 'Settings':
    st.header('Settings & help')
    st.markdown('Now supports NPZ files with arrays stored under keys like "patch". The loader will pick the first 3D array available if no named key matches.')
