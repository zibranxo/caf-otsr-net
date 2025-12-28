
- **CAF-OTSRNet** improves spatial fidelity of thermal imagery  
- **CWSI** converts improved thermal maps into crop stress metrics  
- **SWTI** converts improved thermal maps into thermal pollution intelligence  

---

## CAF-OTSRNet (Core Model)

**CAF-OTSRNet** performs **optical-guided thermal super-resolution** using a physics-aware, safety-conscious design.

### Key Features

#### Optical–Thermal Alignment (Explicit handling of satellite misregistration)
- Global affine alignment
- Learned deformable fine alignment

#### Dual Encoders
- Thermal encoder for structural preservation
- Optical encoder for VIS, SWIR, and spectral indices

#### Cross-Attention Fusion
- Thermal features query optical features
- Prevents optical dominance and hallucinated detail

#### Texture-Guided Safety Mechanism
- Suppresses unsafe high-frequency hallucinations
- Enforces physically plausible reconstruction

#### Progressive Laplacian Decoder
- Multi-scale residual reconstruction
- Preserves thermal gradients and continuity

#### Uncertainty Estimation
- Pixel-wise uncertainty output
- Enables reliability-aware downstream analysis

> The model is designed for **scientific reliability**, not cosmetic super-resolution.

---

## CWSI — Crop Water Stress Intelligence

CWSI transforms super-resolved thermal data into **crop water stress metrics** using established agro-hydrological principles.

### Capabilities
- NDVI-based emissivity estimation
- Emissivity-corrected Land Surface Temperature (LST)
- Hot / cold envelope modeling
- **Crop Water Stress Index (CWSI)** computation
- Stress classification maps
- Patch-level spatial statistics
- Time-series analysis across dates
- Hotspot detection and export
- PNG, CSV, GIF, and MP4 outputs

### Use Cases
- Precision agriculture
- Irrigation planning
- Crop stress monitoring
- Field-level temporal analysis

---

## SWTI — Surface Water Thermal Intelligence

SWTI detects and analyzes **thermal anomalies and plumes** in rivers, lakes, and coastal waters.

### Capabilities
- Water masking using NDVI
- Hole filling and optional NL-means denoising
- Multi-scale thermal gradient analysis
- Thermal Anomaly Index (TAI)
- Adaptive plume detection
- Morphological refinement
- Plume polygon extraction (GeoJSON)
- Plume centroid tracking
- Single-date and time-series analysis modes

### Use Cases
- Thermal pollution monitoring
- Power plant discharge detection
- Environmental compliance
- Aquatic ecosystem analysis

---

## Why Super-Resolution Matters

Thermal sensors are typically **lower resolution** than optical sensors.  
CAF-OTSRNet improves:

- Boundary sharpness
- Gradient reliability
- Small-scale anomaly detection
- Temporal consistency

This directly improves the **accuracy and stability** of both **CWSI** and **SWTI** outputs.

---

## Tech Stack

- **Deep Learning:** PyTorch  
- **Applications:** Streamlit, Gradio  
- **Image Processing:** NumPy, SciPy, scikit-image, OpenCV  
- **Geospatial:** Rasterio, GeoPandas, Shapely  
- **Machine Learning:** Scikit-learn, XGBoost  
- **Visualization:** Matplotlib  

---

## Repository Layout

```
├── caf-otsrnet.ipynb        # Model architecture & experiments
├── app.py                  # Gradio demo for SR inference
├── cwsi_final/                   # Crop Water Stress app
├── swti_final/                   # Surface Water Thermal Intelligence app
├── model_weights.pth       # Trained model weights
├── requirements.txt
└── README.md
```

---

## How to Run

### 1️. Install dependencies
```bash
pip install -r requirements.txt
```

### 2️. Run CWSI
```bash
streamlit run cwsi_app.py
```

### 3️. Run SWTI
```bash
streamlit run swti_app.py
```

### 4️. Run Super-Resolution Demo
```bash
python app.py
```

---

## Key Contributions

- Alignment-aware optical-guided thermal super-resolution  
- Safety-aware texture fusion  
- Uncertainty-aware thermal reconstruction  
- End-to-end environmental intelligence pipeline  
- Real-world applicability to agriculture and water systems  

---

## License

This project is intended for **research and educational use**.  
Please contact the author for commercial or operational deployment.


