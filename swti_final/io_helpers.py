
import numpy as np
from matplotlib import pyplot as plt
import json
from shapely.geometry import mapping

def save_array_png(path, arr, cmap='viridis', vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)

def features_to_geojson(features, path):
    fc = {"type":"FeatureCollection","features":[]}
    for feat in features:
        geom = mapping(feat['geometry'])
        props = {k:v for k,v in feat.items() if k!='geometry'}
        fc['features'].append({"type":"Feature","geometry":geom,"properties":props})
    with open(path,'w') as f:
        json.dump(fc, f)
    return path
