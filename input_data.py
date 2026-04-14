import subprocess
import sys

# Install required packages
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "osmnx", "geopandas", "pyproj", "rasterio", "matplotlib", "segmentation-models-pytorch"])

# Handle Google Colab mount (optional)
try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    print("Not running in Google Colab")

import rasterio
import numpy as np
import osmnx as ox
from rasterio.features import rasterize
from pyproj import Transformer
import geopandas as gpd

IMG_PATH = "/content/drive/MyDrive/Sentinel_AI_Project.tif"

img = rasterio.open(IMG_PATH)
bounds = img.bounds
crs = img.crs
transform = img.transform
H, W = img.height, img.width

# Read bands
blue  = img.read(1).astype(np.float32)
green = img.read(2).astype(np.float32)
red   = img.read(3).astype(np.float32)
nir   = img.read(4).astype(np.float32)

# Indices
eps = 1e-6
ndvi = (nir - red) / (nir + red + eps)
ndwi = (green - nir) / (green + nir + eps)

veg = ndvi > 0.25
water = ndwi > 0.1

# OSM fetch
transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
west, south = transformer.transform(bounds.left, bounds.bottom)
east, north = transformer.transform(bounds.right, bounds.top)

cx, cy = (east+west)/2, (north+south)/2

buildings = ox.features_from_point((cy, cx), tags={'building': True}, dist=3000)
roads = ox.graph_to_gdfs(ox.graph_from_point((cy, cx), dist=3000), nodes=False)

# Reproject
buildings = buildings.to_crs(crs)
roads = roads.to_crs(crs)

# Rasterize
def rasterize_gdf(gdf):
    shapes = [(g, 1) for g in gdf.geometry if g is not None]
    return rasterize(shapes, out_shape=(H, W), transform=transform)

road_mask = rasterize_gdf(roads)
building_mask = rasterize_gdf(buildings)

# Final mask
mask = np.zeros((H, W), dtype=np.uint8)
mask[veg] = 1
mask[water] = 2
mask[building_mask == 1] = 3
mask[road_mask == 1] = 0

# Save
with rasterio.open("/content/drive/MyDrive/training_mask_osm.tif", "w",
    driver="GTiff",
    height=H,
    width=W,
    count=1,
    dtype=mask.dtype,
    crs=crs,
    transform=transform) as dst:

    dst.write(mask, 1)

print("✅ Mask created")