import rasterio
import numpy as np
import os

IMG_PATH = "/content/drive/MyDrive/Sentinel_AI_Project.tif"
MASK_PATH = "/content/drive/MyDrive/training_mask_osm.tif"

# Save tiles in Colab (temporary but fine for training)
os.makedirs("tiles/images", exist_ok=True)
os.makedirs("tiles/masks", exist_ok=True)

img = rasterio.open(IMG_PATH).read()
mask = rasterio.open(MASK_PATH).read(1)

tile_size = 128
count = 0

for y in range(0, mask.shape[0] - tile_size, tile_size):
    for x in range(0, mask.shape[1] - tile_size, tile_size):

        img_tile = img[:, y:y+tile_size, x:x+tile_size]
        mask_tile = mask[y:y+tile_size, x:x+tile_size]

        np.save(f"tiles/images/img_{count}.npy", img_tile)
        np.save(f"tiles/masks/mask_{count}.npy", mask_tile)

        count += 1

print("✅ Tiles created:", count)