import subprocess
import sys

# Install required packages
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch", "rasterio"])

import rasterio
import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

# -----------------------------
# DEVICE (GPU)
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# DATASET
# -----------------------------
class TileDataset(Dataset):
    def __init__(self):
        self.files = sorted(os.listdir("/content/tiles/images"))  # sorted for consistency

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = np.load(f"/content/tiles/images/{self.files[idx]}").astype(np.float32)
        mask = np.load(f"/content/tiles/masks/{self.files[idx].replace('img', 'mask')}")

        img = img / (img.max() + 1e-6)

        # ✅ FIX 1: Ensure mask is explicitly int64 BEFORE converting to tensor
        mask = mask.astype(np.int64)

        return torch.from_numpy(img), torch.from_numpy(mask)

# ✅ FIX 2: Actually instantiate dataset and loader (was missing!)
dataset = TileDataset()
loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

# -----------------------------
# CHOOSE MODEL
# -----------------------------
MODEL_TYPE = "unet"   # change to "deeplab"

# ✅ FIX 3: dataset[0] now works because dataset is defined above
in_channels = dataset[0][0].shape[0]

if MODEL_TYPE == "unet":
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=in_channels,
        classes=4
    )
elif MODEL_TYPE == "deeplab":
    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=in_channels,
        classes=4
    )

model = model.to(device)

# -----------------------------
# LOSS + OPTIMIZER
# -----------------------------
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -----------------------------
# TRAINING LOOP
# -----------------------------
EPOCHS = 5

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)  # will be torch.int64 ✅

        preds = model(imgs)       # shape: (B, 4, H, W)
        loss = loss_fn(preds, masks)  # masks must be (B, H, W) int64

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

# -----------------------------
# SAVE MODEL
# -----------------------------
torch.save(model.state_dict(), "/content/drive/MyDrive/model.pth")
print("✅ Model saved to Drive")

# ─── Load and tile satellite imagery ────────────────────────────────────────
IMG_PATH  = "/content/drive/MyDrive/Sentinel_AI_Project.tif"
MASK_PATH = "/content/drive/MyDrive/training_mask_osm.tif"
TILE_SIZE = 128
MIN_VALID = 0.7   # discard tiles where >30% of pixels are nodata/black

os.makedirs("tiles/images", exist_ok=True)
os.makedirs("tiles/masks",  exist_ok=True)

img_src  = rasterio.open(IMG_PATH)
mask_src = rasterio.open(MASK_PATH)

img  = img_src.read()                    # (C, H, W)  float or uint16
mask = mask_src.read(1).astype(np.uint8) # (H, W)

print(f"Image shape: {img.shape}  |  Mask shape: {mask.shape}")
print(f"Image dtype: {img.dtype}  |  Mask classes: {np.unique(mask)}")

H, W  = mask.shape
count = 0
skipped = 0

for y in range(0, H - TILE_SIZE + 1, TILE_SIZE):
    for x in range(0, W - TILE_SIZE + 1, TILE_SIZE):

        img_tile  = img[:, y:y+TILE_SIZE, x:x+TILE_SIZE].astype(np.float32)
        mask_tile = mask[y:y+TILE_SIZE, x:x+TILE_SIZE]

        # FIX: skip near-empty tiles (all zeros = nodata)
        if img_tile.max() < 1e-6:
            skipped += 1
            continue

        # Normalize per-tile (safe with small epsilon)
        img_tile = img_tile / (img_tile.max() + 1e-6)

        np.save(f"tiles/images/img_{count}.npy",  img_tile)
        np.save(f"tiles/masks/mask_{count}.npy",  mask_tile)
        count += 1

print(f"\n✅ Tiles saved: {count}  |  Skipped (empty): {skipped}")
print(f"   Image channels: {img.shape[0]}  ← use this as in_channels in the model")