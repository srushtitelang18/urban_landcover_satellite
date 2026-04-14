import subprocess
import sys

# Install required packages
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch", "rasterio", "matplotlib"])

import torch
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import segmentation_models_pytorch as smp
import os
import warnings
warnings.filterwarnings("ignore")

# ─── Config ──────────────────────────────────────────────────────────────────
MODEL_PATH = "/content/drive/MyDrive/model_best.pth"
IMG_PATH   = "/content/drive/MyDrive/Sentinel_AI_Project.tif"
OUT_DIR    = "/content/drive/MyDrive/predictions"
TILE_SIZE  = 128
ALPHA      = 0.45

os.makedirs(OUT_DIR, exist_ok=True)

# ─── Device ──────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ─── Load image info ─────────────────────────────────────────────────────────
with rasterio.open(IMG_PATH) as src:
    IN_CHANNELS = src.count
    H_full, W_full = src.height, src.width
    print(f"Image: {IN_CHANNELS} bands, {H_full}×{W_full} px")

# ─── Load checkpoint ─────────────────────────────────────────────────────────
# ✅ FIX 1: weights_only=False  (PyTorch 2.6 changed default to True)
checkpoint  = torch.load(MODEL_PATH, map_location=device, weights_only=False)

# ✅ FIX 2: read config saved during training
IN_CHANNELS = checkpoint["in_channels"]   # 8
NUM_CLASSES = checkpoint["num_classes"]   # 3
MODEL_TYPE  = checkpoint["model_type"]    # "unet"

print(f"Checkpoint → in_channels={IN_CHANNELS}, num_classes={NUM_CLASSES}, model={MODEL_TYPE}")

# ─── Build model ─────────────────────────────────────────────────────────────
if MODEL_TYPE == "unet":
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=IN_CHANNELS,
        classes=NUM_CLASSES,
    )
elif MODEL_TYPE == "deeplab":
    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=IN_CHANNELS,
        classes=NUM_CLASSES,
    )

# ✅ FIX 3: load only the weights dict, not the whole checkpoint
model.load_state_dict(checkpoint["model_state"])
model = model.to(device)
model.eval()
print("✅ Model loaded")

# ─── Class colors + names (3 classes) ────────────────────────────────────────
# ─── Class colors + names (4 classes — matches your checkpoint) ──────────────
CLASS_COLORS = np.array([
    [80,  80,  80,  255],   # 0 road/background — gray
    [56, 168,  0,   255],   # 1 vegetation      — green
    [0,  112, 255,  255],   # 2 water            — blue
    [215, 75,  0,   255],   # 3 building         — orange-red
], dtype=np.uint8)

CLASS_NAMES = ["Road / background", "Vegetation", "Water", "Building"]

# ─── Load full image ─────────────────────────────────────────────────────────
with rasterio.open(IMG_PATH) as src:
    img_full  = src.read().astype(np.float32)   # (C, H, W)
    profile   = src.profile
    transform = src.transform
    crs       = src.crs

# ─── Pad image so it divides evenly into tiles ───────────────────────────────
C, H, W = img_full.shape
pad_h = (TILE_SIZE - H % TILE_SIZE) % TILE_SIZE
pad_w = (TILE_SIZE - W % TILE_SIZE) % TILE_SIZE

img_padded = np.pad(img_full, ((0,0),(0,pad_h),(0,pad_w)), mode='reflect')
pred_full  = np.zeros((img_padded.shape[1], img_padded.shape[2]), dtype=np.uint8)

Hp, Wp = img_padded.shape[1], img_padded.shape[2]
total_tiles = (Hp // TILE_SIZE) * (Wp // TILE_SIZE)
done = 0

print(f"Running inference on {total_tiles} tiles ...")

# ─── Tile-based inference ─────────────────────────────────────────────────────
with torch.no_grad():
    for y in range(0, Hp, TILE_SIZE):
        for x in range(0, Wp, TILE_SIZE):
            tile = img_padded[:, y:y+TILE_SIZE, x:x+TILE_SIZE]
            tile = tile / (tile.max() + 1e-6)

            tile_t = torch.from_numpy(tile).unsqueeze(0).to(device)  # (1,C,128,128)
            logits = model(tile_t)                                    # (1,3,128,128)
            pred   = logits.argmax(dim=1).squeeze(0)                  # (128,128)
            pred_full[y:y+TILE_SIZE, x:x+TILE_SIZE] = pred.cpu().numpy()

            done += 1
            if done % 50 == 0 or done == total_tiles:
                print(f"  {done}/{total_tiles} tiles done", end="\r")

# Crop back to original size
pred_full = pred_full[:H, :W]
print(f"\n✅ Inference complete. Shape: {pred_full.shape}")

# ─── Class distribution ───────────────────────────────────────────────────────
print("\nPredicted class distribution:")
unique, counts = np.unique(pred_full, return_counts=True)
for cls, cnt in zip(unique, counts):
    print(f"  Class {cls} ({CLASS_NAMES[cls]}): {cnt:,} px  ({100*cnt/pred_full.size:.1f}%)")

# ─── Helper: mask → RGBA ──────────────────────────────────────────────────────
def mask_to_rgba(mask, colors):
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    for cls_id, color in enumerate(colors):
        rgba[mask == cls_id] = color
    return rgba

pred_rgba = mask_to_rgba(pred_full, CLASS_COLORS)

# ─── Helper: Sentinel bands → RGB preview ────────────────────────────────────
def sentinel_to_rgb(img, r=2, g=1, b=0):
    rgb = img[[r, g, b]].transpose(1, 2, 0)
    p2, p98 = np.percentile(rgb, (2, 98))
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1)
    return (rgb * 255).astype(np.uint8)

rgb_preview = sentinel_to_rgb(img_full)

# ─── Legend patches ───────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(color=CLASS_COLORS[i][:3] / 255, label=CLASS_NAMES[i])
    for i in range(NUM_CLASSES)
]

# ─── Output 1: Pure prediction map ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 10))
ax.imshow(pred_rgba)
ax.set_title("Segmentation prediction", fontsize=14, fontweight='bold')
ax.axis("off")
ax.legend(handles=legend_patches, loc="lower right",
          fontsize=10, framealpha=0.85, edgecolor='gray')
out1 = f"{OUT_DIR}/prediction_map.png"
fig.savefig(out1, dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print(f"✅ Saved: {out1}")

# ─── Output 2: Overlay ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 10))
ax.imshow(rgb_preview)
ax.imshow(pred_rgba, alpha=ALPHA)
ax.set_title("Overlay: RGB + prediction", fontsize=14, fontweight='bold')
ax.axis("off")
ax.legend(handles=legend_patches, loc="lower right",
          fontsize=10, framealpha=0.85, edgecolor='gray')
out2 = f"{OUT_DIR}/overlay.png"
fig.savefig(out2, dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print(f"✅ Saved: {out2}")

# ─── Output 3: Side-by-side comparison ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(22, 10))
axes[0].imshow(rgb_preview)
axes[0].set_title("Original (RGB)", fontsize=14, fontweight='bold')
axes[0].axis("off")
axes[1].imshow(pred_rgba)
axes[1].set_title("Model prediction", fontsize=14, fontweight='bold')
axes[1].axis("off")
axes[1].legend(handles=legend_patches, loc="lower right",
               fontsize=10, framealpha=0.85, edgecolor='gray')
plt.tight_layout(pad=1.5)
out3 = f"{OUT_DIR}/comparison.png"
fig.savefig(out3, dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print(f"✅ Saved: {out3}")

# ─── Output 4: Per-class confidence maps ─────────────────────────────────────
print("\nGenerating confidence maps ...")
prob_full = np.zeros((NUM_CLASSES, H, W), dtype=np.float32)

with torch.no_grad():
    for y in range(0, Hp, TILE_SIZE):
        for x in range(0, Wp, TILE_SIZE):
            tile   = img_padded[:, y:y+TILE_SIZE, x:x+TILE_SIZE]
            tile   = tile / (tile.max() + 1e-6)
            tile_t = torch.from_numpy(tile).unsqueeze(0).to(device)
            logits = model(tile_t)
            probs  = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            y_end = min(y + TILE_SIZE, H)
            x_end = min(x + TILE_SIZE, W)
            prob_full[:, y:y_end, x:x_end] = probs[:, :y_end-y, :x_end-x]

cmaps = ['Greys_r', 'Greens', 'Blues', 'Oranges']
fig, axes = plt.subplots(1, NUM_CLASSES, figsize=(18, 6))
for i in range(NUM_CLASSES):
    im = axes[i].imshow(prob_full[i], cmap=cmaps[i], vmin=0, vmax=1)
    axes[i].set_title(f"P({CLASS_NAMES[i]})", fontsize=11, fontweight='bold')
    axes[i].axis("off")
    plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

plt.suptitle("Per-class prediction confidence", fontsize=13, fontweight='bold')
plt.tight_layout()
out4 = f"{OUT_DIR}/confidence_maps.png"
fig.savefig(out4, dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print(f"✅ Saved: {out4}")

# ─── Output 5: GeoTIFF (opens in QGIS / ArcGIS) ──────────────────────────────
profile.update(count=1, dtype=rasterio.uint8, compress='lzw')
out5 = f"{OUT_DIR}/prediction.tif"
with rasterio.open(out5, "w", **profile) as dst:
    dst.write(pred_full.astype(np.uint8), 1)
print(f"✅ Saved GeoTIFF: {out5}")

# ─── Summary ─────────────────────────────────────────────────────────────────
print("\n── All outputs saved to Drive ───────────────────────────────────────")
for path, label in [
    (out1, "Prediction map  "),
    (out2, "Overlay         "),
    (out3, "Comparison      "),
    (out4, "Confidence maps "),
    (out5, "GeoTIFF         "),
]:
    print(f"  {label} → {path}")