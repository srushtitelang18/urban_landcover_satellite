import subprocess
import sys

# Install required packages
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch"])

import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, random_split
import segmentation_models_pytorch as smp

# ─── Device ──────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ─── Dataset ─────────────────────────────────────────────────────────────────
class TileDataset(Dataset):
    def __init__(self, img_dir="tiles/images", mask_dir="tiles/masks"):
        self.img_dir  = img_dir
        self.mask_dir = mask_dir
        self.files    = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        img  = np.load(f"{self.img_dir}/{name}").astype(np.float32)
        mask = np.load(f"{self.mask_dir}/{name.replace('img', 'mask')}")

        img  = img / (img.max() + 1e-6)
        mask = mask.astype(np.int64)   # required by CrossEntropyLoss

        return torch.from_numpy(img), torch.from_numpy(mask)

# ─── Instantiate dataset ─────────────────────────────────────────────────────
dataset = TileDataset()
print(f"Total tiles: {len(dataset)}")

sample_img, sample_mask = dataset[0]
IN_CHANNELS = sample_img.shape[0]

# ✅ FIX: auto-detect number of classes from actual mask values
all_classes = set()
for i in range(len(dataset)):
    _, m = dataset[i]
    all_classes.update(m.numpy().flatten().tolist())
NUM_CLASSES = len(all_classes)

print(f"Input channels : {IN_CHANNELS}")
print(f"Mask classes   : {sorted(all_classes)}  →  NUM_CLASSES = {NUM_CLASSES}")

# ─── Train / val split ───────────────────────────────────────────────────────
val_size   = max(1, int(0.1 * len(dataset)))
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=8,  shuffle=True,
                          num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=8,  shuffle=False,
                          num_workers=2, pin_memory=True)

print(f"Train: {train_size} tiles  |  Val: {val_size} tiles")

# ─── Model ───────────────────────────────────────────────────────────────────
MODEL_TYPE = "unet"   # or "deeplab"

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

model = model.to(device)
print(f"Model : {MODEL_TYPE}  |  Params: {sum(p.numel() for p in model.parameters()):,}")

# ─── Class weights (handles imbalanced land-cover) ───────────────────────────
# Count pixels per class across all masks
print("Computing class weights ...")
class_counts = np.zeros(NUM_CLASSES, dtype=np.float64)
for i in range(len(dataset)):
    _, m = dataset[i]
    for c in range(NUM_CLASSES):
        class_counts[c] += (m.numpy() == c).sum()

# Inverse-frequency weighting
total = class_counts.sum()
class_weights = total / (NUM_CLASSES * class_counts + 1e-6)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
print(f"Class weights: {class_weights.cpu().numpy().round(3)}")

# ─── Loss + optimiser ────────────────────────────────────────────────────────
loss_fn   = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ✅ FIX: removed verbose=True (dropped in PyTorch 2.4+)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

# ─── Metric helper (mean IoU) ─────────────────────────────────────────────────
def mean_iou(preds, masks, num_classes):
    """preds: (B,H,W) int  |  masks: (B,H,W) int"""
    ious = []
    for c in range(num_classes):
        tp = ((preds == c) & (masks == c)).sum().item()
        fp = ((preds == c) & (masks != c)).sum().item()
        fn = ((preds != c) & (masks == c)).sum().item()
        denom = tp + fp + fn
        ious.append(tp / denom if denom > 0 else float('nan'))
    valid = [v for v in ious if not np.isnan(v)]
    return np.mean(valid) if valid else 0.0

# ─── Training loop ───────────────────────────────────────────────────────────
EPOCHS        = 20
best_val_loss = float("inf")
best_epoch    = 0
history       = {"train_loss": [], "val_loss": [], "val_iou": []}

print("\n── Training ─────────────────────────────────────────────────────────")

for epoch in range(EPOCHS):

    # ── Train ────────────────────────────────────────────────────────────────
    model.train()
    train_loss = 0.0

    for imgs, masks in train_loader:
        imgs  = imgs.to(device)
        masks = masks.to(device)          # int64, shape (B, H, W)

        preds = model(imgs)               # (B, NUM_CLASSES, H, W)
        loss  = loss_fn(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ── Validate ─────────────────────────────────────────────────────────────
    model.eval()
    val_loss = 0.0
    val_iou  = 0.0

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs  = imgs.to(device)
            masks = masks.to(device)

            preds     = model(imgs)
            val_loss += loss_fn(preds, masks).item()

            pred_cls  = preds.argmax(dim=1)   # (B, H, W)
            val_iou  += mean_iou(pred_cls, masks, NUM_CLASSES)

    val_loss /= len(val_loader)
    val_iou  /= len(val_loader)

    # ── LR scheduler step (replaces verbose=True) ────────────────────────────
    old_lr = optimizer.param_groups[0]['lr']
    scheduler.step(val_loss)
    new_lr = optimizer.param_groups[0]['lr']
    lr_tag = f"  ↓ LR {old_lr:.6f}→{new_lr:.6f}" if new_lr < old_lr else ""

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
          f"Train loss: {train_loss:.4f} | "
          f"Val loss: {val_loss:.4f} | "
          f"Val mIoU: {val_iou:.4f} | "
          f"LR: {new_lr:.6f}{lr_tag}")

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_iou"].append(val_iou)

    # ── Save best model ───────────────────────────────────────────────────────
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch    = epoch + 1
        torch.save({
            "epoch"       : epoch + 1,
            "model_state" : model.state_dict(),
            "optimizer"   : optimizer.state_dict(),
            "val_loss"    : val_loss,
            "val_iou"     : val_iou,
            "num_classes" : NUM_CLASSES,
            "in_channels" : IN_CHANNELS,
            "model_type"  : MODEL_TYPE,
        }, "/content/drive/MyDrive/model_best.pth")
        print(f"  ✅ Best model saved  (epoch {best_epoch}, val loss {best_val_loss:.4f})")

# ─── Save final model ─────────────────────────────────────────────────────────
torch.save({
    "epoch"       : EPOCHS,
    "model_state" : model.state_dict(),
    "num_classes" : NUM_CLASSES,
    "in_channels" : IN_CHANNELS,
    "model_type"  : MODEL_TYPE,
}, "/content/drive/MyDrive/model_final.pth")
print(f"\n✅ Final model saved")
print(f"   Best epoch: {best_epoch}  |  Best val loss: {best_val_loss:.4f}")

# ─── Plot training curves ─────────────────────────────────────────────────────
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

ax1.plot(history["train_loss"], label="Train loss")
ax1.plot(history["val_loss"],   label="Val loss")
ax1.axvline(best_epoch - 1, color='red', linestyle='--', alpha=0.5, label=f"Best (ep {best_epoch})")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.set_title("Loss curves"); ax1.legend()

ax2.plot(history["val_iou"], color='green', label="Val mIoU")
ax2.axvline(best_epoch - 1, color='red', linestyle='--', alpha=0.5)
ax2.set_xlabel("Epoch"); ax2.set_ylabel("mIoU")
ax2.set_title("Validation mIoU"); ax2.legend()

plt.tight_layout()
plt.savefig("/content/drive/MyDrive/training_curves.png", dpi=120)
plt.show()
print("✅ Training curves saved")