import os
import cv2
import time
import random
import numpy as np
from pathlib import Path
from dataclasses import dataclass

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
import segmentation_models_pytorch as smp


# =========================================================
# Config
# =========================================================

@dataclass
class Config:
    data_root: str = r"C:\Users\keylo\Desktop\LAP\dataset\attention_unet_dataset"

    train_dir: str = "train"
    val_dir: str = "valid"
    test_dir: str = "test"

    image_size: int = 512
    batch_size: int = 4
    num_workers: int = 2

    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42

    encoder_name: str = "resnet34"
    encoder_weights: str = "imagenet"
    in_channels: int = 3
    classes: int = 1

    threshold: float = 0.5

    save_dir: str = r"C:\Users\keylo\Desktop\LAP\dataset\runs\attention_unet_resnet34"
    best_model_name: str = "best_model.pth"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CFG = Config()


# =========================================================
# Utils
# =========================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def get_image_mask_pairs(images_dir: Path, masks_dir: Path):
    image_paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    pairs = []

    for img_path in image_paths:
        stem = img_path.stem

        candidate_masks = [
            masks_dir / f"{stem}.png",
            masks_dir / f"{stem}.jpg",
            masks_dir / f"{stem}.jpeg",
        ]

        mask_path = None
        for cm in candidate_masks:
            if cm.exists():
                mask_path = cm
                break

        if mask_path is None:
            continue

        pairs.append((img_path, mask_path))

    return pairs


def compute_iou_dice(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7):
    preds = preds.float()
    targets = targets.float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + eps) / (union + eps)

    dice = (2 * intersection + eps) / (preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + eps)

    return iou.mean().item(), dice.mean().item()


# =========================================================
# Dataset
# =========================================================

class CableDataset(Dataset):
    def __init__(self, pairs, transform=None):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Image okunamadı: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Mask okunamadı: {mask_path}")

        # binary mask
        mask = (mask > 0).astype(np.uint8)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # HWC -> CHW
        image = image.transpose(2, 0, 1).astype(np.float32)
        mask = np.expand_dims(mask, axis=0).astype(np.float32)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)


# =========================================================
# Augmentations
# =========================================================

def get_train_transform(image_size: int):
    return A.Compose([
        A.Resize(image_size, image_size),

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=20, border_mode=cv2.BORDER_REFLECT_101, p=0.5),

        A.ColorJitter(
            brightness=0.20,
            contrast=0.20,
            saturation=0.20,
            hue=0.08,
            p=0.40
        ),

        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.30
        ),

        A.ISONoise(
            color_shift=(0.01, 0.04),
            intensity=(0.08, 0.30),
            p=0.20
        ),

        A.GaussNoise(
            std_range=(0.01, 0.04),
            p=0.15
        ),

        A.OneOf([
            A.Defocus(radius=(1, 2), alias_blur=0.10, p=1.0),
            A.MotionBlur(blur_limit=(3, 5), p=1.0),
            A.GaussianBlur(blur_limit=(3, 3), p=1.0),
        ], p=0.20),

        A.ImageCompression(
            quality_range=(80, 95),
            p=0.20
        ),

        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0
        ),
    ])


def get_val_transform(image_size: int):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0
        ),
    ])


# =========================================================
# Loss
# =========================================================

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets, eps=1e-7):
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        denom = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2.0 * intersection + eps) / (denom + eps)
        dice_loss = 1.0 - dice.mean()

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# =========================================================
# Model
# =========================================================

def build_model(cfg: Config):
    model = smp.Unet(
        encoder_name=cfg.encoder_name,
        encoder_weights=cfg.encoder_weights,
        in_channels=cfg.in_channels,
        classes=cfg.classes,
        decoder_attention_type="scse",
    )
    return model


# =========================================================
# Train / Eval
# =========================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc="Train", leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()
    running_loss = 0.0
    iou_scores = []
    dice_scores = []

    pbar = tqdm(loader, desc="Eval", leave=False)
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()

        iou, dice = compute_iou_dice(preds, masks)

        running_loss += loss.item() * images.size(0)
        iou_scores.append(iou)
        dice_scores.append(dice)

    epoch_loss = running_loss / len(loader.dataset)
    mean_iou = float(np.mean(iou_scores))
    mean_dice = float(np.mean(dice_scores))

    return epoch_loss, mean_iou, mean_dice


# =========================================================
# Main
# =========================================================

def main():
    set_seed(CFG.seed)
    ensure_dir(CFG.save_dir)

    train_images = Path(CFG.data_root) / CFG.train_dir / "images"
    train_masks = Path(CFG.data_root) / CFG.train_dir / "masks"

    val_images = Path(CFG.data_root) / CFG.val_dir / "images"
    val_masks = Path(CFG.data_root) / CFG.val_dir / "masks"

    test_images = Path(CFG.data_root) / CFG.test_dir / "images"
    test_masks = Path(CFG.data_root) / CFG.test_dir / "masks"

    train_pairs = get_image_mask_pairs(train_images, train_masks)
    val_pairs = get_image_mask_pairs(val_images, val_masks)
    test_pairs = get_image_mask_pairs(test_images, test_masks)

    print(f"Train samples: {len(train_pairs)}")
    print(f"Val samples  : {len(val_pairs)}")
    print(f"Test samples : {len(test_pairs)}")

    train_ds = CableDataset(train_pairs, transform=get_train_transform(CFG.image_size))
    val_ds = CableDataset(val_pairs, transform=get_val_transform(CFG.image_size))
    test_ds = CableDataset(test_pairs, transform=get_val_transform(CFG.image_size))

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True
    )

    model = build_model(CFG).to(CFG.device)
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG.lr,
        weight_decay=CFG.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5
    )

    best_val_dice = -1.0
    best_model_path = Path(CFG.save_dir) / CFG.best_model_name

    print(f"\nUsing device: {CFG.device}")
    print("Training started...\n")

    for epoch in range(1, CFG.epochs + 1):
        start = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, CFG.device)
        val_loss, val_iou, val_dice = evaluate(
            model, val_loader, criterion, CFG.device, threshold=CFG.threshold
        )

        scheduler.step(val_dice)

        elapsed = time.time() - start
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch:03d}/{CFG.epochs}] | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_iou={val_iou:.4f} | "
            f"val_dice={val_dice:.4f} | "
            f"lr={current_lr:.6f} | "
            f"time={elapsed:.1f}s"
        )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Best model saved: {best_model_path}")

    print("\nTraining finished.")
    print(f"Best val dice: {best_val_dice:.4f}")

    print("\nLoading best model for test evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=CFG.device))

    test_loss, test_iou, test_dice = evaluate(
        model, test_loader, criterion, CFG.device, threshold=CFG.threshold
    )

    print("\n=== TEST RESULTS ===")
    print(f"test_loss : {test_loss:.4f}")
    print(f"test_iou  : {test_iou:.4f}")
    print(f"test_dice : {test_dice:.4f}")


if __name__ == "__main__":
    main()