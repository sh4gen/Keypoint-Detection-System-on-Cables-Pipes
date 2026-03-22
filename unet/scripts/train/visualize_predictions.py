import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass

import torch
import segmentation_models_pytorch as smp
import albumentations as A


# =========================================================
# Config
# =========================================================

@dataclass
class Config:
    data_root: str = r"C:\Users\keylo\Desktop\LAP\dataset\attention_unet_dataset"
    test_dir: str = "test"

    image_size: int = 512
    threshold: float = 0.5

    encoder_name: str = "resnet34"
    encoder_weights: str = None   # inference'te pretrained gerekmez
    in_channels: int = 3
    classes: int = 1

    model_path: str = r"C:\Users\keylo\Desktop\LAP\dataset\runs\attention_unet_resnet34\best_model.pth"
    output_dir: str = r"C:\Users\keylo\Desktop\LAP\dataset\runs\attention_unet_resnet34\test_visualizations"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CFG = Config()


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
# Transforms
# =========================================================

def get_transform(image_size: int):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0
        )
    ])


# =========================================================
# Utils
# =========================================================

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_image(path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Image okunamadı: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_mask(path: Path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Mask okunamadı: {path}")
    mask = (mask > 0).astype(np.uint8) * 255
    return mask


def get_pairs(images_dir: Path, masks_dir: Path):
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

        if mask_path is not None:
            pairs.append((img_path, mask_path))

    return pairs


def predict_mask(model, image_rgb, transform, device, threshold=0.5):
    original_h, original_w = image_rgb.shape[:2]

    augmented = transform(image=image_rgb)
    image = augmented["image"]  # HWC float32
    image = image.transpose(2, 0, 1).astype(np.float32)  # CHW
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits)
        pred = (probs > threshold).float()

    pred_mask = pred.squeeze().cpu().numpy().astype(np.uint8) * 255
    pred_mask = cv2.resize(pred_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    return pred_mask


def make_overlay(image_rgb, mask, color=(0, 255, 0), alpha=0.35):
    overlay = image_rgb.copy()
    mask_bool = mask > 0

    color_arr = np.array(color, dtype=np.uint8)
    overlay[mask_bool] = (
        overlay[mask_bool] * (1 - alpha) + color_arr * alpha
    ).astype(np.uint8)

    return overlay


def put_title(img, text):
    out = img.copy()
    cv2.putText(
        out,
        text,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )
    return out


def to_3ch(mask):
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)


# =========================================================
# Main
# =========================================================

def main():
    ensure_dir(CFG.output_dir)

    test_images = Path(CFG.data_root) / CFG.test_dir / "images"
    test_masks = Path(CFG.data_root) / CFG.test_dir / "masks"

    pairs = get_pairs(test_images, test_masks)
    print(f"Found {len(pairs)} test pairs.")

    model = build_model(CFG).to(CFG.device)
    model.load_state_dict(torch.load(CFG.model_path, map_location=CFG.device))
    model.eval()

    transform = get_transform(CFG.image_size)

    for idx, (img_path, mask_path) in enumerate(pairs):
        image_rgb = load_image(img_path)
        gt_mask = load_mask(mask_path)

        pred_mask = predict_mask(
            model=model,
            image_rgb=image_rgb,
            transform=transform,
            device=CFG.device,
            threshold=CFG.threshold
        )

        overlay = make_overlay(image_rgb, pred_mask, color=(0, 255, 0), alpha=0.35)

        # Paneller
        panel_image = put_title(image_rgb, "Image")
        panel_gt = put_title(to_3ch(gt_mask), "Ground Truth")
        panel_pred = put_title(to_3ch(pred_mask), "Prediction")
        panel_overlay = put_title(overlay, "Overlay")

        # Aynı yükseklik/genişlikte tut
        h, w = image_rgb.shape[:2]
        panel_gt = cv2.resize(panel_gt, (w, h))
        panel_pred = cv2.resize(panel_pred, (w, h))
        panel_overlay = cv2.resize(panel_overlay, (w, h))

        top = np.concatenate([panel_image, panel_gt], axis=1)
        bottom = np.concatenate([panel_pred, panel_overlay], axis=1)
        canvas = np.concatenate([top, bottom], axis=0)

        out_name = f"{idx:03d}_{img_path.stem}_viz.jpg"
        out_path = Path(CFG.output_dir) / out_name

        cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        print(f"[OK] saved: {out_path.name}")

    print(f"\nAll visualizations saved to:\n{CFG.output_dir}")


if __name__ == "__main__":
    main()