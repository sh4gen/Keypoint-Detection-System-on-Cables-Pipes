"""
Cable Segmentation - Safer Augmentation Pipeline (albumentations)
=================================================================
Giriş : COCO format JSON + görüntü klasörü
Çıkış : augmented görüntüler + yeni COCO JSON

Kurulum:
    pip install albumentations opencv-python

Kullanım:
    python augment_cables_albu_safe.py \
        --images_dir  /dataset/train \
        --coco_json   /dataset/train/_annotations_coco.json \
        --output_dir  /dataset/train_augmented \
        --n_aug       6 \
        --seed        42 \
        --visualize
"""
"""
import cv2
import numpy as np
import json
import argparse
import random
from pathlib import Path

import albumentations as A


# ─────────────────────────────────────────────
#  Augmentation pipeline
# ─────────────────────────────────────────────

def build_transform():
    return A.Compose([
        # Geometric
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(
            limit=25,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5
        ),
        A.RandomResizedCrop(
            size=(640, 640),
            scale=(0.90, 1.00),
            ratio=(0.95, 1.05),
            p=0.20
        ),

        # Very light optical distortion
        A.OpticalDistortion(
            distort_limit=0.05,
            p=0.10
        ),

        # Color / illumination
        A.ColorJitter(
            brightness=0.20,
            contrast=0.20,
            saturation=0.20,
            hue=0.08,
            p=0.45
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.30
        ),

        # Noise
        A.ISONoise(
            color_shift=(0.01, 0.04),
            intensity=(0.08, 0.30),
            p=0.22
        ),
        A.GaussNoise(
            std_range=(0.01, 0.04),
            p=0.15
        ),

        # Blur family: one blur type, controlled and visible
        A.OneOf([
            A.Defocus(radius=(2, 5), alias_blur=0.2, p=1.0),
            A.MotionBlur(blur_limit=(5, 9), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.35),

        # Compression
        A.ImageCompression(
            quality_range=(75, 95),
            p=0.22
        ),
    ])


def build_val_transform():
    return A.Compose([])


# ─────────────────────────────────────────────
#  COCO utils
# ─────────────────────────────────────────────

def polygons_to_mask(segmentations, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)

    # RLE format
    if isinstance(segmentations, dict) and "counts" in segmentations:
        counts = segmentations["counts"]
        if isinstance(counts, list):
            flat = np.zeros(height * width, dtype=np.uint8)
            idx, val = 0, 0
            for c in counts:
                flat[idx:idx + c] = val
                idx += c
                val = 1 - val
            mask = flat.reshape((height, width), order="F")
        return mask

    # Polygon format
    for seg in segmentations:
        if isinstance(seg, dict):
            continue
        if len(seg) < 6:
            continue
        pts = np.array(seg, dtype=np.float32).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)

    return mask


def mask_to_polygons(mask, min_area=20):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue

        # cnt shape: (N, 1, 2) → (N, 2)
        cnt = cnt.reshape(-1, 2)

        # polygon için en az 3 nokta lazım
        if len(cnt) < 3:
            continue

        poly = cnt.flatten().tolist()

        # En az 3 nokta = 6 sayı
        if len(poly) < 6:
            continue

        polygons.append(poly)

    return polygons


def compute_bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any():
        return [0, 0, 0, 0]

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [int(cmin), int(rmin), int(cmax - cmin + 1), int(rmax - rmin + 1)]


# ─────────────────────────────────────────────
#  Ana augment döngüsü
# ─────────────────────────────────────────────

def run_augmentation(images_dir, coco_json, output_dir, n_aug=6, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    img_out = output_dir / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    with open(coco_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    ann_map = {}
    for ann in coco["annotations"]:
        ann_map.setdefault(ann["image_id"], []).append(ann)

    transform = build_transform()

    new_images = []
    new_annotations = []
    new_img_id = 0
    new_ann_id = 0

    for img_info in coco["images"]:
        fname = img_info["file_name"]
        fname_alt = fname.replace(".rf.", "_rf_").replace(".jpg.", "_jpg_")

        img_path = images_dir / fname
        if not img_path.exists():
            img_path = images_dir / fname_alt

        if not img_path.exists():
            print(f"[SKIP] {fname}")
            continue

        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"[SKIP] okunamadı: {fname}")
            continue

        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        anns = ann_map.get(img_info["id"], [])

        instance_masks = []
        for ann in anns:
            m = polygons_to_mask(ann["segmentation"], h, w)
            instance_masks.append((ann, m))

        all_masks = [m for _, m in instance_masks]

        # Orijinali kaydet
        cv2.imwrite(
            str(img_out / fname),
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 95]
        )

        new_images.append({
            "id": new_img_id,
            "file_name": fname,
            "width": w,
            "height": h
        })

        for ann, m in instance_masks:
            polys = mask_to_polygons(m)
            if polys:
                new_annotations.append({
                    "id": new_ann_id,
                    "image_id": new_img_id,
                    "category_id": ann["category_id"],
                    "segmentation": polys,
                    "bbox": compute_bbox(m),
                    "area": int(m.sum()),
                    "iscrowd": 0,
                })
                new_ann_id += 1

        new_img_id += 1

        # Augmented
        for aug_idx in range(n_aug):
            result = transform(image=image, masks=all_masks)
            aug_image = result["image"]
            aug_masks = result["masks"]

            ah, aw = aug_image.shape[:2]
            stem = Path(fname).stem
            aug_fname = f"{stem}_aug{aug_idx:03d}.jpg"

            cv2.imwrite(
                str(img_out / aug_fname),
                cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 95]
            )

            new_images.append({
                "id": new_img_id,
                "file_name": aug_fname,
                "width": aw,
                "height": ah
            })

            for (ann, _), aug_m in zip(instance_masks, aug_masks):
                aug_m_bin = (aug_m > 0).astype(np.uint8)
                polys = mask_to_polygons(aug_m_bin)
                if not polys:
                    continue

                new_annotations.append({
                    "id": new_ann_id,
                    "image_id": new_img_id,
                    "category_id": ann["category_id"],
                    "segmentation": polys,
                    "bbox": compute_bbox(aug_m_bin),
                    "area": int(aug_m_bin.sum()),
                    "iscrowd": 0,
                })
                new_ann_id += 1

            new_img_id += 1

        print(f"✓ {fname[:55]}  →  {n_aug} aug")

    new_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": coco["categories"],
        "images": new_images,
        "annotations": new_annotations,
    }

    out_json = output_dir / "_annotations_aug_coco.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(new_coco, f, indent=2)

    print(f"\n{'─' * 52}")
    print(f"Orijinal görüntü  : {len(coco['images'])}")
    print(f"Aug. görüntü      : {len(new_images) - len(coco['images'])}")
    print(f"Toplam görüntü    : {len(new_images)}")
    print(f"Toplam annotation : {len(new_annotations)}")
    print(f"JSON              : {out_json}")
    print(f"Görüntüler        : {img_out}")

    return new_coco


# ─────────────────────────────────────────────
#  Sanity check — mask overlay görselleri
# ─────────────────────────────────────────────

def visualize_sample(coco_data, output_dir, n_samples=6):
    vis_dir = Path(output_dir) / "vis_check"
    vis_dir.mkdir(exist_ok=True)

    ann_map = {}
    for ann in coco_data["annotations"]:
        ann_map.setdefault(ann["image_id"], []).append(ann)

    aug_imgs = [img for img in coco_data["images"] if "_aug" in img["file_name"]]
    if len(aug_imgs) == 0:
        print("Augmented görsel yok, visualize atlandı.")
        return

    samples = random.sample(aug_imgs, min(n_samples, len(aug_imgs)))

    for img_info in samples:
        img_path = Path(output_dir) / "images" / img_info["file_name"]
        if not img_path.exists():
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        h, w = image.shape[:2]
        overlay = image.copy()

        for ann in ann_map.get(img_info["id"], []):
            m = polygons_to_mask(ann["segmentation"], h, w)
            overlay[m == 1] = (
                overlay[m == 1] * 0.45 + np.array([0, 220, 80]) * 0.55
            ).astype(np.uint8)

            contours, _ = cv2.findContours(
                m.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE
            )
            cv2.drawContours(overlay, contours, -1, (0, 255, 60), 2)

        cv2.imwrite(str(vis_dir / img_info["file_name"]), overlay)

    print(f"Görsel kontrol: {vis_dir}  ({len(samples)} örnek)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--coco_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_aug", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    print(f"albumentations {A.__version__}")
    print(f"n_aug={args.n_aug}\n")

    new_coco = run_augmentation(
        args.images_dir,
        args.coco_json,
        args.output_dir,
        n_aug=args.n_aug,
        seed=args.seed,
    )

    if args.visualize:
        visualize_sample(new_coco, args.output_dir)
"""

"""
Cable Segmentation - Safer Augmentation Pipeline (albumentations)
=================================================================
Giriş : COCO format JSON + görüntü klasörü
Çıkış : augmented görüntüler + yeni COCO JSON

Kurulum:
    pip install albumentations opencv-python

Kullanım:
    python augment_cables_albu_safe.py \
        --images_dir  /dataset/train \
        --coco_json   /dataset/train/_annotations_coco.json \
        --output_dir  /dataset/train_augmented \
        --n_aug       6 \
        --seed        42 \
        --visualize
"""

"""
Cable Segmentation - Safer Augmentation Pipeline (albumentations)
=================================================================
Giriş : COCO format JSON + görüntü klasörü
Çıkış : augmented görüntüler + yeni COCO JSON

Kurulum:
    pip install albumentations opencv-python pycocotools
    # Windows'ta gerekirse:
    # pip install pycocotools-windows

Kullanım:
    python augment_cables_albu_safe.py \
        --images_dir  /dataset/train/images \
        --coco_json   /dataset/train/_annotations.coco.json \
        --output_dir  /dataset/train_augmented \
        --n_aug       6 \
        --seed        42 \
        --visualize
"""

import cv2
import numpy as np
import json
import argparse
import random
from pathlib import Path

import albumentations as A

try:
    from pycocotools import mask as maskUtils
except ImportError:
    maskUtils = None


# ─────────────────────────────────────────────
#  Augmentation pipeline
# ─────────────────────────────────────────────

def build_transform():
    return A.Compose([
        # Geometric
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(
            limit=25,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5
        ),
        A.RandomResizedCrop(
            size=(640, 640),
            scale=(0.90, 1.00),
            ratio=(0.95, 1.05),
            p=0.20
        ),

        # Very light optical distortion
        A.OpticalDistortion(
            distort_limit=0.05,
            p=0.10
        ),

        # Color / illumination
        A.ColorJitter(
            brightness=0.20,
            contrast=0.20,
            saturation=0.20,
            hue=0.08,
            p=0.45
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.30
        ),

        # Noise
        A.ISONoise(
            color_shift=(0.01, 0.04),
            intensity=(0.08, 0.30),
            p=0.22
        ),
        A.GaussNoise(
            std_range=(0.01, 0.04),
            p=0.15
        ),

        # Blur family
        A.OneOf([
            A.Defocus(radius=(1, 2), alias_blur=0.10, p=1.0),
            A.MotionBlur(blur_limit=(3, 5), p=1.0),
            A.GaussianBlur(blur_limit=(3, 3), p=1.0),
        ], p=0.22),

        # Compression
        A.ImageCompression(
            quality_range=(75, 95),
            p=0.22
        ),
    ])


def build_val_transform():
    return A.Compose([])


# ─────────────────────────────────────────────
#  COCO utils
# ─────────────────────────────────────────────

def polygons_to_mask(segmentations, height, width):
    """
    COCO segmentation -> binary mask
    Destek:
      - Polygon
      - Uncompressed RLE
      - Compressed RLE (pycocotools ile)
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    # RLE format
    if isinstance(segmentations, dict) and "counts" in segmentations:
        counts = segmentations["counts"]

        # Uncompressed RLE: counts list
        if isinstance(counts, list):
            flat = np.zeros(height * width, dtype=np.uint8)
            idx, val = 0, 0
            for c in counts:
                flat[idx:idx + c] = val
                idx += c
                val = 1 - val
            mask = flat.reshape((height, width), order="F")
            return mask

        # Compressed RLE: counts string
        if isinstance(counts, str):
            if maskUtils is None:
                raise ImportError(
                    "Compressed RLE decode için pycocotools gerekli.\n"
                    "Kur: pip install pycocotools\n"
                    "Windows'ta olmazsa: pip install pycocotools-windows"
                )

            rle = {
                "counts": counts.encode("utf-8") if isinstance(counts, str) else counts,
                "size": segmentations["size"]
            }
            decoded = maskUtils.decode(rle)

            # bazen (H, W, 1) dönebilir
            if decoded.ndim == 3:
                decoded = decoded[:, :, 0]

            mask = (decoded > 0).astype(np.uint8)
            return mask

        return mask

    # Polygon format
    for seg in segmentations:
        if isinstance(seg, dict):
            continue
        if len(seg) < 6:
            continue
        pts = np.array(seg, dtype=np.float32).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)

    return mask


def mask_to_polygons(mask, min_area=20):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return []

    # sadece en büyük contour
    cnt = max(contours, key=cv2.contourArea)

    if cv2.contourArea(cnt) < min_area:
        return []

    cnt = cnt.reshape(-1, 2)
    if len(cnt) < 3:
        return []

    poly = cnt.flatten().tolist()
    if len(poly) < 6:
        return []

    return [poly]


def compute_bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any():
        return [0, 0, 0, 0]

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [int(cmin), int(rmin), int(cmax - cmin + 1), int(rmax - rmin + 1)]


# ─────────────────────────────────────────────
#  Main augmentation loop
# ─────────────────────────────────────────────

def run_augmentation(images_dir, coco_json, output_dir, n_aug=6, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    img_out = output_dir / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    with open(coco_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    ann_map = {}
    for ann in coco["annotations"]:
        ann_map.setdefault(ann["image_id"], []).append(ann)

    transform = build_transform()

    new_images = []
    new_annotations = []
    new_img_id = 0
    new_ann_id = 0

    for img_info in coco["images"]:
        fname = img_info["file_name"]
        fname_alt = fname.replace(".rf.", "_rf_").replace(".jpg.", "_jpg_")

        img_path = images_dir / fname
        if not img_path.exists():
            img_path = images_dir / fname_alt

        if not img_path.exists():
            print(f"[SKIP] image not found: {fname}")
            continue

        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"[SKIP] image unreadable: {fname}")
            continue

        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        anns = ann_map.get(img_info["id"], [])

        instance_masks = []
        for ann in anns:
            m = polygons_to_mask(ann["segmentation"], h, w)
            if m.max() == 0:
                continue
            instance_masks.append((ann, m))

        all_masks = [m for _, m in instance_masks]

        # Orijinal görseli kaydet
        cv2.imwrite(
            str(img_out / fname),
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 95]
        )

        new_images.append({
            "id": new_img_id,
            "file_name": fname,
            "width": w,
            "height": h
        })

        # Orijinal annotation'ları yaz
        for ann, m in instance_masks:
            polys = mask_to_polygons(m)
            if not polys:
                continue

            new_annotations.append({
                "id": new_ann_id,
                "image_id": new_img_id,
                "category_id": ann["category_id"],
                "segmentation": polys,
                "bbox": compute_bbox(m),
                "area": int(m.sum()),
                "iscrowd": 0,
            })
            new_ann_id += 1

        new_img_id += 1

        # Eğer orijinalde annotation yoksa augment üretme
        if len(instance_masks) == 0:
            print(f"[WARN] no valid annotation: {fname}")
            continue

        # Augmented versiyonlar
        for aug_idx in range(n_aug):
            result = transform(image=image, masks=all_masks)
            aug_image = result["image"]
            aug_masks = result["masks"]

            ah, aw = aug_image.shape[:2]
            stem = Path(fname).stem
            aug_fname = f"{stem}_aug{aug_idx:03d}.jpg"

            cv2.imwrite(
                str(img_out / aug_fname),
                cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 95]
            )

            new_images.append({
                "id": new_img_id,
                "file_name": aug_fname,
                "width": aw,
                "height": ah
            })

            for (ann, _), aug_m in zip(instance_masks, aug_masks):
                aug_m_bin = (aug_m > 0).astype(np.uint8)

                polys = mask_to_polygons(aug_m_bin)
                if not polys:
                    continue

                new_annotations.append({
                    "id": new_ann_id,
                    "image_id": new_img_id,
                    "category_id": ann["category_id"],
                    "segmentation": polys,
                    "bbox": compute_bbox(aug_m_bin),
                    "area": int(aug_m_bin.sum()),
                    "iscrowd": 0,
                })
                new_ann_id += 1

            new_img_id += 1

        print(f"✓ {fname[:55]}  →  {n_aug} aug")

    new_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "categories": coco["categories"],
        "images": new_images,
        "annotations": new_annotations,
    }

    out_json = output_dir / "_annotations_aug_coco.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(new_coco, f, indent=2)

    print(f"\n{'─' * 60}")
    print(f"Original images     : {len(coco['images'])}")
    print(f"Augmented images    : {len(new_images) - len(coco['images'])}")
    print(f"Total images        : {len(new_images)}")
    print(f"Total annotations   : {len(new_annotations)}")
    print(f"COCO JSON           : {out_json}")
    print(f"Images dir          : {img_out}")

    return new_coco


# ─────────────────────────────────────────────
#  Visualization
# ─────────────────────────────────────────────

def visualize_sample(coco_data, output_dir, n_samples=6):
    vis_dir = Path(output_dir) / "vis_check"
    vis_dir.mkdir(exist_ok=True)

    ann_map = {}
    for ann in coco_data["annotations"]:
        ann_map.setdefault(ann["image_id"], []).append(ann)

    aug_imgs = [img for img in coco_data["images"] if "_aug" in img["file_name"]]
    if len(aug_imgs) == 0:
        print("No augmented image found, visualize skipped.")
        return

    samples = random.sample(aug_imgs, min(n_samples, len(aug_imgs)))

    for img_info in samples:
        img_path = Path(output_dir) / "images" / img_info["file_name"]
        if not img_path.exists():
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        h, w = image.shape[:2]
        overlay = image.copy()

        for ann in ann_map.get(img_info["id"], []):
            m = polygons_to_mask(ann["segmentation"], h, w)

            overlay[m == 1] = (
                overlay[m == 1] * 0.65 + np.array([0, 220, 80]) * 0.35
            ).astype(np.uint8)

            contours, _ = cv2.findContours(
                m.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, (0, 255, 60), 1)

        cv2.imwrite(str(vis_dir / img_info["file_name"]), overlay)

    print(f"Visual sanity check: {vis_dir}  ({len(samples)} sample)")


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--coco_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_aug", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    print(f"albumentations {A.__version__}")
    print(f"n_aug={args.n_aug}\n")

    new_coco = run_augmentation(
        args.images_dir,
        args.coco_json,
        args.output_dir,
        n_aug=args.n_aug,
        seed=args.seed,
    )

    if args.visualize:
        visualize_sample(new_coco, args.output_dir)