import json
import cv2
import numpy as np
from pathlib import Path

try:
    from pycocotools import mask as maskUtils
except ImportError:
    maskUtils = None


DATASET_ROOT = Path(r"C:\Users\keylo\Desktop\LAP\dataset\attention_unet_dataset")


def polygons_to_mask(segmentations, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)

    # RLE format
    if isinstance(segmentations, dict) and "counts" in segmentations:
        if maskUtils is None:
            raise ImportError(
                "pycocotools gerekli. Kur: pip install pycocotools "
                "veya Windows için pip install pycocotools-windows"
            )

        rle = segmentations

        # COCO compressed RLE decode
        decoded = maskUtils.decode(rle)

        # decode sonucu bazen (H,W,1) gelir
        if decoded.ndim == 3:
            decoded = decoded[:, :, 0]

        mask = (decoded > 0).astype(np.uint8) * 255
        return mask

    # Polygon format
    for seg in segmentations:
        if isinstance(seg, dict):
            continue
        if len(seg) < 6:
            continue
        pts = np.array(seg, dtype=np.float32).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)

    return mask


def convert_split(split_name: str):
    split_dir = DATASET_ROOT / split_name
    coco_json = split_dir / "_annotations.coco.json"
    masks_dir = split_dir / "masks"
    masks_dir.mkdir(exist_ok=True)

    with open(coco_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    ann_map = {}
    for ann in coco["annotations"]:
        ann_map.setdefault(ann["image_id"], []).append(ann)

    for img_info in coco["images"]:
        image_id = img_info["id"]
        file_name = img_info["file_name"]
        h = img_info["height"]
        w = img_info["width"]

        mask = np.zeros((h, w), dtype=np.uint8)

        anns = ann_map.get(image_id, [])
        for ann in anns:
            m = polygons_to_mask(ann["segmentation"], h, w)
            mask = np.maximum(mask, m)

        stem = Path(file_name).stem
        out_path = masks_dir / f"{stem}.png"
        cv2.imwrite(str(out_path), mask)

    print(f"[OK] {split_name} masks created: {masks_dir}")


if __name__ == "__main__":
    for split in ["train", "valid", "test"]:
        convert_split(split)