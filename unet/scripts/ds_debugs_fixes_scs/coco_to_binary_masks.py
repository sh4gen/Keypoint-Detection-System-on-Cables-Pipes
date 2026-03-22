import json
import cv2
import numpy as np
from pathlib import Path


DATASET_ROOT = Path(r"C:\Users\keylo\Desktop\LAP\dataset\attention_unet_dataset")


def polygons_to_mask(segmentations, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)

    # RLE support
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

    # Polygon support
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
    images_dir = split_dir / "images"
    coco_json = split_dir / "_annotations.coco.json"
    masks_dir = split_dir / "masks"

    masks_dir.mkdir(exist_ok=True)

    with open(coco_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # image_id -> annotations
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
            mask = np.maximum(mask, m)  # tüm cable instance'larını birleştir

        stem = Path(file_name).stem
        out_path = masks_dir / f"{stem}.png"
        cv2.imwrite(str(out_path), mask)

    print(f"[OK] {split_name} masks created: {masks_dir}")


if __name__ == "__main__":
    for split in ["train", "valid", "test"]:
        convert_split(split)