import json
import cv2
import numpy as np
from pathlib import Path

DATASET_ROOT = Path(r"C:\Users\keylo\Desktop\LAP\dataset\attention_unet_dataset")

def check_split(split_name):
    split_dir = DATASET_ROOT / split_name
    coco_path = split_dir / "_annotations.coco.json"
    masks_dir = split_dir / "masks"

    with open(coco_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    ann_map = {}
    for ann in coco["annotations"]:
        ann_map.setdefault(ann["image_id"], []).append(ann)

    print(f"\n=== {split_name.upper()} ===")

    no_annotation = []
    black_masks = []

    for img_info in coco["images"]:
        image_id = img_info["id"]
        file_name = img_info["file_name"]
        stem = Path(file_name).stem
        mask_path = masks_dir / f"{stem}.png"

        anns = ann_map.get(image_id, [])
        if len(anns) == 0:
            no_annotation.append(file_name)

        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None or np.max(mask) == 0:
                black_masks.append(file_name)

    print(f"Toplam image           : {len(coco['images'])}")
    print(f"Annotation olmayan     : {len(no_annotation)}")
    print(f"Tam siyah mask         : {len(black_masks)}")

    if no_annotation:
        print("\n[NO ANNOTATION EXAMPLES]")
        for x in no_annotation[:10]:
            print(" -", x)

    if black_masks:
        print("\n[BLACK MASK EXAMPLES]")
        for x in black_masks[:10]:
            print(" -", x)

for split in ["train", "valid", "test"]:
    check_split(split)