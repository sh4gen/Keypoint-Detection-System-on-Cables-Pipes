import json
from pathlib import Path

DATASET_ROOT = Path(r"C:\Users\keylo\Desktop\LAP\dataset\attention_unet_dataset")

targets = {
    "train": [
        "IMG_20260319_003358_jpg.rf.21bb87b424ebe574601a6609f0b60654.jpg",
        "IMG_20260319_002239_1_jpg.rf.4ed9602a182b8978db47de202570c135.jpg",
    ],
    "valid": [
        "IMG_20260319_001652_jpg.rf.d5a799004ee8ed370b514038a3b801f0.jpg",
    ]
}

for split, filenames in targets.items():
    coco_path = DATASET_ROOT / split / "_annotations.coco.json"
    with open(coco_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    image_map = {img["file_name"]: img for img in coco["images"]}
    ann_map = {}
    for ann in coco["annotations"]:
        ann_map.setdefault(ann["image_id"], []).append(ann)

    print(f"\n=== {split.upper()} ===")
    for fn in filenames:
        img_info = image_map.get(fn)
        if img_info is None:
            print(f"{fn} -> IMAGE NOT FOUND IN JSON")
            continue

        anns = ann_map.get(img_info["id"], [])
        print(f"{fn}")
        print(f"  image_id: {img_info['id']}")
        print(f"  annotations: {len(anns)}")
        if len(anns) > 0:
            print(f"  first ann keys: {anns[0].keys()}")
            print(f"  first ann segmentation type: {type(anns[0].get('segmentation'))}")
            print(f"  first ann segmentation preview: {str(anns[0].get('segmentation'))[:200]}")