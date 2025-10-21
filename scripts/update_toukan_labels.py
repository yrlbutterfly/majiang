#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional


TRAIN_JSON_ABS_PATH = \
    "/home/ceyao/Projects/yrl/LLaMA-Factory/data/toukan_train.json"
TEST_JSON_ABS_PATH = \
    "/home/ceyao/Projects/yrl/LLaMA-Factory/data/toukan_test.json"


def infer_label_from_image_path(image_path: str) -> Optional[str]:
    """Infer standardized label from image path.

    Rules:
    - If image is in a "1-x" folder → "right"
    - If image is in a "2-x" folder → "opposite"
    - If image is in a "3-x" folder → "left"

    We detect the segment after "toukan_train"/"toukan_test":
    ./data/toukan_train/1/1-0/1.png → second folder is "1-0" → right
    ./data/toukan_test/1/2-1/0.png  → second folder is "2-1" → opposite

    Entries under "0" (e.g., ./data/toukan_test/0/18.png) are left unchanged.
    """
    p = PurePosixPath(image_path)
    parts = p.parts

    def find_root_index() -> Optional[int]:
        for marker in ("toukan_train", "toukan_test"):
            if marker in parts:
                return parts.index(marker)
        return None

    idx = find_root_index()
    if idx is None:
        return None

    # Expected structure (after the marker):
    # idx+1 → first folder (e.g., "1" or "0")
    # idx+2 → second folder (e.g., "1-0", "2-1", "3-2"). Might be missing for "0" cases.
    if len(parts) <= idx + 2:
        return None

    second_folder = parts[idx + 2]
    if second_folder.startswith("1-"):
        return "right"
    if second_folder.startswith("2-"):
        return "opposite"
    if second_folder.startswith("3-"):
        return "left"
    return None


def normalize_file(json_abs_path: str) -> int:
    """Normalize assistant content values in the given JSON file.

    Returns the number of records updated.
    """
    with open(json_abs_path, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    updated_count = 0
    for record in data:
        images: List[str] = record.get("images", []) or []
        if not images:
            continue

        label = infer_label_from_image_path(images[0])
        if not label:
            continue

        messages: List[Dict[str, Any]] = record.get("messages", []) or []
        changed = False
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("content") != label:
                msg["content"] = label
                changed = True

        if changed:
            updated_count += 1

    # Write back with stable, readable formatting
    with open(json_abs_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    return updated_count


def main() -> None:
    train_updates = normalize_file(TRAIN_JSON_ABS_PATH)
    test_updates = normalize_file(TEST_JSON_ABS_PATH)
    print(
        f"Updated {train_updates} records in toukan_train.json, "
        f"{test_updates} records in toukan_test.json."
    )


if __name__ == "__main__":
    main()


