#!/usr/bin/env python3
"""
Split images referenced by train/test JSON files into separate output folders.

This utility reads two JSON files with the following schema (array of samples):
[
  {
    "messages": [...],
    "images": ["/abs/or/relative/path/to/image.png"]
  },
  ...
]

It collects all image paths from each JSON and copies/symlinks/moves them into:
  <output_dir>/train/
  <output_dir>/test/

Usage example (parameters; avoid shell commands in agent replies):
  - train-json: path to train JSON file
  - test-json: path to test JSON file
  - output-dir: directory where train/ and test/ will be created
  - mode: one of {copy, symlink, move} (default: copy)
  - preserve-hierarchy: if set, recreate original parent directories under split dir
  - dry-run: if set, print planned operations without performing them
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split images referenced by train/test JSON files into output/train and output/test."
        )
    )
    parser.add_argument("--train-json", required=True, help="Path to train JSON file")
    parser.add_argument("--test-json", required=True, help="Path to test JSON file")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory to write train/ and test/ subfolders",
    )
    parser.add_argument(
        "--mode",
        choices=["copy", "symlink", "move"],
        default="copy",
        help="Operation to place files into destination (default: copy)",
    )
    parser.add_argument(
        "--preserve-hierarchy",
        action="store_true",
        help=(
            "Preserve original directory hierarchy under train/ or test/ instead of flattening"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned operations without changing the filesystem",
    )
    return parser.parse_args()


def load_image_list(json_path: Path) -> List[Path]:
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    with json_path.open("r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {json_path}: {e}") from e

    image_paths: List[Path] = []
    for idx, sample in enumerate(data):
        images_field = sample.get("images")
        if not images_field:
            # Skip entries without images
            continue
        if not isinstance(images_field, list):
            raise TypeError(
                f"Entry {idx} in {json_path} has non-list 'images': {type(images_field)}"
            )
        for image_str in images_field:
            if not isinstance(image_str, str):
                raise TypeError(
                    f"Entry {idx} in {json_path} has non-string image path: {image_str!r}"
                )
            image_paths.append(Path(image_str))
    return image_paths


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def unique_destination_path(dest_dir: Path, source_path: Path, used_names: Set[Path]) -> Path:
    """
    Create a destination path in dest_dir using source_path's name, avoiding collisions.
    If a file with the same basename already exists (or was planned), append a numeric suffix.
    """
    base_name = source_path.name
    candidate = dest_dir / base_name
    if candidate not in used_names and not candidate.exists():
        return candidate

    stem = Path(base_name).stem
    suffix = Path(base_name).suffix
    counter = 1
    while True:
        candidate = dest_dir / f"{stem}__dup{counter}{suffix}"
        if candidate not in used_names and not candidate.exists():
            return candidate
        counter += 1


def destination_for_source(
    source: Path,
    group_dir: Path,
    preserve_hierarchy: bool,
    used_names: Set[Path],
) -> Path:
    if preserve_hierarchy:
        # Recreate full parent hierarchy under the group_dir
        # Use absolute parents as-is if the source is absolute, else relative parents
        if source.is_absolute():
            relative_parts = source.parts[1:]  # drop leading '/'
            dest_path = group_dir.joinpath(*relative_parts)
        else:
            dest_path = group_dir / source
        ensure_directory(dest_path.parent)
        return dest_path
    else:
        ensure_directory(group_dir)
        return unique_destination_path(group_dir, source, used_names)


def place_file(source: Path, dest: Path, mode: str, dry_run: bool) -> None:
    if dry_run:
        print(f"[DRY] {mode} -> {dest}")
        return
    ensure_directory(dest.parent)
    if mode == "copy":
        shutil.copy2(str(source), str(dest))
    elif mode == "symlink":
        # Replace existing broken link/file if present
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        os.symlink(os.fspath(source), os.fspath(dest))
    elif mode == "move":
        shutil.move(str(source), str(dest))
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def validate_sources_exist(paths: Iterable[Path]) -> List[Path]:
    missing: List[Path] = []
    for p in paths:
        if not p.exists():
            missing.append(p)
    return missing


def process_group(
    image_paths: List[Path],
    group_dir: Path,
    mode: str,
    preserve_hierarchy: bool,
    dry_run: bool,
) -> int:
    used_names: Set[Path] = set()
    processed = 0
    for source in image_paths:
        dest = destination_for_source(source, group_dir, preserve_hierarchy, used_names)
        used_names.add(dest)
        place_file(source, dest, mode, dry_run)
        processed += 1
    return processed


def main() -> int:
    args = parse_args()

    train_json = Path(args.train_json)
    test_json = Path(args.test_json)
    output_dir = Path(args.output_dir)
    mode = args.mode
    preserve_hierarchy = args.preserve_hierarchy
    dry_run = args.dry_run

    try:
        train_images = load_image_list(train_json)
        test_images = load_image_list(test_json)
    except (FileNotFoundError, ValueError, TypeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    # Validate sources exist and report missing
    missing_train = validate_sources_exist(train_images)
    missing_test = validate_sources_exist(test_images)
    if missing_train or missing_test:
        if missing_train:
            print(f"警告: 训练集缺失 {len(missing_train)} 张图片。示例: {missing_train[0]}")
        if missing_test:
            print(f"警告: 测试集缺失 {len(missing_test)} 张图片。示例: {missing_test[0]}")
        # Proceed with existing ones only
        train_images = [p for p in train_images if p not in missing_train]
        test_images = [p for p in test_images if p not in missing_test]

    train_out = output_dir / "train"
    test_out = output_dir / "test"

    print(
        "开始分拣: "
        f"train {len(train_images)} 张, test {len(test_images)} 张 -> 输出目录 {output_dir}"
    )

    try:
        processed_train = process_group(
            train_images, train_out, mode, preserve_hierarchy, dry_run
        )
        processed_test = process_group(
            test_images, test_out, mode, preserve_hierarchy, dry_run
        )
    except Exception as e:
        print(f"处理失败: {e}", file=sys.stderr)
        return 3

    print(
        f"完成: train 处理 {processed_train} 张, test 处理 {processed_test} 张. 模式={mode}, "
        f"preserve_hierarchy={preserve_hierarchy}, dry_run={dry_run}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())


