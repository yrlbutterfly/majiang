#python scripts/make_video_from_jsonl.py --input-dir ./results/qwen2_5vl-7b/full/sft/majiang_turn_1016_annotated_origin --output ./results/qwen2_5vl-7b/full/sft/majiang_turn_1016_annotated_origin/predictions_video.mp4
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2


def parse_image_index(filename: str) -> Optional[int]:
    """
    Extract the leading integer index from an image filename.
    Expected formats include: "123_3.png", "3_1.png", "21.png".
    Returns None if no leading integer is found.
    """
    base = Path(filename).name
    # 新逻辑：允许前缀+下划线的形式，例如 a_123.png / b_45_0.png
    # 1) 若以字母+下划线开头，如 a_ 或 b_，则尝试在其后的片段中解析首个整数作为 index
    m = re.match(r"^[A-Za-z]+_(.+)$", base)
    if m:
        remain = m.group(1)
        # remain 可能为 123.png 或 123_0.png 等，从开头提取整数
        idx = _regex_leading_int(remain)
        if idx is not None:
            return idx
    # 2) 兼容原逻辑：若文件名以数字开头或采用 <index>_*.png，则解析首段数字
    if "_" in base:
        head = base.split("_", 1)[0]
        return int(head) if head.isdigit() else _regex_leading_int(base)
    return _regex_leading_int(base)


def _regex_leading_int(text: str) -> Optional[int]:
    match = re.match(r"^(\d+)", text)
    if match:
        return int(match.group(1))
    return None


def _extract_primary_secondary_indices(filename: str) -> Tuple[Optional[int], int]:
    """
    从文件名中提取主序号与次序号：
    - 主序号：首个出现的整数
    - 次序号：第二个出现的整数（若没有则为 0）
    同时支持字母前缀+下划线（如 a_1506_1.png）。
    """
    base = Path(filename).name
    m = re.match(r"^[A-Za-z]+_(.+)$", base)
    remain = m.group(1) if m else base
    nums = re.findall(r"\d+", remain)
    if not nums:
        return None, 0
    primary = int(nums[0])
    secondary = int(nums[1]) if len(nums) > 1 else 0
    return primary, secondary


def normalize_path(img_path: str, project_root: Path) -> Path:
    p = Path(img_path)
    if str(p).startswith("./"):
        return (project_root / str(p)[2:]).resolve()
    if not p.is_absolute():
        return (project_root / p).resolve()
    return p


def measure_text(text: str, font, font_scale: float, thickness: int) -> Tuple[int, int, int]:
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    return w, h, baseline


def draw_top_left(frame, text: str, padding: int, font, font_scale: float, thickness: int, color: Tuple[int, int, int]):
    w, h, baseline = measure_text(text, font, font_scale, thickness)
    x = padding
    y = padding + h
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, lineType=cv2.LINE_AA)


def draw_top_right_two_segments(
    frame,
    left_text: str,
    right_text: str,
    left_color: Tuple[int, int, int],
    right_color: Tuple[int, int, int],
    padding: int,
    gap: int,
    font,
    font_scale: float,
    thickness: int,
):
    h_frame, w_frame = frame.shape[:2]
    left_w, left_h, left_baseline = measure_text(left_text, font, font_scale, thickness)
    right_w, right_h, right_baseline = measure_text(right_text, font, font_scale, thickness)

    total_w = left_w + gap + right_w
    x_start = max(padding, w_frame - padding - total_w)
    y = padding + max(left_h, right_h)

    # Draw left segment (e.g., predict)
    cv2.putText(frame, left_text, (x_start, y), font, font_scale, left_color, thickness, lineType=cv2.LINE_AA)
    # Draw right segment (e.g., label) directly after left + gap
    cv2.putText(
        frame,
        right_text,
        (x_start + left_w + gap, y),
        font,
        font_scale,
        right_color,
        thickness,
        lineType=cv2.LINE_AA,
    )


def load_first_valid_frame(items: List[dict]) -> Optional[Tuple]:
    for it in items:
        img = cv2.imread(str(it["abs_path"]))
        if img is not None:
            return img, img.shape[1], img.shape[0]
    return None


def draw_top_center(frame, text: str, padding: int, font, font_scale: float, thickness: int, color: Tuple[int, int, int]):
    h_frame, w_frame = frame.shape[:2]
    w, h, baseline = measure_text(text, font, font_scale, thickness)
    x = max(padding, (w_frame - w) // 2)
    y = padding + h
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, lineType=cv2.LINE_AA)


def draw_bottom_right(
    frame,
    text: str,
    padding: int,
    font,
    font_scale: float,
    thickness: int,
    color: Tuple[int, int, int],
):
    h_frame, w_frame = frame.shape[:2]
    tw, th, baseline = measure_text(text, font, font_scale, thickness)
    x = max(padding, w_frame - padding - tw)
    y = max(th + padding, h_frame - padding)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, lineType=cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Make a video from JSONL predictions with overlays.")
    parser.add_argument(
        "--jsonl",
        type=str,
        required=False,
        default=str(
            Path(__file__).resolve().parents[1]
            / "results/qwen2_5vl-7b/lora/sft/majiang_turn_1009_large_all/generated_predictions.jsonl"
        ),
        help="Path to the generated_predictions.jsonl",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=False,
        help=(
            "包含 predict_majiang_turn_train_multi / predict_majiang_turn_test_multi 两个子目录的目录，"
            "每个子目录下应有 generated_predictions.jsonl。若提供该参数，将优先于 --jsonl。"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        default=str(Path(__file__).resolve().parents[1] / "results/predictions_video.mp4"),
        help="Output video file path (mp4)。当存在多个前缀分组时，会在该路径的文件名基础上追加前缀，例如 predictions_video__a.mp4",
    )
    parser.add_argument("--fps", type=int, default=5, help="Frames per second for the output video")
    parser.add_argument("--padding", type=int, default=12, help="Padding (pixels) from edges for text overlays")
    parser.add_argument("--font-scale", type=float, default=0.8, help="OpenCV font scale for text")
    parser.add_argument("--thickness", type=int, default=2, help="OpenCV text thickness")
    parser.add_argument(
        "--mismatch-repeats",
        type=int,
        default=10,
        help="当 predict 与 label 不一致时，重复写入该帧的次数（>=1）",
    )
    parser.add_argument(
        "--mismatch-list",
        type=str,
        required=False,
        default=str(
            Path(__file__).resolve().parents[1]
            / "results/mismatched_filenames.txt"
        ),
        help="将所有 predict≠label 的图片文件名写入到该文件（每行一个）",
    )
    parser.add_argument(
        "--mismatch-save-dir",
        type=str,
        required=False,
        default=str(
            Path(__file__).resolve().parents[1]
            / "results/mismatched_frames"
        ),
        help="将所有 predict≠label 的已标注图片保存到该目录（按分组区分）",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    items: List[dict] = []

    def load_jsonl_file(jsonl_fp: Path, split_name: Optional[str]):
        if not jsonl_fp.exists():
            return
        with open(jsonl_fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                images = data.get("images") or []
                if not images:
                    continue

                predict_raw = data.get("predict", "")
                label_raw = data.get("label", "")
                predict = str(predict_raw).strip()
                label = str(label_raw).strip()

                # 遍历该样本的所有图片，逐张输出
                for img_path in images:
                    abs_path = normalize_path(img_path, project_root)
                    basename = Path(img_path).name
                    # 提取前缀：匹配 <letters>_ 作为 video_id（例如 a_、b_）。若不存在，则置为 "default"
                    m_prefix = re.match(r"^([A-Za-z]+)_", basename)
                    video_id = m_prefix.group(1) if m_prefix else "default"
                    idx, sub_idx = _extract_primary_secondary_indices(basename)
                    if idx is None:
                        continue
                    items.append(
                        {
                            "video_id": video_id,
                            "idx": idx,
                            "sub_idx": sub_idx,
                            "abs_path": abs_path,
                            "filename": basename,
                            "predict": predict,
                            "label": label,
                            "split": (split_name or ""),
                        }
                    )

    # 优先读取 --input-dir（包含 train/test 两个子目录，支持多种命名）
    if args.input_dir:
        base_dir = Path(args.input_dir)
        # 遍历 base_dir 下的所有子目录：只要目录名包含 'train' 或 'test' 即认为是对应 split
        found_any = False
        subdirs = [p for p in base_dir.iterdir() if p.is_dir()]
        # 先处理 train，再处理 test，保证输出顺序稳定
        for split_name, keyword in (("train", "train"), ("test", "test")):
            for d in subdirs:
                name_lower = d.name.lower()
                if keyword in name_lower:
                    jsonl_path = d / "generated_predictions.jsonl"
                    if jsonl_path.exists():
                        found_any = True
                        load_jsonl_file(jsonl_path, split_name)
        if not found_any:
            raise FileNotFoundError(
                f"在 {base_dir} 下未找到任何名称包含 'train' 或 'test' 的子目录中的 generated_predictions.jsonl，请检查目录结构"
            )
    else:
        # 兼容旧逻辑：从单一 --jsonl 读取（此时不标注 split）
        jsonl_path = Path(args.jsonl)
        if not jsonl_path.exists():
            raise FileNotFoundError(f"JSONL 文件不存在: {jsonl_path}")
        load_jsonl_file(jsonl_path, split_name=None)

    # 分组/排序：
    # - 当使用 --input-dir 时，按全局 idx 升序排序并输出为单一视频（避免不同来源交错）
    # - 否则（单一 --jsonl），保持原有分组逻辑
    if not items:
        raise RuntimeError("未从 JSONL 中解析到任何图像条目")

    if args.input_dir:
        items_sorted = sorted(
            items,
            key=lambda x: (x["idx"], x.get("sub_idx", 0), x["filename"]),
        )
        groups = {"__global__": items_sorted}
    else:
        groups = {}
        for it in items:
            groups.setdefault(it["video_id"], []).append(it)
        for gid in groups:
            groups[gid].sort(key=lambda x: (x["idx"], x.get("sub_idx", 0), x["filename"]))

    font = cv2.FONT_HERSHEY_SIMPLEX
    padding = int(args.padding)
    gap = 10
    font_scale = float(args.font_scale)
    thickness = int(args.thickness)

    color_white = (255, 255, 255)
    color_green = (0, 200, 0)
    color_red = (0, 0, 255)

    # 输出命名：若只有一个分组，则直接使用 --output；若多个分组，则在文件名后追加 __<group>
    base_out_path = Path(args.output)
    base_out_path.parent.mkdir(parents=True, exist_ok=True)

    total_written = 0
    for gid, gitems in groups.items():
        # 计算每组的视频输出路径
        if len(groups) == 1:
            out_path = base_out_path
        else:
            out_path = base_out_path.with_name(
                f"{base_out_path.stem}__{gid}{base_out_path.suffix}"
            )

        # 确定帧尺寸（按组）
        first = load_first_valid_frame(gitems)
        if first is None:
            # 跳过无效组
            continue
        _, frame_w, frame_h = first

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (frame_w, frame_h))

        written = 0
        mismatched_names_group: List[str] = []

        # 准备不匹配图片保存目录（按分组）
        group_save_dir = None
        if args.mismatch_save_dir:
            base_dir = Path(args.mismatch_save_dir)
            if len(groups) == 1:
                group_save_dir = base_dir
            else:
                group_save_dir = base_dir.with_name(f"{base_dir.name}__{gid}")
            group_save_dir.mkdir(parents=True, exist_ok=True)
        for it in gitems:
            img = cv2.imread(str(it["abs_path"]))
            if img is None:
                continue
            if img.shape[1] != frame_w or img.shape[0] != frame_h:
                img = cv2.resize(img, (frame_w, frame_h), interpolation=cv2.INTER_AREA)

            filename_text = it["filename"]

            draw_top_left(
                img,
                filename_text,
                padding=padding,
                font=font,
                font_scale=font_scale,
                thickness=thickness,
                color=(0, 0, 0),
            )

            is_correct = it["predict"] == it["label"]
            predict_text = f"predict: {it['predict']}"
            label_text = f"label: {it['label']}"

            draw_top_right_two_segments(
                img,
                left_text=predict_text,
                right_text=label_text,
                left_color=(color_green if is_correct else color_red),
                right_color=color_white,
                padding=padding,
                gap=gap,
                font=font,
                font_scale=font_scale,
                thickness=thickness,
            )

            # 右下角绘制来源（train/test），若存在
            split_text = str(it.get("split") or "").strip()
            if split_text:
                draw_bottom_right(
                    img,
                    text=split_text,
                    padding=padding,
                    font=font,
                    font_scale=font_scale,
                    thickness=thickness,
                    color=(0, 0, 0),
                )

            if not is_correct:
                mismatched_names_group.append(it["filename"])
                draw_top_center(
                    img,
                    text="wrong!",
                    padding=padding,
                    font=font,
                    font_scale=font_scale * 2,
                    thickness=max(2, thickness),
                    color=color_red,
                )

                # 保存不匹配的已标注帧到分组目录
                if group_save_dir is not None:
                    save_path = group_save_dir / it["filename"]
                    cv2.imwrite(str(save_path), img)

            repeats = 1 if is_correct else max(1, int(args.mismatch_repeats))
            for _ in range(repeats):
                writer.write(img)
                written += 1

        writer.release()
        total_written += written

        # 针对每个分组写出不匹配文件名列表
        if args.mismatch_list:
            base_list_path = Path(args.mismatch_list)
            if len(groups) == 1:
                out_list_path = base_list_path
            else:
                out_list_path = base_list_path.with_name(
                    f"{base_list_path.stem}__{gid}{base_list_path.suffix}"
                )
            out_list_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_list_path, "w", encoding="utf-8") as f:
                for name in mismatched_names_group:
                    f.write(name + "\n")

    if total_written == 0:
        raise RuntimeError("没有任何帧被写入视频，请检查图片路径解析/分组")


if __name__ == "__main__":
    main()


