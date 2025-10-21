import argparse
import json
import os
import sys
from typing import Any, Dict

import yaml
import time


def _load_vllm_infer_callable():
    """Dynamically load vllm_infer from scripts/vllm_infer.py.

    We avoid package import issues if scripts/ is not a Python package.
    """
    module_dir = os.path.dirname(os.path.abspath(__file__))
    module_path = os.path.join(module_dir, "vllm_infer.py")

    if not os.path.exists(module_path):
        raise FileNotFoundError(f"Cannot find vllm_infer.py at: {module_path}")

    import importlib.util

    spec = importlib.util.spec_from_file_location("_vllm_infer_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("Failed to create import spec for vllm_infer.py")

    module = importlib.util.module_from_spec(spec)
    sys.modules["_vllm_infer_module"] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "vllm_infer"):
        raise AttributeError("vllm_infer.py does not define `vllm_infer` callable")

    return module.vllm_infer


def _build_vllm_config_from_yaml(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Merge user vllm_config with convenient top-level fields from YAML.

    - infer_dtype -> dtype
    - vllm_maxlen -> max_model_len
    - vllm_enforce_eager -> enforce_eager
    - vllm_gpu_util -> gpu_memory_utilization
    - trust_remote_code -> trust_remote_code
    """
    vllm_cfg: Dict[str, Any] = {}

    yaml_vllm_cfg = cfg.get("vllm_config") or {}
    if isinstance(yaml_vllm_cfg, dict):
        vllm_cfg.update(yaml_vllm_cfg)

    if "infer_dtype" in cfg and cfg["infer_dtype"]:
        vllm_cfg["dtype"] = cfg["infer_dtype"]

    if "vllm_maxlen" in cfg and cfg["vllm_maxlen"]:
        vllm_cfg["max_model_len"] = cfg["vllm_maxlen"]

    if "vllm_enforce_eager" in cfg:
        vllm_cfg["enforce_eager"] = cfg["vllm_enforce_eager"]

    if "vllm_gpu_util" in cfg and cfg["vllm_gpu_util"]:
        vllm_cfg["gpu_memory_utilization"] = cfg["vllm_gpu_util"]

    if "trust_remote_code" in cfg:
        vllm_cfg["trust_remote_code"] = cfg["trust_remote_code"]

    return vllm_cfg


def _compute_jsonl_accuracy(jsonl_path: str) -> tuple[int, int, float]:
    matches = 0
    total = 0
    if not os.path.exists(jsonl_path):
        return 0, 0, 0.0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            pred = str(obj.get("predict", "")).strip()
            label = str(obj.get("label", "")).strip()
            total += 1
            if pred == label:
                matches += 1
    accuracy = (matches / total) if total else 0.0
    return matches, total, accuracy


def run_from_yaml(config_path: str, save_name: str | None = None) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Required/basic fields
    model_name_or_path = cfg["model_name_or_path"]
    template = cfg["template"]
    dataset = cfg["dataset"]

    # Optional generation args
    cutoff_len = int(cfg.get("cutoff_len", 2048))
    temperature = float(cfg.get("temperature", 0.95))
    top_p = float(cfg.get("top_p", 0.7))
    top_k = int(cfg.get("top_k", 50))
    max_new_tokens = int(cfg.get("max_new_tokens", 1024))
    repetition_penalty = float(cfg.get("repetition_penalty", 1.0))

    # Multimodal knobs
    image_max_pixels = int(cfg.get("image_max_pixels", 768 * 768))
    video_fps = float(cfg.get("video_fps", 2.0))
    video_maxlen = int(cfg.get("video_maxlen", 128))

    # vLLM engine config
    vllm_config = _build_vllm_config_from_yaml(cfg)

    # Output file
    if not save_name:
        save_name = cfg.get("save_name", "generated_predictions.jsonl")

    vllm_infer = _load_vllm_infer_callable()

    # Retry on insufficient free memory by lowering gpu_memory_utilization.
    # Start from provided value (or 0.9 if not set) and step down to 0.5.
    start_util = float(vllm_config.get("gpu_memory_utilization", 0.9))
    tried_utils = []
    total_start_time = time.time()
    for step in range(5):
        current_util = round(max(0.5, start_util - 0.1 * step), 2)
        if tried_utils and current_util == tried_utils[-1]:
            continue
        tried_utils.append(current_util)
        vllm_config["gpu_memory_utilization"] = current_util

        try:
            # Call underlying generator. We pass dict for vllm_config; downstream supports dict/JSON.
            vllm_infer(
                model_name_or_path=model_name_or_path,
                adapter_name_or_path=None,
                dataset=dataset,
                dataset_dir=cfg.get("dataset_dir", "data"),
                template=template,
                cutoff_len=cutoff_len,
                max_samples=cfg.get("max_samples"),
                vllm_config=vllm_config,  # type: ignore[arg-type]
                save_name=save_name,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                skip_special_tokens=bool(cfg.get("skip_special_tokens", True)),
                default_system=cfg.get("default_system"),
                enable_thinking=bool(cfg.get("enable_thinking", True)),
                seed=cfg.get("seed"),
                pipeline_parallel_size=int(cfg.get("pipeline_parallel_size", 1)),
                image_max_pixels=image_max_pixels,
                image_min_pixels=int(cfg.get("image_min_pixels", 32 * 32)),
                video_fps=video_fps,
                video_maxlen=video_maxlen,
                batch_size=int(cfg.get("batch_size", 1024)),
            )
            # Success if no exception raised; compute and print accuracy from the output jsonl
            matches, total, accuracy = _compute_jsonl_accuracy(save_name)
            elapsed_seconds = time.time() - total_start_time
            print(json.dumps({
                "output": save_name,
                "total": total,
                "matches": matches,
                "accuracy": accuracy,
                "elapsed_seconds": elapsed_seconds
            }, ensure_ascii=False))
            return
        except Exception as e:
            msg = str(e)
            # If error not related to memory utilization, raise immediately
            if "Free memory on device" not in msg and "GPU memory utilization" not in msg:
                raise
            # Otherwise, continue to next lower utilization
            if current_util <= 0.5:
                raise

    # If loop exits without return, raise a generic error
    raise RuntimeError(
        f"vLLM failed to start after trying gpu_memory_utilization values: {tried_utils}. "
        "Consider lowering tensor_parallel_size or max sequence length."
    )


def main():
    parser = argparse.ArgumentParser(description="Run vLLM batch generation from a YAML config")
    default_cfg = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "examples",
        "eval",
        "qwen2_5vl_lora_sft_multi_train_vllm.yaml",
    )
    parser.add_argument("--config", type=str, default=default_cfg, help="Path to YAML config file")
    parser.add_argument("--save-name", type=str, default=None, help="Override output jsonl path")
    args = parser.parse_args()

    run_from_yaml(os.path.abspath(args.config), args.save_name)


if __name__ == "__main__":
    main()


