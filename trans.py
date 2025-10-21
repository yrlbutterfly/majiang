from typing import List, Optional, Union
import numpy as np

# 可选：如果你的 preds/labels 是 torch.Tensor
try:
    import torch
    TORCH = True
except Exception:
    TORCH = False

# 1) 准备 tokenizer（与训练一致）
from transformers import AutoTokenizer
# 用你当前模型路径（与 YAML 一致）
MODEL_PATH = "/mnt/data/personal/ceyao/models/Qwen/Qwen2.5-VL-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

IGNORE_INDEX = -100  # 项目里常用的忽略值


def to_numpy(x):
    if TORCH and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def decode_pred_to_text(
    pred_ids: Union[List[int], np.ndarray],
    label_ids: Optional[Union[List[int], np.ndarray]] = None,
    ignore_index: int = IGNORE_INDEX,
    only_valid_targets: bool = False,
    skip_special_tokens: bool = True,
) -> str:
    """
    - pred_ids: 一维 token id 序列（即你拿到的 pred）
    - label_ids: 可选，若传入则可进行右移+掩码，仅解码有效目标位（与 accuracy 一致）
    - only_valid_targets: True 时启用右移+掩码筛选有效位
    """
    pred = to_numpy(pred_ids).reshape(-1)  # (seq_len,)
    if label_ids is not None and only_valid_targets:
        labels = to_numpy(label_ids).reshape(-1)  # (seq_len,)
        # 与 accuracy 对齐：pred[:-1] 对 label[1:]
        pred_step = pred[:-1]
        label_step = labels[1:]
        mask = (label_step != ignore_index)
        tokens = pred_step[mask].tolist()
    else:
        tokens = pred.tolist()

    text = tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)
    return text


# 2) 把你的 pred 粘贴到这里（示例：用占位变量 pred）
pred = [151645, 151645, 151645,     25, 151645, 151645,  17847, 151645,
       151645,    198,  18724,    198, 151645,     18,   1805,     21,
          220,     18, 100153,     17,    220, 151645,    481,     16,
        99741,    220, 151645,    220,    198,     17,    220, 151645,
          389,     17,    220,     17,     19,     16,     17,     17,
          220,     11, 111805,    220,    220,     11,    220, 151645,
        99583, 151645,  15235,     17,     18,     16,    220,    198,
           18,     20,     17,     18,     19,     16,     17,     16,
           11,  44991,    220,     18,     17,     11,  36556,    220,
           16,     16,  98460,  98460,     17,    220,     18,    220,
       151645, 151645,    350, 151645,     17,     18,    198, 151645,
           17,     17,    220,     18,     16,    220,    220,    220,
       151645,    220,     17,     18,    220,    220, 102321,    220,
          220,     11,  10496,    220,    220, 107246,     17,     15,
           16,     11,   7002,   2326,     11,    389,     18,    220,
          220,     17,    220,     17,    220,   1293,     18,    304,
           18,     18,    220,     16,  64355,     18,     18,     16,
           16,     18,     17,  99696,    389, 102885,    220,    220,
          220,     18,     15,    220,     17,    220,    220,    220,
           18,     17,     16,     16,    350, 151645,     18,     16,
           11,   2163,     11,     18,     17,     11,     16,    220,
         8622, 100679,     18,    220,     17,    220,    220,    220,
       151645, 151645,     16,     37,    350,    198,     17,    198,
           11,    220,     18,     18,     16,     16,     17,    220,
           17,    220,  98460,     70,  43216,    330,  13639,     70,
          220,     19,     16,     37,     20,    350,    220,  10023,
           16,     11,    220,     16,     16,   1809,     17,    220,
          304,     18,  20493,  20493,     16,    389,    220,     16,
          220,     17,     19,     17,    350,   1378,     17,     16,
          220,    220,     18,     17,     16,   6476,     17,    220,
           16,     16, 100400,   6303,    220,     19,   7002,   9956,
           17, 151645,    198,     22,     17,     21,     21,     16,
           18,     17, 102011,    625,     18,  13876,  18137,     16,
         3040,    220,    220,    323,  20493, 151645,     17,   6249,
         6476,     16,     18,   4217,     16,     11,     22,     18,
           17,    220,     11,     17,     16,    220,     18,    304,
           17,     19,     17,     17,     16,   5543,     18,    220,
          220,     18,     18,   3705,    220,    220,     20,     11,
           17,     15,     18,     16,     17,   1965,     19,     17,
        10769,     18,     18,   3040,  22674,    220,    557,   4479,
           19,     19,     18,    220,    220,     18,     18,     17,
           17,    220,     18,     17,     17,     11,     17,     16,
       151645,     18,    431,   4038,     19,     19,     19,     11,
           19,  35491,     20,     20,     11, 151645,  20493, 151645,
       151645, 151645, 151645, 151645, 151645, 151645,    198,  18724,
       151645,    198,     16, 151645,    198,  18724,  18724]  # 将你 ipdb 打印的数字列表粘贴进来

# 3) 直接整段解码（不依赖 labels）
print(decode_pred_to_text(pred, only_valid_targets=False))

# 4) 若你也有 labels，并希望与 accuracy 完全一致（右移 + 掩码）
# text = decode_pred_to_text(pred, label_ids=labels, only_valid_targets=True)
# print(text)