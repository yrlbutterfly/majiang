import json

file_path = "/home/ceyao/Projects/yrl/LLaMA-Factory/generated_predictions.jsonl"

matches = 0
total = 0

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        pred = str(obj.get("predict", "")).strip()
        label = str(obj.get("label", "")).strip()
        total += 1
        if pred == label:
            matches += 1

accuracy = matches / total if total else 0.0
print({"total": total, "matches": matches, "accuracy": accuracy})