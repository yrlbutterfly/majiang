#FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/qwen2_5vl_full_sft_multi.yaml
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/qwen2_5vl_full_sft_fetch.yaml
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/qwen2_5vl_full_sft_toukan.yaml
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/qwen2_5vl_full_sft_annotated.yaml
#FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/qwen2_5vl_full_sft_qa.yaml