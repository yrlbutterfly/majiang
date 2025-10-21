# Q&A

为了在训练过程中每隔eval_steps可以进行eval，需要修改src/llamafactory/train/sft/workflow.py

### 1.<img width="1986" height="398" alt="image" src="https://github.com/user-attachments/assets/49295fb5-d5d4-435d-a6b9-6754a3ac70d8" />
pre:

pad_to_multiple_of=8 if training_args.do_train else None

now:

pad_to_multiple_of=8 if (training_args.do_train and not training_args.predict_with_generate) else None

### 2.将这段设置left-padding的代码移到do_train前面
<img width="1176" height="196" alt="image" src="https://github.com/user-attachments/assets/75babb62-1435-4a98-b760-a3edd57a13a8" />
