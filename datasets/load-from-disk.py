from datasets import load_dataset

# 直接加载 Arrow 文件
dataset = load_dataset("arrow", data_files=r"D:\datasets\NousResearch\hermes-function-calling-v1\NousResearch___hermes-function-calling-v1\func_calling_singleturn\0.0.0\8f025148382537ba84cd325e1834b706e1461692\hermes-function-calling-v1-train.arrow")

print(dataset)  # 查看数据集结构

train_data = dataset["train"]
for data in train_data:
    print(data)