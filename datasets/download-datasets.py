import os
from datasets import load_dataset

# 设置代理（通过环境变量，兼容旧版本 datasets）
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
os.environ["HF_PROXY"] = "http://127.0.0.1:7890"  # Hugging Face 专用代理

# 定义缓存路径（使用正斜杠避免转义问题）
cache_dir = r"D:/datasets/NousResearch/hermes-function-calling-v1"

# 创建缓存目录
os.makedirs(cache_dir, exist_ok=True)

try:
    # 下载数据集（移除 proxies 参数，改用环境变量）
    dataset = load_dataset(
        "NousResearch/hermes-function-calling-v1",
        cache_dir=cache_dir,
        # verify=False  # 如果需要禁用 SSL 验证，取消注释此行
    )
    print(f"数据集已下载到: {cache_dir}")
except Exception as e:
    print(f"下载失败: {str(e)}")
    print("请检查代理是否运行、路径格式是否正确，或尝试更新 datasets 库版本")