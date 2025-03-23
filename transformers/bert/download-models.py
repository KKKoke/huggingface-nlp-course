import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# 设置代理（与 curl 命令中的代理一致）
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

# 如果使用 huggingface_hub 库，还需额外配置
os.environ["HF_PROXY"] = "http://127.0.0.1:7890"

model_name = "google-bert/bert-base-chinese"
cache_dir = r"D:\models\google-bert\bert-base-chinese"  # 使用原始字符串避免转义问题

# 确保缓存目录存在
os.makedirs(cache_dir, exist_ok=True)

# 下载模型和分词器（显式指定代理）
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        proxies={"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        proxies={"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}
    )
    print(f"模型和分词器已下载到 {cache_dir}")
except Exception as e:
    print(f"下载失败: {str(e)}")
    print("请检查代理配置或网络连接")