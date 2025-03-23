from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

model_dir = r"D:\models\google-bert\bert-base-chinese\models--google-bert--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f"
model = BertForSequenceClassification.from_pretrained(model_dir)
tokenizer = BertTokenizer.from_pretrained(model_dir)

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device="cuda")

result = classifier("你好，我是一款语言模型")
print(result)