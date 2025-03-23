from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_dir = r"D:\models\uer\gpt2-chinese-cluecorpussmall\models--uer--gpt2-chinese-cluecorpussmall\snapshots\c2c0249d8a2731f269414cc3b22dff021f8e07a3"

model = AutoModelForCausalLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# 使用加载的模型和分词器创建生成文本的 pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device="cuda")

# output = generator("你好，我是一款语言模型，", max_length=50, num_return_sequences=1)
output = generator("你好，我是一款语言模型，", max_length=50,
                   num_return_sequences=1, # 指定返回多少个独立生成的文本序列
                   truncation=True, # 该参数决定是否截断输入文本以适应模型的最大输入长度
                   temperature=0.7, # 该参数控制生成文本的随机性。值越低，生成的文本约保守（偏向于选择概率较高的词）；值越高，生成的文本越多样（倾向于选择更多不同的词）
                   top_k=50, # 该参数限制模型在每一步生成时仅从概率最高的 K 个词中选择下一个词
                   top_p=0.9, # 该参数又称为核采样，进一步限制模型生成时的词汇选择范围。它会选择一组累积概率达到 p 的词汇，模型只会从这个概率集合中采样
                   clean_up_tokenization_spaces=False) # 该参数控制生成的文本是否清理分词时引入的空格

print(output)