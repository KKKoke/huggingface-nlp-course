import requests

# Use token to access online models
API_URL = "https://api-inference.hugqingface.co/models/uer/gpt2-chinese-clvecorpussmaLl"
API_TOKEN = "XXXXXXXXX"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

response = requests.post(url=API_URL, headers=headers, json={"inputs":"你好，Hugging face"})
print(response.json())