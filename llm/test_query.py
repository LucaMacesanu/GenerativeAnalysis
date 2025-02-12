import requests

query = {"question": "What is the capital of France?"}
response = requests.post("http://127.0.0.1:5000/query", json=query)
print(response.json())
