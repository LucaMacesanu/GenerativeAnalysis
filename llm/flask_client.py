import requests

url = 'http://127.0.0.1:5000/echo'
message = {'message': 'suggest a date idea for valentines day'}

response = requests.post(url, json=message)
# print('Response from server:', response.json()['echo'])
print(response.json()['echo'][0]['generated_text'])
breakpoint()
