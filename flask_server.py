from flask import Flask, request, jsonify
from flask_cors import CORS  
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import websocket
import json

app = Flask(__name__)
CORS(app) 

MODEL_NAME = "facebook/opt-350m"

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

AWS_WEBSOCKET_URL = "wss://pdobdpl064.execute-api.us-east-1.amazonaws.com/production/"

def send_message_to_aws(message):
    """Send user message to AWS WebSocket API and get a response."""
    try:
        ws = websocket.WebSocket()
        ws.connect(AWS_WEBSOCKET_URL)
        ws.send(json.dumps({"action": "sendMessage", "message": message}))
        response = ws.recv()  
        ws.close()
        return json.loads(response)  
    except Exception as e:
        return {"error": str(e)}

@app.route('/query', methods=['POST'])
def query_model():
    """Handles chat requests from the frontend and forwards them to AWS WebSocket."""
    data = request.json
    user_input = data.get('question', '')

    if not user_input:
        return jsonify({'error': 'No question provided'}), 400

    response = send_message_to_aws(user_input)
    return jsonify({'response': response.get('message', 'Error in AWS communication')})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
