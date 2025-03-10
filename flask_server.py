from flask import Flask, request, jsonify
from flask_cors import CORS  
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

app = Flask(__name__)
CORS(app) 

MODEL_NAME = "facebook/opt-350m"

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

@app.route('/query', methods=['POST'])
def query_model():
    data = request.json
    user_input = data.get('question', '')
    print("User input:", user_input)
    
    result = pipe(user_input, max_length=100, temperature=0.7, do_sample=True, truncation=True)
    
    return jsonify({'response': result[0]['generated_text']})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
