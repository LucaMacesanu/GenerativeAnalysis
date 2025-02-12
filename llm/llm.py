from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Local path where the model is stored
model_path = "C:\\Users\\lucam\\.llama\\checkpoints\\Llama3.1-8B-Instruct"  # Replace with your actual path

# Load the tokenizer and model from the local directory
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",            # Automatically maps to GPU if available
    load_in_8bit=True             # Use 8-bit quantization to save VRAM
)


@app.route('/query', methods=['POST'])
def query_model():
    data = request.json
    user_input = data.get('question', '')
    
    inputs = tokenizer(user_input, return_tensors='pt').to('cuda')
    outputs = model.generate(**inputs, max_new_tokens=150)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return jsonify({'response': response})
ll
if __name__ == '__main__':
    print("App running")
    app.run(port=5000)
