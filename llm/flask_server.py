from flask import Flask, request, jsonify
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

app = Flask(__name__)


# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,            # Enable 4-bit loading
    bnb_4bit_use_double_quant=True,   # Use double quantization for better performance
    bnb_4bit_quant_type='nf4',        # NormalFloat4 quantization type (best for most tasks)
    bnb_4bit_compute_dtype='float16'  # Use float16 for faster computations
)

# Load model with 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf", 
    quantization_config=bnb_config,
    device_map="auto"  # Automatically select the appropriate device (GPU)
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

@app.route('/echo', methods=['POST'])
def echo():
    data = request.get_json()
    message = data.get('message', '')
    print("User input: ", message)
    
    result = pipe(message, max_length=200, temperature=0.7)
    

    # message = "woot woot"
    return jsonify({'echo': result})

if __name__ == '__main__':
    app.run(debug=False)
