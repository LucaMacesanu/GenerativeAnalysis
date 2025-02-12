# # Use a pipeline as a high-level helper
# from transformers import pipeline

# pipe = pipeline("text-generation", model="meta-llama/Llama-2-7b-hf")

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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

# Test prompt
prompt = "Explain the significance of the moon landing in 1969."
result = pipe(prompt, max_length=200, temperature=0.7)
print(result[0]['generated_text'])



breakpoint()