# import transformers
# import torch

# model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# pipeline = transformers.pipeline(
#     "text-generation",
#     model="meta-llama/Meta-Llama-3.1-8B-Instruct",
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device="cuda",
# )

# breakpoint()

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
mpath = "C:\\Users\\lucam\\.llama\\checkpoints\\Llama3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(mpath)
model = AutoModelForCausalLM.from_pretrained(mpath, torch_dtype=torch.float16, device_map='auto')

# Function to generate a response
def generate_response(prompt, max_length=200):
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
    outputs = model.generate(**inputs, max_length=max_length, do_sample=True, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Test the model
if __name__ == "__main__":
    prompt = "Explain the significance of the moon landing in 1969."
    response = generate_response(prompt)
    print(f"Prompt: {prompt}\nResponse: {response}")