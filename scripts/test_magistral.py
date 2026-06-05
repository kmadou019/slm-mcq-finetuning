from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "mistralai/Magistral-Small-2509"

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# prompt
messages = [
    {"role": "user", "content": "Explain gradient descent simply."}
]

# chat template
inputs = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt",
    add_generation_prompt=True
).to(model.device)

# generation
outputs = model.generate(
    inputs,
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.95
)

# decode
response = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print(response)