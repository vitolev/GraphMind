import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained("OuteAI/Lite-Oute-1-300M").to(device)
tokenizer = AutoTokenizer.from_pretrained("OuteAI/Lite-Oute-1-300M")

def generate_response(message: str, temperature: float = 0.4, repetition_penalty: float = 1.12) -> str:
    # Convert message to PyTorch tensors
    input_ids = tokenizer.encode(
        message, return_tensors="pt"
    ).to(device)
    # Generate the response
    output = model.generate(
        input_ids,
        max_length=256,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        do_sample=True
    ) 
    # Decode the generated output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text
message = "Write Python code to calculate the GCD of 10293 and 29384 and print the result"
response = generate_response(message)
print(response)