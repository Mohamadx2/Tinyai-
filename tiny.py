from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

checkpoint = "HuggingFaceTB/SmolLM2-135M-Instruct"

# Load tokenizer and model (forced CPU, save memory)
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    torch_dtype=torch.float32,  # CPU recommended to use float32
    device_map="cpu"
)
print("Model loaded! Type 'exit' to quit.\n")

messages = [
    {"role": "system", "content": "You are a helpful, concise assistant."}
]

while True:
    user_input = input("You: ").strip()
    if not user_input:
        continue
    if user_input.lower() in ["exit", "quit"]:
        break

    messages.append({"role": "user", "content": user_input})

    # Apply chat template
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer.encode(input_text, return_tensors="pt")

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=128,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    print(f"AI: {response}")
    messages.append({"role": "assistant", "content": response})