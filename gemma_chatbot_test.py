# python3 -m venv gemma_env
# source gemma_env/bin/activate
#
# pip install "transformers[torch]" accelerate bitsandbytes
#
# python3 -m pip install transformers accelerate torch
#
# python3 -c "import transformers; print('Transformers version:', transformers.__version__)"
# (Transformers version: 4.57.3)
#
# huggingface-cli login
#
# python3 gemma_chatbot_test.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "google/gemma-3-12b-it"

# Define the 4-bit configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 1. Load Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load in 4-bit to save VRAM (requires 'pip install bitsandbytes')
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",          # Automatically picks your NVIDIA GPU
    torch_dtype=torch.bfloat16 # Best precision for Gemma
)

def chat():
    messages = []
    print("--- PyTorch Gemma 3 Chat (Ubuntu + NVIDIA) ---")
    
    while True:
        # How many states are there in the United States?
        # Print all prefectures of Japan
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]: break

        messages.append({"role": "user", "content": user_input})
        
        # Apply the Gemma 3 template
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Move inputs to the GPU
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        # Generate
        outputs = model.generate(**inputs, max_new_tokens=500)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's new response
        answer = response.split("model\n")[-1]
        print(f"\nGemma: {answer}")
        
        messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    chat()