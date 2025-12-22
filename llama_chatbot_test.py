# source gemma_env/bin/activate
#
# pip install -U keras-hub
# pip install -U torch
# pip install -U sentencepiece
# pip install -U huggingface-hub
#
# huggingface-cli login
#
# python3 llama_chatbot_test.py

import os

# 1. FORCE THE GPU BACKEND
os.environ["KERAS_BACKEND"] = "torch"

import keras_hub
import torch

# 1. Check if CUDA (GPU support) is available
gpu_check = torch.cuda.is_available()
print(f"Is CUDA available? {gpu_check}")

# 2. Get details of the active GPU
if gpu_check:
    current_id = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(current_id)
    print(f"Using GPU Device: {gpu_name} (ID: {current_id})")
else:
    print("WARNING: Running on CPU!")

# 2. LOAD THE MODEL
# Using 3B-Instruct as it fits well on most modern GPUs (8GB+ VRAM)
model_id = "hf://meta-llama/Llama-3.2-3B-Instruct"

print(f"Loading {model_id} on GPU...")
model = keras_hub.models.Llama3CausalLM.from_preset(
    model_id,
    dtype="bfloat16" # Essential for speed and memory on NVIDIA 30/40 series
)

# 1. Start with the "Begin Text" token and a System Prompt
full_conversation = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>"

# 3. INITIALIZE HISTORY
# Llama 3.2 uses specific header tags: <|begin_of_text|>, <|start_header_id|>, etc.
# KerasHub handles these formatting details via the preprocessor.
chat_history = []

print("\n--- LLAMA CHATBOT ACTIVE (Type 'exit' to stop) ---")

while True:
    # Get user input
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    # 2. Append the User's turn with specific Llama 3 headers
    full_conversation += f"<|start_header_id|>user<|end_header_id|>\n\n{user_input}<|eot_id|>"
    
    # 3. Add the Assistant header so the model knows it's its turn to speak
    full_conversation += "<|start_header_id|>assistant<|end_header_id|>\n\n"

    # 4. Generate - passing a STRING avoids the 'object' error
    response = model.generate(
        full_conversation, 
        max_length=2048, 
        strip_prompt=True
    )
    
    # 5. Clean and print the response
    print(f"Llama: {response}")
    
    # 6. Append the model's answer and the End of Turn token to the history
    full_conversation += f"{response}<|eot_id|>"
