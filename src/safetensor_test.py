from utils import (
    load_json,
    load_safetensors,
)

import torch

if __name__ == "__main__":
    params_files = [
        "/Users/richard/MyLlama/model/Llama-3.2-1B-Instruct/model.safetensors"
    ]
    
    params = load_safetensors(params_files)
    
    print(params)
    
    