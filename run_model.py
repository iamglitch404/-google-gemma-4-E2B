import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def main():
    model_id = "google/gemma-4-E2B"
    prompt = os.environ.get("USER_PROMPT", "Explain what GitHub Actions is.")
    
    print(f"Loading model {model_id}...")
    try:
        # Check if HF_TOKEN is provided. Gemma models typically require an access token.
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            print("Warning: HF_TOKEN environment variable is not set. If the model is gated, loading will fail.")

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # We load the model in 8-bit or standard depending on memory, but GitHub actions has limited memory.
        # So we use standard loading and hope it fits, or you can add quantization.
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32, # CPU inference works best with float32
            low_cpu_mem_usage=True
        )
        
        print(f"Model loaded. Generating response for prompt: '{prompt}'")
        
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Generate output
        outputs = model.generate(**inputs, max_new_tokens=100)
        
        print("\n--- Output ---")
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        print("--------------\n")
        
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
