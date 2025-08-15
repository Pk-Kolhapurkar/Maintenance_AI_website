import litserve as ls
from transformers import AutoModel, AutoTokenizer
import torch
import warnings

class Llama3LitAPI(ls.LitAPI):
    def setup(self, device):
        model_name = "Prathamesh1420/Llama-3.2-3B-Instruct-bnb-4bit-finetuned"
        
        # Suppress warnings about quantization
        warnings.filterwarnings("ignore", message=".*load_in_4bit.*")
        warnings.filterwarnings("ignore", message=".*8-bit optimizer.*")
        
        try:
            # Try loading with trust_remote_code first
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(
                model_name,
                device_map="auto",
                trust_remote_code=True
            )
        except Exception as e:
            # Fallback to CPU-only loading
            print(f"Failed to load with device_map, falling back to CPU: {str(e)}")
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True
            ).to('cpu')
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Ensure model is on CPU if no GPU
        if not torch.cuda.is_available():
            self.model = self.model.to('cpu')
        
        self.default_gen_config = {
            "max_new_tokens": 64,  # Very conservative for CPU
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        }

    def decode_request(self, request):
        return request["messages"], request.get("generation_config", {})

    def predict(self, inputs):
        messages, gen_config = inputs
        generation_config = {**self.default_gen_config, **gen_config}
        
        try:
            # Format chat template
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_config)
            
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def encode_response(self, output):
        return {"response": output}

if __name__ == "__main__":
    print("Initializing server...")
    api = Llama3LitAPI()
    
    # Force CPU mode to avoid any GPU-related issues
    print("Running in CPU-only mode")
    server = ls.LitServer(api, accelerator="cpu", devices=1)
    server.run(port=8000)