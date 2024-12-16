from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = 'meta-llama/Llama-3.2-11B-Vision'
quant_path = model_path.split('/')[1].lower() + '-bnb'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_path, low_cpu_mem_usage=True, load_in_4bit=True
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Save quantized model
model.save_pretrained(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')