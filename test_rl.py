from unsloth import FastLanguageModel
from vllm import SamplingParams
from datasets import load_dataset
from latex2sympy2 import latex2sympy
import tqdm

#eval_dataset = load_dataset("HuggingFaceH4/MATH-500", split='test')
eval_dataset = load_dataset("gsm8k", 'main', split='test')


model_name = "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-unsloth-bnb-4bit"
#model_name = "./outputs/deepseek-r1-distill-math-rl/checkpoint-467"
max_seq_length = 20000 # Can increase for longer reasoning traces
lora_rank = 64 # Larger rank = smarter, but slower
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = False, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.5, # Reduce if out of memory
)
'''model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)'''

model = FastLanguageModel.for_inference(model)

total_success = 0
total_problems = 0

for sample in tqdm.tqdm(eval_dataset):
    text = tokenizer.apply_chat_template([
        {"role" : "user", "content" : sample['question']},
        {"role" : "assistant", "content" : "<think>"}
    ], tokenize = False, add_generation_prompt = False)

    sampling_params = SamplingParams(
        #temperature = 0.8,
        temperature = 0.0,
        top_p = 0.95,
        max_tokens = 15000,
    )
    output = model.fast_generate(
        [text],
        sampling_params = sampling_params,
        lora_request = None,
    )[0].outputs[0].text
    try:
        answer = latex2sympy(f"{output.split('</think>')[-1].split('boxed{')[-1].split('}')[0]}")
        if answer.equals(latex2sympy(f"{sample['answer'].split('####')[-1].strip()}")):
            total_success += 1
    except:
        pass
    total_problems += 1
    print('Success rate so far: ', total_success/total_problems)
    if total_problems >= 10:
        break
print('Success rate: ', total_success/total_problems)