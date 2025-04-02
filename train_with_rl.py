# script.py
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
import verifiers as vf
from verifiers.prompts.system_prompts import SIMPLE_PROMPT
from trl import GRPOTrainer

model_name = "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-unsloth-bnb-4bit"
max_seq_length = 8192 # Can increase for longer reasoning traces
lora_rank = 64 # Larger rank = smarter, but slower
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.5, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)
model = FastLanguageModel.for_training(model=model, use_gradient_checkpointing=True)
vf_env = vf.envs.R1MathEnv(dataset="gsm8k")
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    reward_funcs=vf_env.get_rubric(),
    args=vf.get_default_grpo_config(run_name="deepseek-r1-distill-math-rl", num_gpus=1),
    train_dataset=vf_env.get_dataset(),
)
trainer.train()
# vf_env.eval(batch_size=32) (coming soon)