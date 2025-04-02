from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl
apply_liger_kernel_to_qwen2_vl(cross_entropy=True, fused_linear_cross_entropy=False)
from open_r1.trainer.alfred_grpo_trainer import Qwen2VLAlfredGRPOTrainer
from trl import GRPOConfig, get_peft_config, get_quantization_config, ModelConfig
from datasets import load_from_disk
import tqdm
from copy import deepcopy
import difflib
import torch
from PIL import Image
from alfworld.agents.environment.alfred_thor_env import AlfredThorEnv
import alfworld.agents.modules.generic as generic

def get_default_model_config():
    r = 64
    return ModelConfig(
        load_in_4bit=True,
        lora_r=r,
        lora_target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=r,
        use_peft=True,
        use_bnb_nested_quant=True,
    )

def get_default_grpo_config(run_name: str, num_gpus: int = 1) -> GRPOConfig:
    return GRPOConfig(
        output_dir=f"outputs/{run_name}",
        run_name=run_name,
        learning_rate=5e-6,
        warmup_steps=50,
        num_train_epochs=1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        bf16=False,
        fp16=True,
        adam_beta1=0.9,
        adam_beta2=0.99,
        max_grad_norm=0.1,
        beta=0.04,
        max_prompt_length=1024,
        max_completion_length=128,
        per_device_train_batch_size=1,
        num_generations=1,
        gradient_accumulation_steps=int(64 / num_gpus),
        gradient_checkpointing=True,
        save_strategy="epoch",
        save_only_model=True,
        use_vllm=True,
        vllm_device=f"cuda:{num_gpus-1}",
        vllm_gpu_memory_utilization=0.1,
        logging_steps=1,
        log_on_each_node=False,
        report_to="wandb",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        model_init_kwargs={
            'load_in_4bit': True,
            'bnb_4bit_compute_dtype': torch.float16,
        },
    )

def reward_format(completions, **kwargs):
    rewards = []
    for completion in completions:
        reward = 0.0
        if isinstance(completion, dict):
            messages_to_verify = [completion,]
        elif isinstance(completion, list):
            messages_to_verify = [c for c in completion if c['role'] == 'assistant']
        elif isinstance(completion, str):
            messages_to_verify = [{'role': 'assistant', 'content': completion}]
        for msg in messages_to_verify:
            individual_messages = [msg['content'],]
            for im in individual_messages:
                for tag in ['think', 'action']:
                    if im.count(f'<{tag}>') == 1 and im.count(f'</{tag}>') == 1:
                        reward += 0.1
        rewards.append(reward)
    return rewards

def reward_correctness(completions, **kwargs):
    rewards = []
    for completion, response in zip(completions, kwargs['responses']):
        reward = 0.0
        if isinstance(completion, dict):
            messages_to_verify = [completion,]
        elif isinstance(completion, list):
            messages_to_verify = [c for c in completion if c['role'] == 'assistant']
        elif isinstance(completion, str):
            messages_to_verify = [{'role': 'assistant', 'content': completion}]
        for msg in messages_to_verify:
            individual_messages = [msg['content'],]
            for im in individual_messages:
                predicted_action = im.split('<action>')[-1].split('</action>')[0].strip()
                if predicted_action == response:
                    reward += 1.0
        rewards.append(reward)
    return rewards

def reward_available_action(completions, **kwargs):
    rewards = []
    for completion, response, choices in zip(completions, kwargs['responses'], kwargs['choices']):
        reward = 0.0
        if isinstance(completion, dict):
            messages_to_verify = [completion,]
        elif isinstance(completion, list):
            messages_to_verify = [c for c in completion if c['role'] == 'assistant']
        elif isinstance(completion, str):
            messages_to_verify = [{'role': 'assistant', 'content': completion}]
        for msg in messages_to_verify:
            individual_messages = [msg['content'],]
            for im in individual_messages:
                predicted_action = im.split('<action>')[-1].split('</action>')[0].strip()
                if predicted_action in choices:
                    reward += 0.1
        rewards.append(reward)
    return rewards

def reward_difflib_score(completions, **kwargs):
    rewards = []
    for completion, response, choices in zip(completions, kwargs['responses'], kwargs['choices']):
        reward = 0.0
        if isinstance(completion, dict):
            messages_to_verify = [completion,]
        elif isinstance(completion, list):
            messages_to_verify = [c for c in completion if c['role'] == 'assistant']
        elif isinstance(completion, str):
            messages_to_verify = [{'role': 'assistant', 'content': completion}]
        for msg in messages_to_verify:
            individual_messages = [msg['content'],]
            for im in individual_messages:
                predicted_action = im.split('<action>')[-1].split('</action>')[0].strip()
                closest_action = difflib.get_close_matches(word=predicted_action, possibilities=choices, n=1, cutoff=0.0)[0]
                sm = difflib.SequenceMatcher(None, predicted_action, closest_action, autojunk=False)
                reward += 0.1 * sm.ratio()
        rewards.append(reward)
    return rewards

def preprocess_dataset(example, **kwargs):
    '''texts = []
    for conversation in example['conversations']:
        text = f"SYSTEM: {conversation[0]['content'][0]['text']}\n\nUSER: {conversation[1]['content'][0]['text']}"
        texts.append(text)'''
    return {"prompt": example['conversations']}

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    '''conversations = []
    for convo in convos:
        conversations.append(tokenizer.apply_chat_template(
            convo, tokenize = False, add_generation_prompt = True
        ))'''
    return { "prompt": convos}


if __name__ == '__main__':
    #model_name = "./model_sft_vl_qwen2_5_vl"
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    max_seq_length = 2048 # Can increase for longer reasoning traces
    lora_rank = 64 # Larger rank = smarter, but slower
    
    config = generic.load_config()
    env_type = config['env']['type']
    env = AlfredThorEnv(config, train_eval='train')
    env: AlfredThorEnv = env.init_env(batch_size=1)
    dataset = [{'prompt': 'dummy'},]*4000*50
    #indices = list(range(15000, 30000))
    #dataset = dataset.select(indices=indices)
    #dataset = standardize_sharegpt(dataset)
    #dataset = dataset.map(formatting_prompts_func, batched=True)

    trainer = Qwen2VLAlfredGRPOTrainer(
        model=model_name,
        args=get_default_grpo_config(run_name='alphahome-rl-vl-3B', num_gpus=1),
        train_env=env,
        peft_config=get_peft_config(model_args=get_default_model_config()),
        min_pixels=(300*300),
        max_pixels=(300*300)
    )

    trainer.train()
    trainer.save_model("model_rl_vl")