from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import standardize_sharegpt
from trl import GRPOTrainer, GRPOConfig
from datasets import load_from_disk
import tqdm
from copy import deepcopy
from collections import deque
import difflib


def get_default_grpo_config(run_name: str, num_gpus: int = 1) -> GRPOConfig:
    return GRPOConfig(
        output_dir=f"outputs/{run_name}",
        run_name=run_name,
        learning_rate=5e-6,
        warmup_steps=50,
        num_train_epochs=1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        bf16=True,
        adam_beta1=0.9,
        adam_beta2=0.99,
        max_grad_norm=0.1,
        beta=0.04,
        max_prompt_length=1024,
        max_completion_length=2048,
        per_device_train_batch_size=1,
        num_generations=2,
        gradient_accumulation_steps=int(4 / num_gpus),
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
                closest_action = difflib.get_close_matches(word=predicted_action, possibilities=choices, n=1, cutoff=0.0)
                reward += 0.1 * difflib.SequenceMatcher(None, predicted_action, closest_action).ratio()
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
    return { "prompt": convos, }

if __name__ == '__main__':
    model_name = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
    max_seq_length = 8192 # Can increase for longer reasoning traces
    lora_rank = 64 # Larger rank = smarter, but slower
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        load_in_4bit = True, # False for LoRA 16bit
        fast_inference = True, # Enable vLLM fast inference
        max_lora_rank = lora_rank,
        gpu_memory_utilization = 0.4, # Reduce if out of memory
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "qwen-2.5",
    )

    dataset = load_from_disk(dataset_path="./datasets/alphahome-rl")
    #indices = list(range(15000, 30000))
    #dataset = dataset.select(indices=indices)
    #dataset = standardize_sharegpt(dataset)
    dataset = dataset.map(formatting_prompts_func, batched=True)

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

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_format, reward_correctness, reward_available_action, reward_difflib_score],
        args=get_default_grpo_config(run_name='alphahome-rl-1.5B', num_gpus=1),
        train_dataset=dataset
    )

    trainer.train()
    model.save_pretrained_merged("model_rl", tokenizer, save_method = "merged_16bit",)