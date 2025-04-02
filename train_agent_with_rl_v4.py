from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl
apply_liger_kernel_to_qwen2_vl(cross_entropy=True, fused_linear_cross_entropy=False)
from open_r1.trainer.grpo_trainer import Qwen2VLGRPOTrainer
from trl import GRPOConfig, get_peft_config, get_quantization_config, ModelConfig
from datasets import load_from_disk
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLProcessor
import tqdm
from copy import deepcopy
import difflib
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
import numpy as np
import base64
from io import BytesIO


def image_to_base_64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str.decode('utf-8')

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
        max_prompt_length=2048,
        max_completion_length=64,
        per_device_train_batch_size=1,
        num_generations=2,
        gradient_accumulation_steps=int(32 / num_gpus),
        gradient_checkpointing=True,
        save_strategy="epoch",
        save_only_model=True,
        use_vllm=False,
        vllm_device=f"cuda:{num_gpus-1}",
        vllm_gpu_memory_utilization=0.1,
        logging_steps=1,
        log_on_each_node=False,
        report_to="wandb",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        #model_init_kwargs={
        #    'load_in_4bit': True,
        #    'bnb_4bit_compute_dtype': torch.float16,
        #},
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
                reward += 0.1 * difflib.SequenceMatcher(None, predicted_action, closest_action).ratio()
        rewards.append(reward)
    return rewards

def formatting_prompts_func(example):
    responses = example['responses']
    conversations = example['conversations']
    images = example['image']
    new_conversations = []
    for conversation, image, response in zip(conversations, images, responses):
        #image.save('/tmp/img.jpg')
        #image = Image.open('/tmp/img.jpg')
        new_conversation = deepcopy(conversation)
        for index in range(len(new_conversation)-1):
            if 'image' in new_conversation[index]['content'][-1]:
                new_conversation[index]['content'][-1].pop('image')
        new_conversation[-1]['content'][-1]['image'] = image
        #print(new_conversation)
        #exit()
        new_conversation.append({
            'role': 'assistant',
            'content': [{'type': 'text', 'text': f"<action>{response}</action>"}]
        })
        new_conversations.append(new_conversation)
    return {'messages': new_conversations, 'images': images}

def preprocess_dataset(example, **kwargs):
    return {"prompt": example['conversations']}

def filter_dataset(example, **kwargs):
    keep = []
    for conversation in example['conversations']:
        keep.append(len(conversation) < 15)
    return keep

if __name__ == '__main__':
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    max_seq_length = 2048 # Can increase for longer reasoning traces
    lora_rank = 64 # Larger rank = smarter, but slower
    
    dataset = load_from_disk(dataset_path='./datasets/alphahome-rl-vl')
    dataset = dataset.filter(filter_dataset, batched=True)
    dataset = dataset.map(preprocess_dataset, batched=True)
    #indices = list(range(15000, 30000))
    #dataset = dataset.select(indices=indices)
    #dataset = standardize_sharegpt(dataset)
    #dataset = dataset.map(formatting_prompts_func, batched=True)
    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model and tokenizer
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        device_map="auto",
        # attn_implementation="flash_attention_2", # not supported for training
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    processor = AutoProcessor.from_pretrained(model_name)
    trainer = Qwen2VLGRPOTrainer(
        model=model,
        reward_funcs=[reward_format, reward_correctness, reward_available_action, reward_difflib_score],
        args=get_default_grpo_config(run_name='alphahome-rl-vl-3B', num_gpus=1),
        train_dataset=dataset,
        peft_config=get_peft_config(model_args=get_default_model_config()),
    )

    trainer.train()
    trainer.save_model("model_rl_vl")