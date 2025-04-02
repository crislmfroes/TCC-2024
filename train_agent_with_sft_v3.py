from liger_kernel.transformers import apply_liger_kernel_to_qwen2_vl
apply_liger_kernel_to_qwen2_vl(cross_entropy=True, fused_linear_cross_entropy=False)
from trl import SFTTrainer, SFTConfig, get_peft_config, get_quantization_config, ModelConfig
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
    r = 32
    return ModelConfig(
        load_in_4bit=True,
        lora_r=r,
        lora_target_modules=[
            #"q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=r,
        use_peft=True,
        use_bnb_nested_quant=True,
    )

def get_default_sft_config(run_name: str, num_gpus: int = 1) -> SFTConfig:
    return SFTConfig(
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
        per_device_train_batch_size=1,
        gradient_accumulation_steps=int(64 / num_gpus),
        gradient_checkpointing=True,
        save_strategy="epoch",
        save_only_model=True,
        logging_steps=1,
        log_on_each_node=False,
        report_to="wandb",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        model_init_kwargs={
            'load_in_4bit': True,
            'bnb_4bit_compute_dtype': torch.float16,
        },
        dataset_kwargs={
            'skip_prepare_dataset': True
        },
        dataset_text_field="",
        remove_unused_columns=False
    )

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

def collate_fn(examples):
    # Get the texts and images, and apply the chat template
    #exit()
    texts = []
    image_inputs = []
    for example in examples:
        messages = example['conversations']
        for index in range(len(messages)):
            for j in range(len(messages[index]['content'])):
                if 'image' in messages[index]['content'][j].keys() and messages[index]['content'][j]['type'] != 'image':
                    messages[index]['content'][j].pop('image')
                if 'text' in messages[index]['content'][j].keys() and messages[index]['content'][j]['type'] != 'text':
                    messages[index]['content'][j].pop('text')
        image = example['image']
        messages[-1]['content'][1]['image'] = image
        response = example['responses']
        messages.append({
            'role': 'assistant',
            'content': [{'type': 'text', 'text': f"<action>{response}</action>"}]
        })
        texts += [processor.apply_chat_template(messages, tokenize=False)]
        #print(texts)
        image_inputs += [process_vision_info(messages)[0]]

    # Tokenize the texts and process the images
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100  #
    # Ignore the image token index in the loss computation (model specific)
    if isinstance(processor, Qwen2_5_VLProcessor):
        image_tokens = [151652,151653,151655]
    else: 
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100
    batch["labels"] = labels

    return batch


if __name__ == '__main__':
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    max_seq_length = 2048 # Can increase for longer reasoning traces
    lora_rank = 64 # Larger rank = smarter, but slower
    
    dataset = load_from_disk(dataset_path='./datasets/alphahome-rl-vl')
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
    trainer = SFTTrainer(
        model=model,
        data_collator=collate_fn,
        #processing_class=Qwen2_5_VLProcessor,
        args=get_default_sft_config(run_name='alphahome-sft-vl-3B', num_gpus=1),
        train_dataset=dataset,
        peft_config=get_peft_config(model_args=get_default_model_config()),
    )

    trainer.train()
    trainer.save_model("model_sft_vl")