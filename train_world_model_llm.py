from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, KTOConfig, KTOTrainer
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from peft.auto import AutoPeftModelForCausalLM
from transformers import AutoProcessor, AutoModelForVision2Seq, Qwen2VLProcessor, Idefics3Processor
from PIL import Image
from qwen_vl_utils import process_vision_info
import torch
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator

import json

model = "actor"
model_id = "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit"
resized_height = 300
resized_width = 300

# loadcsv dataset
dataset = load_dataset("json", data_files=f"./{model}_dataset/train.jsonl", split="train")
#dataset = load_dataset('THUDM/AgentInstruct', split='alfworld')
tokenizer = AutoTokenizer.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

#dataset = dataset.filter(lambda e: e['label'] == True)

def convert_to_messages(example):
    '''messages = []
    for j, message in enumerate(example['messages']):
        content = message['content'].copy()
        for i in range(len(content)):
            if content[i]['type'] == 'image':
                img = Image.open(content[i]['image'])
                img.load()
                message['content'][i]['image'] = img
                message['content'][i].pop('text')
            if content[i]['type'] == 'text':
                message['content'][i].pop('image')
        if j != len(example['messages'])-2:
            message['content'] = [e for e in message['content'] if e['type'] != 'image']
        messages.append(message)
    return messages'''
    #img = Image.open(example['image'])
    #img.load()
    return [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'text',
                        'text': example['prompt'].replace('<image>', '')
                    },
                    {
                        'type': 'image',
                        'image': example['image'],
                        #'resized_height': resized_height,
                        #'resized_width': resized_width
                    }
                ]
            },
            {
                'role': 'assistant',
                'content': example['completion']
            }
        ]

new_dataset = []
for example in dataset:
    new_dataset.append({
        'messages': convert_to_messages(example)
    })
dataset = new_dataset
#print(dataset[0])
#exit()

def formatting_prompts_func(example):
    output_texts = []
    output_images = []
    for i in range(len(example['prompt'])):
        messages = [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'text',
                        'text': example['prompt'][i].replace('<image>', '')
                    },
                    {
                        'type': 'image',
                        'image': example['image'][i]
                    }
                ]
            },
            {
                'role': 'assistant',
                'content': example['completion'][i]
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False)
        output_texts.append(text)
        image = process_vision_info(messages)[0]
        output_images.append(image)
    return processor(text=output_texts, images=output_images, return_tensors='pt', padding=True).cpu()

# Create a data collator to encode text and image pairs
def collate_fn(examples):
    # Get the texts and images, and apply the chat template
    texts = [
        processor.apply_chat_template(example, tokenize=False, add_generation_prompt=True) for example in examples
    ]  # Prepare texts for processing
    image_inputs = [[Image.open(example[0]['content'][1]['image'])] for example in examples]#[process_vision_info(example)[0] for example in examples]  # Process the images to extract inputs
    # Tokenize the texts and process the images
    batch = processor(
        text=texts, images=image_inputs, return_tensors="pt", padding=True
    )  # Encode texts and images into tensors

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()  # Clone input IDs for labels
    labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels

    # Ignore the image token index in the loss computation (model specific)
    if isinstance(processor, Qwen2VLProcessor):  # Check if the processor is Qwen2VLProcessor
        image_tokens = [151652, 151653, 151655]  # Specific image token IDs for Qwen2VLProcessor
    elif isinstance(processor, Idefics3Processor):
        image_tokens = [49153,]
    else:
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]  # Convert image token to ID

    # Mask image token IDs in the labels
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100  # Mask image token IDs in labels

    batch["labels"] = labels  # Add labels to the batch
    return batch  # Return the prepared batch

def formatting_prompts_func_agent_instruct(example):
    output_texts = []
    for i in range(len(example['conversations'])):
        conversation = example['conversations'][i]
        processed_conversation = []
        for message in conversation:
            if message['from'] == 'human':
                role = 'user'
            else:
                role = 'assistant'
            processed_conversation.append({
                'role': role,
                'content': message['value']
            })
        text = tokenizer.apply_chat_template(processed_conversation, tokenize=False)
        output_texts.append(text)
    return output_texts

lora_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=8,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)
training_args = SFTConfig(
    output_dir=f"./{model}_checkpoints_v4",  # Directory to save the model
    num_train_epochs=3,  # Number of training epochs
    per_device_train_batch_size=1,  # Batch size for training
    per_device_eval_batch_size=1,  # Batch size for evaluation
    gradient_checkpointing=True,
    gradient_accumulation_steps=16,
    #gradient_accumulation_steps=1,  # Steps to accumulate gradients
    #gradient_checkpointing=False,  # Enable gradient checkpointing for memory efficiency
    # Optimizer and scheduler settings
    #auto_find_batch_size=True,
    optim="adamw_torch_fused",  # Optimizer type
    learning_rate=2e-4,  # Learning rate for training
    lr_scheduler_type="constant",  # Type of learning rate scheduler
    # Logging and evaluation
    logging_steps=10,  # Steps interval for logging
    #eval_steps=10,  # Steps interval for evaluation
    eval_strategy="no",  # Strategy for evaluation
    #save_strategy="steps",  # Strategy for saving the model
    #save_steps=20,  # Steps interval for saving
    save_total_limit=5,
    metric_for_best_model="eval_loss",  # Metric to evaluate the best model
    greater_is_better=False,  # Whether higher metric values are better
    #load_best_model_at_end=True,  # Load the best model after training
    # Mixed precision and gradient settings
    max_grad_norm=0.3,  # Maximum norm for gradient clipping
    warmup_ratio=0.03,  # Ratio of total steps for warmup
    # Hub and reporting
    push_to_hub=False,  # Whether to push model to Hugging Face Hub
    report_to="wandb",  # Reporting tool for tracking metrics
    # Gradient checkpointing settings
    gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
    # Dataset configuration
    dataset_text_field="",  # Text field in dataset
    dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
    max_seq_length=8192  # Maximum sequence length for input
)

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True
)

# Load model and tokenizer
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",
    load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
    #max_seq_length=4096,
)
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True, # False if not finetuning vision layers
    finetune_language_layers   = True, # False if not finetuning language layers
    finetune_attention_modules = True, # False if not finetuning attention layers
    finetune_mlp_modules       = True, # False if not finetuning MLP layers

    r = 16,           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 16,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
)
FastVisionModel.for_training(model)
#peft_model = get_peft_model(model, lora_config)
training_args.remove_unused_columns = False
trainer = SFTTrainer(
    model,
    peft_config=lora_config,
    args=training_args,
    train_dataset=dataset,
    #formatting_func=lambda e: e,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    #processing_class=processor.image_processor_class
    #tokenizer=processor.tokenizer
)

#training_args = KTOConfig(output_dir=f'./{model}_checkpoints', auto_find_batch_size=True, save_total_limit=5, num_train_epochs=10, logging_steps=100)
#trainer = KTOTrainer(
#    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
#    peft_config=lora_config,
#    args=training_args,
#    train_dataset=dataset,
#    processing_class=tokenizer
    #data_collator=formatting_prompts_func
#)
trainer.train()