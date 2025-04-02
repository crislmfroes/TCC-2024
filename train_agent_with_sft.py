import os
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments, DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported, FastLanguageModel
from datasets import load_from_disk
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import standardize_sharegpt


max_seq_length = 2048
dataset_path = "./datasets/alphahome-sft"
model_name = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"

dataset = load_from_disk(dataset_path=dataset_path)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = False,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 128, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [#"q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "qwen-2.5",
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    '''texts = [
        tokenizer.apply_chat_template(
            convo, tokenize = False, add_generation_prompt = False
        )
        for convo in convos
    ]'''
    #return { "text" : texts, }
    conversations = []
    for convo in convos:
        conversations.append(tokenizer.apply_chat_template(
            convo, tokenize = False, add_generation_prompt = False
        ))
    return { "text": conversations, }


dataset = standardize_sharegpt(dataset)

print(dataset[0])

dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
)

print(dataset[0])
#exit()

trainer = SFTTrainer(
    model = model,
    processing_class = tokenizer,
    train_dataset = dataset,
    #max_seq_length = max_seq_length,
    #data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    #dataset_num_proc = 2,
    #packing = False, # Can make training 5x faster for short sequences.
    args = SFTConfig(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        warmup_steps = 5,
        num_train_epochs = 1, # Set this for 1 full training run.
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs/alphahome-sft-1.5B",
        report_to = "wandb", # Use this for WandB etc
        dataset_kwargs={"skip_prepare_dataset": False},
        max_seq_length=max_seq_length,
        remove_unused_columns=True,
        #label_names=["text"],
        dataset_text_field="text",
        packing=False,
        dataset_batch_size=8,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    ),
)
trainer.train()
model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit",)