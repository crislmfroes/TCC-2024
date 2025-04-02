from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTConfig, SFTTrainer
import pandas as pd
from datasets import Dataset
from PIL import Image

dataset = pd.read_json('./train_cot.jsonl', lines=True)
dataset = dataset.to_dict(orient='records')
new_dataset = []
for example in dataset:
    img = Image.open(example['messages'][1]['content'][0]['image'])
    #img = img.load()
    new_dataset.append({
        'messages': [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'text',
                        'text': f"{example['messages'][0]['content'][0]['text']}\n\n\n{example['messages'][1]['content'][1]['text']}"
                    },
                    {
                        'type': 'image',
                        'image': img
                    }
                ]
            },
            {
                'role': 'assistant',
                'content': [
                    {
                        'type': 'text',
                        'text': example['messages'][2]['content']
                    }
                ]
            }
        ],
        #'image': example['messages'][1]['content'][0]['image']
    })
dataset = new_dataset
#print(dataset[0])
#exit()
model, tokenizer = FastVisionModel.from_pretrained(model_name="unsloth/Qwen2-VL-2B-Instruct-bnb-4bit")
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
training_args = SFTConfig(
    output_dir=f"./cot_vlm_v1",  # Directory to save the model
    num_train_epochs=1,  # Number of training epochs
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
    remove_unused_columns=False,
    dataset_text_field="",  # Text field in dataset
    dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
    max_seq_length=8192  # Maximum sequence length for input
)
trainer = SFTTrainer(
    model,
    args=training_args,
    train_dataset=dataset,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
)
trainer.train()