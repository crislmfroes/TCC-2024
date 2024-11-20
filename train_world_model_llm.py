from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import AutoTokenizer

model = "success_detector"

# loadcsv dataset
dataset = load_dataset("csv", data_files=f"./{model}_dataset/train.csv", split="train", delimiter=';', column_names=['prompt', 'completion'])
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['prompt'])):
        text = tokenizer.apply_chat_template([
            {
                'role': 'user',
                'content': example['prompt'][i]
            },
            {
                'role': 'assistant',
                'content': example['completion'][i]
            }
        ], tokenize=False)
        output_texts.append(text)
    return output_texts

training_args = SFTConfig(packing=False, output_dir=f'./{model}_checkpoints', auto_find_batch_size=True, max_seq_length=1024, save_total_limit=5)
trainer = SFTTrainer(
    "Qwen/Qwen2.5-0.5B-Instruct",
    args=training_args,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func
)
trainer.train()