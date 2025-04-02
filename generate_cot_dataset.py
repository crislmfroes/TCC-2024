from unsloth.models import FastVisionModel
import pandas as pd
from PIL import Image
import tqdm

max_actions_to_sample = 5

df = pd.read_json('./train.jsonl', lines=True)
success_df = df[df['success'] == True]

with open('cot_generation_prompt.md', 'r') as f:
    cot_generation_prompt_template = f.read()

with open('cot_generation_system_prompt.md', 'r') as f:
    cot_generation_system_prompt = f.read()

model, tokenizer = FastVisionModel.from_pretrained(model_name="unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit")
FastVisionModel.for_inference(model)

def generate_prediction(previous_actions: list[str], available_actions: list[str], current_observation_image: str, task_goal: str):
    prompt = cot_generation_prompt_template.format(task_goal=task_goal, previous_actions=previous_actions, available_actions=available_actions)
    messages = [
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': cot_generation_system_prompt
                }
            ]
        },
        {
            'role': 'user',
            'content': [
                {
                    'type': 'image'
                },
                {
                    'type': 'text',
                    'text': prompt
                }
            ]
        }
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(Image.open(current_observation_image), input_text, add_special_tokens=False, return_tensors='pt').to('cuda')
    generation = model.generate(**inputs, max_new_tokens=2048, use_cache=True)
    completion = tokenizer.decode(generation[0].detach().cpu().numpy()[len(inputs['input_ids'][0]):], skip_special_tokens=True)
    return completion

rows = []
for trajectory, task_goal in tqdm.tqdm(zip(success_df['trajectory'], success_df['task_goal'])):
    previous_actions = []
    for trajectory_step in tqdm.tqdm(trajectory):
        for _ in tqdm.trange(max_actions_to_sample):
            prediction = generate_prediction(previous_actions=previous_actions, available_actions=trajectory_step['available_actions'], current_observation_image=trajectory_step['observation'], task_goal=task_goal)
            predicted_action = prediction.split('<Action>')[-1].split('</Action>')[0]
            if predicted_action == trajectory_step['action']:
                prompt = cot_generation_prompt_template.format(task_goal=task_goal, previous_actions=previous_actions, available_actions=trajectory_step['available_actions'])
                rows.append({
                    'messages': [
                        {
                            'role': 'user',
                            'content': [
                                {
                                    'type': 'text',
                                    'text': cot_generation_system_prompt
                                }
                            ]
                        },
                        {
                            'role': 'user',
                            'content': [
                                {
                                    'type': 'image',
                                    'image': trajectory_step['observation']
                                },
                                {
                                    'type': 'text',
                                    'text': prompt
                                }
                            ]
                        },
                        {
                            'role': 'assistant',
                            'content': prediction
                        }
                    ]
                })
                ds = pd.DataFrame.from_dict(data=rows)
                ds.to_json('./train_cot.jsonl', lines=True, orient='records')
                break