import os
import sys
sys.path.append(os.environ['TITOK_TOKENIZER_ROOT'])
sys.path.append(os.environ['ALFRED_ROOT'])
sys.path.append(os.environ['ALFRED_ROOT']+'/gen')
from modeling.titok import TiTok
import torch
import numpy as np
from PIL import Image
from transformers import LlamaTokenizer
import json
import math
from datasets import Dataset

SPLIT = 'train'

HIGH_LEVEL_POLICY_TOKEN = "<hl_policy>"
LOW_LEVEL_POLICY_TOKEN = "<ll_policy>"
HIGH_LEVEL_WORLD_MODEL_TOKEN = "<hl_world_model>"
HIGH_LEVEL_REWARD_MODEL_TOKEN = "<hl_reward_model>"
OBS_BEGIN_TOKEN = "<obs>"
OBS_END_TOKEN = "</obs>"

def load_image(img_path: str):
    return torch.from_numpy(np.array(Image.open(img_path)).astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0

def tokenize_image(img_path: str, titok_tokenizer: TiTok):
    image = load_image(img_path).to(titok_tokenizer.device)
    image_tokens = titok_tokenizer.encode(image.to(device))[1]["min_encoding_indices"].cpu().numpy().flatten()
    image_tokens_str = OBS_BEGIN_TOKEN + "".join([f"<obs_{i}>" for i in image_tokens]) + OBS_END_TOKEN
    return image_tokens_str

def get_obs_tokens(codebook_size: int=4096):
    token_map = {}
    for i in range(codebook_size):
        token_map[i] = f"<obs_{i}>"
    return token_map

def get_added_tokens():
    added_tokens = []
    added_tokens += [v for k,v in get_obs_tokens().items()]
    added_tokens += [
        HIGH_LEVEL_POLICY_TOKEN,
        LOW_LEVEL_POLICY_TOKEN,
        HIGH_LEVEL_WORLD_MODEL_TOKEN,
        HIGH_LEVEL_REWARD_MODEL_TOKEN,
        OBS_BEGIN_TOKEN,
        OBS_END_TOKEN
    ]
    return added_tokens

def tokenize_reward(reward: int):
    return str(reward)

def tokenize_action(action: dict, is_terminal: bool=False):
    processed_action = action.copy()
    if 'mask' in processed_action['args']:
        processed_action['args'].pop('mask')
    if 'bbox' in processed_action['args']:
        processed_action['args']['bbox'] = [float(f"{x/300:.2f}") for x in processed_action['args']['bbox']]
    return json.dumps({
        'action': processed_action['action'],
        'args': processed_action['args'],
        'terminal': is_terminal
    })

if __name__ == '__main__':
    titok_tokenizer = TiTok.from_pretrained("yucornetto/tokenizer_titok_l32_imagenet")
    titok_tokenizer.eval()
    titok_tokenizer.requires_grad_(False)
    device = "cuda"
    titok_tokenizer = titok_tokenizer.to(device)
    llm_tokenizer: LlamaTokenizer = LlamaTokenizer.from_pretrained("Llama-3.2-1B", trust_remote_code=True)
    llm_tokenizer.add_tokens(get_added_tokens())
    output_dataset = {'instruction': [], 'output': []}
    with open(os.environ['ALFRED_ROOT']+'/data/splits/oct21.json', 'r') as f:
        dataset = json.load(f)
    for i in tqdm.trange(len(dataset[SPLIT])):
        task = dataset[SPLIT][i]['task']
        with open(os.environ['ALFRED_ROOT']+'/data/full_2.1.0/'+SPLIT+'/'+task+'/traj_data.json', 'r') as f:
            traj_data = json.load(f)
        task_desc = traj_data['turk_annotations']['anns'][dataset[SPLIT][i]['repeat_idx']]['task_desc']
        high_descs = traj_data['turk_annotations']['anns'][dataset[SPLIT][i]['repeat_idx']]['high_descs']
        low_actions = traj_data['plan']['low_actions']

        # High level policy
        obs_act_history = ""
        obs = ""
        for hidx, high_desc in enumerate(high_descs+['TERMINATE',]):
            for e in traj_data['images']:
                if e['high_idx'] == hidx:
                    obs = tokenize_image(img_path=os.environ['ALFRED_ROOT']+'/data/full_2.1.0/'+SPLIT+'/'+task+'/raw_images/'+e['image_name'], titok_tokenizer=titok_tokenizer)
                    break
            if high_desc == 'TERMINATE':
                obs = tokenize_image(img_path=os.environ['ALFRED_ROOT']+'/data/full_2.1.0/'+SPLIT+'/'+task+'/raw_images/'+traj_data['images'][-1]['image_name'], titok_tokenizer=titok_tokenizer)
            obs_act_history += obs
            instruction = f"{HIGH_LEVEL_POLICY_TOKEN}{task_desc}{obs_act_history}"
            output = f"{high_desc}"
            output_dataset['instruction'].append(instruction)
            output_dataset['output'].append(output)
            obs_act_history += f"{high_desc}"

        # High level world model
        obs_act_history = ""
        obs = ""
        next_obs = ""
        action = ""
        for hidx, high_desc in enumerate(high_descs):
            for e in traj_data['images']:
                if e['high_idx'] == hidx:
                    obs = tokenize_image(img_path=os.environ['ALFRED_ROOT']+'/data/full_2.1.0/'+SPLIT+'/'+task+'/raw_images/'+e['image_name'], titok_tokenizer=titok_tokenizer)
                    break
            for e in traj_data['images']:
                if e['high_idx'] == hidx+1:
                    next_obs = tokenize_image(img_path=os.environ['ALFRED_ROOT']+'/data/full_2.1.0/'+SPLIT+'/'+task+'/raw_images/'+e['image_name'], titok_tokenizer=titok_tokenizer)
                    break
            action = f"{high_desc}"
            obs_act_history += obs + action
            instruction = f"{HIGH_LEVEL_WORLD_MODEL_TOKEN}{task_desc}{obs_act_history}"
            output = next_obs
            output_dataset['instruction'].append(instruction)
            output_dataset['output'].append(output)

        # High level reward model
        obs_act_history = ""
        obs = ""
        for hidx, high_desc in enumerate(high_descs+['TERMINATE',]):
            for e in traj_data['images']:
                if e['high_idx'] == hidx:
                    obs = tokenize_image(img_path=os.environ['ALFRED_ROOT']+'/data/full_2.1.0/'+SPLIT+'/'+task+'/raw_images/'+e['image_name'], titok_tokenizer=titok_tokenizer)
                    break
            if high_desc == 'TERMINATE':
                obs = tokenize_image(img_path=os.environ['ALFRED_ROOT']+'/data/full_2.1.0/'+SPLIT+'/'+task+'/raw_images/'+traj_data['images'][-1]['image_name'], titok_tokenizer=titok_tokenizer)
            obs_act_history += obs
            reward = hidx - len(high_desc) + 1
            instruction = f"{HIGH_LEVEL_REWARD_MODEL_TOKEN}{task_desc}{obs_act_history}"
            output = tokenize_reward(reward)
            output_dataset['instruction'].append(instruction)
            output_dataset['output'].append(output)
            obs_act_history += f"{high_desc}"

        # Low level policy
        obs_act_history = ""
        obs = ""
        for lidx, low_action in enumerate(low_actions):
            for e in traj_data['images']:
                if e['low_idx'] == lidx:
                    obs = tokenize_image(img_path=os.environ['ALFRED_ROOT']+'/data/full_2.1.0/'+SPLIT+'/'+task+'/raw_images/'+e['image_name'], titok_tokenizer=titok_tokenizer)
                    break
            obs_act_history += obs
            high_desc = high_descs[low_action['high_idx']]
            instruction = f"{LOW_LEVEL_POLICY_TOKEN}{high_desc}{obs_act_history}"
            action = tokenize_action(low_action['discrete_action'], is_terminal=(lidx == len(low_actions)-1) or (low_action['high_idx'] != low_actions[lidx+1]['high_idx']))
            output = action
            output_dataset['instruction'].append(instruction)
            output_dataset['output'].append(output)
            obs_act_history += action
    
    
    Dataset.from_dict(mapping=dataset, split=SPLIT).push_to_hub("crislmfroes/AlphaHome-ALFRED", private=True)
    llm_tokenizer.push_to_hub("crislmfroes/AlphaHome-Llama-3.2-1B", private=True)