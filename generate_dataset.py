import os
import sys
import numpy as np
import alfworld.agents.environment as environment
import alfworld.agents.modules.generic as generic
from alfworld.agents.environment.alfred_thor_env import AlfredThorEnv
import json
import time
from collections import deque
import outlines.generate
import outlines.generate.choice
import outlines.generate.choice
import outlines.generate.json
import outlines.models
import outlines.models.transformers
import outlines.samplers
import tqdm
from pydantic import BaseModel, Field
from typing import Optional, Literal, List, Union
import outlines
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoModel, MllamaForConditionalGeneration, LlavaOnevisionForConditionalGeneration, LlavaForConditionalGeneration
import random
from collections import Counter
from copy import deepcopy, copy
import pandas as pd
from uuid import uuid4

with open('world_model_prompt.md', 'r') as f:
    world_model_prompt_template = f.read()

def convert_traj_to_alpaca(trajectory):
    rows = []
    for i in range(1, len(trajectory)):
        previous_steps = trajectory[:i-1]
        current_step = trajectory[i]
        instruction = world_model_prompt_template.format(previous_actions=json.dumps(previous_steps), current_action=current_step['action'])
        output = json.dumps({
            'observation': current_step['observation']
        })
        rows.append({
            'instruction': instruction,
            'output': output
        })
    return rows


config = generic.load_config()
env_type = config['env']['type'] # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'

# setup environment
env = getattr(environment, env_type)(config, train_eval='train')
env = env.init_env(batch_size=1)
rows = []
success = 0
for i in tqdm.trange(2000):
    done = False
    obs, info = env.reset()
    task = obs[0].split('\n\n')[-1]
    image = Image.fromarray(env.get_frames()[0][:,:,::-1])
    image_path = f'./images/{str(uuid4())}.jpg'
    image.save(image_path)
    action = info['extra.expert_plan'][0][0]
    trajectory = [{
        'observation': image_path,
        'available_actions': info['admissible_commands'][0],
        'action': action
    }]
    step_counter = 0
    while not done:
        obs, scores, dones, infos = env.step([action])
        step_counter += 1
        done = dones[0]
        info = infos
        action = info['extra.expert_plan'][0][0]
        image = Image.fromarray(env.get_frames()[0][:,:,::-1])
        image_path = f'./images/{str(uuid4())}.jpg'
        image.save(image_path)
        trajectory += [{
            'observation': image_path,
            'available_actions': info['admissible_commands'][0],
            'action': action
        }]
    row = {
        'trajectory': trajectory,
        'task_goal': task
    }
    if step_counter < 50:
        success += 1
        row['success'] = True
    else:
        row['success'] = False
    rows.append(row)

    print('success rate: ', success/(i+1))

    ds = pd.DataFrame.from_dict(data=rows)
    ds.to_json('./train.jsonl', lines=True, orient='records')