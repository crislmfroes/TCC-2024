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
from montecarlo.montecarlo import MonteCarlo
from montecarlo.node import Node
from copy import deepcopy, copy
import pandas as pd

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
for i in tqdm.trange(1000):
    done = False
    obs, info = env.reset()
    trajectory = [{
        'action': None,
        'observation': obs[0],
    }]
    action = random.choice(info['admissible_commands'][0])
    while not done:
        obs, scores, dones, infos = env.step([action])
        done = dones[0]
        info = infos
        trajectory.append({
            'action': action,
            'observation': obs[0],
        })
        action = random.choice(info['admissible_commands'][0])
    rows += convert_traj_to_alpaca(trajectory=trajectory)

data = {
    'instruction': [r['instruction'] for r in rows],
    'output': [r['output'] for r in rows],
}

ds = pd.DataFrame.from_dict(data=data)
ds.to_csv('./train.csv', sep=';')