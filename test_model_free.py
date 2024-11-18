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
import outlines.models.openai
import outlines.models.transformers
import outlines.models.transformers
import outlines.models.vllm
import outlines.samplers
import instructor
import tqdm
from pydantic import BaseModel, Field
from typing import Optional, Literal, List, Union
import outlines
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoModel, MllamaForConditionalGeneration, LlavaOnevisionForConditionalGeneration, LlavaForConditionalGeneration
import random
from collections import Counter
from airllm.auto_model import AutoModel
from openai import OpenAI

#engine = 'instructor'
engine = 'outlines'

class CustomAutoModel(AutoModel):
    def from_pretrained(pretrained_model_name_or_path, **kwargs):
        model = AutoModel.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)
        model.max_seq_len = 8192
        model.init_model()
        return model

if engine == 'outlines':
    mllm = outlines.models.transformers(
        "Qwen/Qwen2.5-14B-Instruct-AWQ",
        #max_num_seqs=1,
        #enforce_eager=True,
        #max_model_len=8192,
        #quantization="",
        #"OpenGVLab/InternVL2-1B",
        #model_class=Qwen2VLForConditionalGeneration,
        #model_class=CustomAutoModel,
        device="auto",
        #model_kwargs=dict(load_in_4bit=True, bnb_4bit_use_double_quant=True, torch_dtype='auto')
    )

elif engine == 'instructor':
    #mllm = outlines.models.openai("nvidia/llama-3.1-nemotron-70b-instruct")
    #mllm.client.base_url = "https://integrate.api.nvidia.com/v1"
    mllm = instructor.from_openai(OpenAI(base_url='https://integrate.api.nvidia.com/v1'))

def infer_with_model(prompt: str, model: type[BaseModel], images=[]):
    if engine == 'outlines':
        generator = outlines.generate.json(mllm, model, outlines.samplers.greedy())
        return generator(prompt)
    if engine == 'instructor':
        return mllm.chat.completions.create(model='meta/llama-3.1-70b-instruct', response_model=model, messages=[{'role': 'user', 'content': prompt}])


config = generic.load_config()
env_type = config['env']['type'] # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'

class Item(BaseModel):
    name: str = Field()
    cleaning_state: Literal['clean', 'dirty']
    heat_state: Literal['heated', 'cold']

class Furniture(BaseModel):
    name: str = Field()
    items_on_top: List[Item] = Field()
    opening: Optional[Literal['open', 'closed']] = Field()

class Appliance(BaseModel):
    name: str = Field()
    items_inside: List[Item] = Field()
    state: Optional[Literal['on', 'off']] = Field()
    opening: Optional[Literal['open', 'closed']] = Field()

class WorldState(BaseModel):
    furnitures: List[Furniture] = Field()
    appliances: List[Appliance] = Field()
    item_in_hand: Optional[Item] = Field()
    current_location: str = Field()

# setup environment
env = getattr(environment, env_type)(config, train_eval='eval_in_distribution')
env: AlfredThorEnv = env.init_env(batch_size=1)
success = 0.0
n_actions_sampled = 1
with open('prompt_agent.md', 'r') as f:
    prompt_template = f.read()
with open('example.json', 'r') as f:
    example = json.load(f)
for k in example.keys():
    example[k] = example[k][:10]
task_types = ['put', 'clean', 'heat', 'cool', 'put two', 'examine'][::-1]
for i in tqdm.trange(10):
    done = False
    obs, info = env.reset()
    print(obs[0].split('\n\n'))
    task = obs[0].split('\n\n')[-1]
    receptacles = obs[0].split('\n\n')[1][len("You are in the middle of a room. Looking quickly around you, you see"):]
    #exit()
    print(task)
    current_task_types = [t for t in task_types if t in task][:1]
    current_example = []
    for t in current_task_types:
        for k in example.keys():
            if t == 'put':
                current_example.append(example['react_put_0'])
                #current_example.append(example['react_put_1'])
            if t == 'clean':
                current_example.append(example['react_clean_0'])
                #current_example.append(example['react_clean_1'])
            if t == 'heat':
                current_example.append(example['react_heat_0'])
                #current_example.append(example['react_heat_1'])
            if t == 'cool':
                current_example.append(example['react_cool_0'])
                #current_example.append(example['react_cool_1'])
            if t == 'puttwo':
                current_example.append(example['react_puttwo_0'])
                #current_example.append(example['react_puttwo_1'])
            if t == 'examine':
                current_example.append(example['react_examine_0'])
                #current_example.append(example['react_examine_1'])
    #print(obs)
    #print(env.get_frames())
    #exit()
    previous_actions = []
    score = 0
    counter = 0
    try:
        while not done and counter < 50:
            #info['admissible_commands'][0] = [c for c in info['admissible_commands'][0] if c not in ['inventory']]
            class Action(BaseModel):
                #current_world_state: WorldState = Field()
                #desired_world_state: WorldState = Field()
                #observation_description: str = Field()
                #reflection: str = Field()
                think: str = Field()
                #current_step: str = Field()
                #plan: str = Field()
                #action: str = Field()
                action_choice: Union[Literal[tuple(info['admissible_commands'][0])],] = Field()
                #next_step: str = Field()
                #action_choice: str = Field()
            images = [Image.fromarray(f) for f in env.get_frames()]
            vision = f"{''.join(['<|image|>',]*len(images))}"
            prompt = prompt_template.format(task=task, previous_actions=json.dumps(previous_actions), text_obs=obs[0]+"\nReceptacles in the room: "+receptacles+"\nTASK DONE: FALSE", available_actions=json.dumps(info['admissible_commands'][0]), example=json.dumps(current_example), vision=vision)
            #print(prompt)
            #exit()
            actions: List[Action] = []
            for n in range(n_actions_sampled):
                action: Action = infer_with_model(prompt, Action, images)
                actions.append(action)
            choice_counter: Counter = Counter([action.action_choice for action in actions])
            most_common_choice = choice_counter.most_common(1)[0][0]
            action = [a for a in actions if a.action_choice == most_common_choice][0]
            print(obs[0])
            print(action)
            previous_actions.append({
                'observation': obs[0],
                #'reflection': action.reflection,
                'think': action.think,
                #'current_step': action.current_step,
                #'plan': action.plan,
                #'thought': action.thought,
                #'action': action.action,
                'action_choice': action.action_choice,
                #'next_step': action.next_step
            })
            obs, scores, dones, infos = env.step([action.action_choice])
            done = dones[0]
            info = infos
            counter += 1
        if counter < (50):
            success += 1
    except BaseException as e:
        print(e)
        if isinstance(e, KeyboardInterrupt):
            break
    print(f'success rate: {success/(i+1)}')
    print(f'reward: {score}')
    print(f'env steps: {counter}')
    
    