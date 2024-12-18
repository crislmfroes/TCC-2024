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
import outlines.models.transformers_vision
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
import difflib

#engine = 'instructor'
engine = 'outlines'

class CustomAutoModel(AutoModel):
    def from_pretrained(pretrained_model_name_or_path, **kwargs):
        model = AutoModel.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)
        model.max_seq_len = 8192
        model.init_model()
        return model

if engine == 'outlines':
    mllm = outlines.models.transformers_vision(
        './actor_checkpoints_v4/checkpoint-15',
        model_class=Qwen2VLForConditionalGeneration,
        #"Qwen/Qwen2.5-7B-Instruct-AWQ",
        #"Qwen/Qwen2-VL-2B-Instruct",
        #repo_id="bartowski/Marco-o1-GGUF",
        #filename="Marco-o1-Q4_K_M.gguf"
        #trust_remote_code=True,
        #quantization='bitsandbytes'
        #model_class=AutoModelForCausalLM,
        #model_class=AutoModel,
        device="auto",
        #model_kwargs=dict(trust_remote_code=True),
        #processor_kwargs=dict(trust_remote_code=True)
        #model_kwargs=dict(load_in_4bit=True, bnb_4bit_use_double_quant=True, torch_dtype='auto')
    )

elif engine == 'instructor':
    #mllm = outlines.models.openai("nvidia/llama-3.1-nemotron-70b-instruct")
    #mllm.client.base_url = "https://integrate.api.nvidia.com/v1"
    reasoning_mllm = instructor.from_openai(OpenAI(base_url='http://localhost:11434/v1'), mode=instructor.Mode.JSON_O1)
    mllm = instructor.from_openai(OpenAI(base_url='http://localhost:11434/v1'), mode=instructor.Mode.JSON)

def infer_with_model(prompt: str, model: type[BaseModel], images=[], available_actions=[]):
    if engine == 'outlines':
        generator = outlines.generate.json(mllm, model, outlines.samplers.greedy())
        return generator(prompt, images)
    if engine == 'instructor':
        content = reasoning_mllm.client.chat.completions.create(model='marco-o1', messages=[{'role': 'user', 'content': prompt+"\n\nCHOOSE YOUR NEXT ACTION:"}]).choices[0].message.content
        print(content)
        output = content.split("<Output>")[1].split("</Output>")[0]
        thought = output
        action_choice = difflib.get_close_matches(output.split(',')[0], available_actions, cutoff=0.0)[0]
        return model(thought=thought, output=action_choice)
        #return reasoning_mllm.chat.completions.create(model='marco-o1', messages=[{'role': 'user', 'content': prompt}], max_retries=20, strict=True, response_model=model)
        #reasoning = reasoning_mllm.chat.completions.create(model='marco-o1', messages=[{'role': 'user', 'content': prompt}], response_model=str, strict=False)
        #return mllm.chat.completions.create(model='llama3.2', messages=[{'role': 'user', 'content': f'Parse the following message: \"{content.split("<Output>")[1].split("</Output>")[0]}\" into one of the following actions: {available_actions}'}], response_model=model, max_retries=20)


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
env = getattr(environment, env_type)(config, train_eval='eval_out_of_distribution')
env: AlfredThorEnv = env.init_env(batch_size=1)
success = 0.0
n_actions_sampled = 1
with open('action_prompt.md', 'r') as f:
    prompt_template = f.read()
with open('example.json', 'r') as f:
    example = json.load(f)
for k in example.keys():
    example[k] = example[k][:]
task_types = ['put', 'clean', 'heat', 'cool', 'put two', 'examine'][::-1]
episodes = 0
for i in tqdm.trange(192):
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
    previous_actions = [{'thought': None}]
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
                thought: str = Field()
                #current_step: str = Field()
                #plan: str = Field()
                #action: str = Field()
                action_choice: str = Field()#Literal[tuple(info['admissible_commands'][0])] = Field()
                #next_step: str = Field()
                #action_choice: str = Field()
            images = [Image.fromarray(f[:,:,::-1]) for f in env.get_frames()]
            vision = f"{''.join(['<|vision_start|><|image_pad|><|vision_end|>',]*len(images))}"
            prompt = prompt_template.format(task=task, previous_actions=json.dumps(previous_actions), text_obs=obs[0]+"\nReceptacles in the room: "+receptacles+"\nTASK DONE: FALSE", available_actions=json.dumps(info['admissible_commands'][0]), example=json.dumps(current_example), vision=vision)
            prompt += '\n\n### Current Observation\n\n<|vision_start|><|image_pad|><|vision_end|>'
            #print(prompt)
            #exit()
            actions: List[Action] = []
            for n in range(n_actions_sampled):
                action: Action = infer_with_model(prompt, Action, images, info['admissible_commands'][0])
                actions.append(action)
            choice_counter: Counter = Counter([action.action_choice for action in actions])
            most_common_choice = choice_counter.most_common(1)[0][0]
            action = [a for a in actions if a.action_choice == most_common_choice][0]
            print(obs[0])
            print(action.thought)
            print(action.action_choice)
            previous_actions.append({
                'observation': obs[0],
                #'reflection': action.reflection,
                'thought': action.thought,
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
        episodes += 1
    except BaseException as e:
        print(e)
        if isinstance(e, KeyboardInterrupt):
            break
    if episodes > 0:
        print(f'success rate: {success/episodes}')
        print(f'reward: {score}')
        print(f'env steps: {counter}')