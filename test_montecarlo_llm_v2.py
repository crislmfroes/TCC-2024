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
from search_utils.montecarlo import MonteCarlo
from search_utils.node import Node
from copy import deepcopy, copy
import pandas as pd
import torch
from langchain_experimental.tools.python.tool import PythonREPLTool, sanitize_input

random.seed(123)

from transformers import AutoTokenizer, AutoModelForCausalLM
from model_utils.prm_model import PRM_MODEL
from model_utils.io_utils import prepare_input, prepare_batch_input_for_model, derive_step_rewards

prm_model_path = "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
policy_model_path = "Skywork/Skywork-o1-Open-Llama-3.1-8B"
#policy_model_path = "meta-llama/Llama-3.1-8B-Instruct"
prm_tokenizer = AutoTokenizer.from_pretrained(prm_model_path, trust_remote_code=True)
policy_tokenizer = AutoTokenizer.from_pretrained(policy_model_path, trust_remote_code=True)
datas = [
    {
        "problem"  : "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "response" : "To determine how much money Janet makes every day at the farmers' market, we need to follow these steps:\n1. **Calculate the total number of eggs laid by the ducks per day.**\n   Janet's ducks lay 16 eggs per day.\n2. **Determine the number of eggs Janet uses each day.**\n   - She eats 3 eggs for breakfast every morning.\n   - She bakes muffins for her friends every day with 4 eggs.\n   So, the total number of eggs used per day is:\n   \\[\n   3 + 4 = 7 \\text{ eggs}\n   \\]\n3. **Calculate the number of eggs Janet sells at the farmers' market each day.**\n   Subtract the number of eggs used from the total number of eggs laid:\n   \\[\n   16 - 7 = 9 \\text{ eggs}\n   \\]\n4. **Determine how much money Janet makes from selling the eggs.**\n   She sells each egg for $2, so the total amount of money she makes is:\n   \\[\n   9 \\times 2 = 18 \\text{ dollars}\n   \\]\nTherefore, the amount of money Janet makes every day at the farmers' market is \\(\\boxed{18}\\)."
    },
    {
        "problem"  : "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "response" : "To find out how much Janet makes every day at the farmers' market, we need to follow these steps:\n1. Calculate the total number of eggs laid by the ducks: \n   - Janet's ducks lay 16 eggs per day.\n2. Calculate the total number of eggs consumed by Janet and used for baking:\n   - Janet eats 3 eggs for breakfast.\n   - She bakes 4 eggs for muffins.\n   - Total eggs used: 3 (breakfast) + 4 (baking) = 7 eggs.\n3. Calculate the remaining number of eggs for sale:\n   - Total eggs laid: 16\n   - Eggs used: 7\n   - Remaining eggs: 16 - 7 = 9 eggs\n4. Calculate the total amount of money made at the farmers' market:\n   - Price per egg: $2\n   - Number of eggs sold: 9\n   - Total money made: 9 * $2 = $18\nTherefore, Janet makes $\\boxed{18}$ dollars every day at the farmers' market."
    }
]

prm_model = PRM_MODEL.from_pretrained(prm_model_path, device_map="auto", load_in_4bit=True).eval()
policy_model = AutoModelForCausalLM.from_pretrained(policy_model_path, device_map="auto", load_in_4bit=True).eval()

def infer_step_rewards(datas, prm_model, prm_tokenizer):
    processed_data = [prepare_input(d["problem"], d["response"], tokenizer=prm_tokenizer, step_token="\n") for d in datas]
    input_ids, steps, reward_flags = zip(*processed_data)

    input_ids, attention_mask, reward_flags = prepare_batch_input_for_model(input_ids, reward_flags, prm_tokenizer.pad_token_id)
    _, _, rewards = prm_model(input_ids=input_ids, attention_mask=attention_mask, return_probs=True)
    step_rewards = derive_step_rewards(rewards, reward_flags)
    return step_rewards

def encode_to_policy(messages, policy_tokenizer, n=1, add_generation_prompt=True):
    input_ids = [policy_tokenizer.apply_chat_template(
        messages, 
        tokenize=True, 
        add_generation_prompt=add_generation_prompt,
        return_tensors="pt").to(policy_model.device) for idx in range(n)]
    #print(input_ids)
    return input_ids

def decode_from_policy(generation, input_ids, policy_tokenizer):
    completions = [policy_tokenizer.decode(
        generation[idx][0][len(input_ids[idx][0]):], 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=True) for idx in range(len(generation))]
    return completions

def infer_policy(input_ids, policy_model, policy_tokenizer, do_sample=True, stop_strings=['\n']):
    generation = [policy_model.generate(
        input_ids=input_ids[idx],
        max_new_tokens=2048,
        do_sample=do_sample,
        pad_token_id=128009,
        temperature=0.7,
        stop_strings=stop_strings,
        tokenizer=policy_tokenizer) for idx in range((len(input_ids)))]
    #print(generation)
    return generation

def generate_mcts(prompt, system_message, policy_model, policy_tokenizer, prm_model, prm_tokenizer, n=5):
    state = {
        'problem': prompt,
        'response': '',
        'input_ids': encode_to_policy(messages=[{
            'role': 'system',
            'content': system_message
        },
        {
            'role': 'user',
            'content': prompt
        }], policy_tokenizer=policy_tokenizer, n=n),
        'is_terminal': False
    }
    def child_finder(node: Node, montecarlo: MonteCarlo):
        generation = infer_policy(input_ids=node.state['input_ids'], policy_model=policy_model, policy_tokenizer=policy_tokenizer)
        completions = decode_from_policy(generation=generation, input_ids=node.state['input_ids'], policy_tokenizer=policy_tokenizer)
        child_states = []
        for idx in range(len(completions)):
            #print(completions[idx])
            new_state = deepcopy(node.state)
            new_state['response'] += '\n' + completions[idx].strip('\n')
            new_state['is_terminal'] = generation[idx][0][-1] == policy_tokenizer.eos_token_id
            #print('terminal: ', new_state['is_terminal'])
            #print('---')
            #print(new_state['response'])
            #print('---')
            child_states.append(new_state)
        datas = [{'problem': e['problem'], 'response': e['response']} for e in child_states]
        rewards = [sum(r)/len(r) for r in infer_step_rewards(datas=datas, prm_model=prm_model, prm_tokenizer=prm_tokenizer)]
        for idx in range(len(child_states)):
            child_states[idx]['input_ids'] = encode_to_policy(messages=[{
                    'role': 'system',
                    'content': system_message
                },
                {
                    'role': 'user',
                    'content': prompt
                },
                {
                    'role': 'assistant',
                    'content': child_states[idx]['response']
                }], policy_tokenizer=policy_tokenizer, n=n, add_generation_prompt=False)
            if child_states[idx]['input_ids'][0].shape[-1] >= 2048:
                raise BaseException('Thought for too long ...')
        for idx in range(len(child_states)):
            child_node = Node(state=child_states[idx])
            reward = 0
            if len(rewards) > 0:
                #print(rewards)
                reward = rewards[idx]
            node.add_child(child=child_node)
            child_node.update_policy_value(value=reward)
            if child_node.state['is_terminal'] == True:
                win_value = 1
            else:
                win_value = 0
            child_node.update_win_value(value=win_value)
    root_node = Node(state=state)
    root_node.visits = 1
    #root_node.win_value = 1 if root_node.state['is_terminal'] == True else 0
    mcts = MonteCarlo(root_node=root_node)
    mcts.child_finder = child_finder
    mcts.child_finder(node=mcts.root_node, montecarlo=mcts)
    counter = 1
    while mcts.root_node.win_value != 1:
        mcts.simulate(expansion_count=1)
        print('Thinking' + '.'*counter)
        counter += 1
    choice: Node = mcts.make_choice()
    while len(choice.children) != 0:
        mcts.root_node = choice
        choice = mcts.make_choice()
    return choice.state['response']

def generate_greedy(prompt, system_message, policy_model, policy_tokenizer):
    input_ids = encode_to_policy(messages=[{
        'role': 'system',
        'content': system_message
    },
    {
        'role': 'user',
        'content': prompt
    }], policy_tokenizer=policy_tokenizer, n=1, add_generation_prompt=True)
    generation = infer_policy(input_ids=input_ids, policy_model=policy_model, policy_tokenizer=policy_tokenizer, do_sample=True, stop_strings=None)
    completions = decode_from_policy(generation=generation, input_ids=input_ids, policy_tokenizer=policy_tokenizer)
    return completions[0]

system_prompt = """You are Skywork-o1, a thinking model developed by Skywork AI, specializing in solving complex problems involving mathematics, coding, and logical reasoning through deep thought. When faced with a user's request, you first engage in a lengthy and in-depth thinking process to explore possible solutions to the problem. After completing your thoughts, you then provide a detailed explanation of the solution process in your response."""
prompt_template = """Given the following observation: {observation}

Given the following previous observation/action history:
{observation_action_history}

Given the following available actions: {available_actions}

You have access to the following tools:

- execute_action(action: str)->Tuple[Union[str, List[str]]]: This tool executes some action in the ALFRED environment, and return the observation obtained after executing the action (str), as well as the available actions after executing the last action (List[str]).
- print(content: str): Use the python print function to print relevant information, such as observations

Write some python code that uses the tools to accomplish the task"""

def generate_code(observation, observation_action_history, available_actions):
    prompt = prompt_template.format(
        observation=observation,
        observation_action_history=observation_action_history,
        available_actions=available_actions
    )
    #response = generate_mcts(prompt=prompt, system_message=system_prompt, policy_model=policy_model, policy_tokenizer=policy_tokenizer, prm_model=prm_model, prm_tokenizer=prm_tokenizer, n=1)
    response = generate_greedy(prompt=prompt, system_message=system_prompt, policy_model=policy_model, policy_tokenizer=policy_tokenizer)
    return response.split('```python')[-1].split('```')[0]

def execute_action(action: str):
    global obs, infos, done, env, counter
    obs, scores, dones, infos = env.step([action])
    done = dones[0]
    counter += 1
    return obs[0], infos['admissible_commands'][0]

config = generic.load_config()
env_type = config['env']['type'] # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'

# setup environment
env = getattr(environment, env_type)(config, train_eval='eval_out_of_distribution')
env = env.init_env(batch_size=1)
rows = []
success = 0

code_interpreter = PythonREPLTool()
code_interpreter.python_repl.locals = dict(execute_action=execute_action)
code_interpreter.python_repl.globals = dict(execute_action=execute_action)
for i in tqdm.trange(100):
    done = False
    obs, info = env.reset()
    observation_action_history = []
    counter = 0
    while not done:
        print('Observation: ', obs[0])
        code = generate_code(observation=obs[0], observation_action_history=observation_action_history, available_actions=info['admissible_commands'][0])
        print('Action: ', code)
        observation_action_history.append({
            'observation': obs[0],
            'available_actions': info['admissible_commands'][0],
            'generated_code': code
        })
        if 'import' not in code and 'def execute_action(' not in code and 'def print(' not in code:
            code = sanitize_input(code)
            obs[0] = code_interpreter.python_repl.run(command=code)
        counter += 1
    if counter < 50:
        success += 1
    print('success rate: ', success/(i+1))