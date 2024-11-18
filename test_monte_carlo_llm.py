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
import outlines.generate.json
import outlines.models
import outlines.models.openai
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
import time
from threading import Thread, Event

mllm = outlines.models.transformers(
    "Qwen/Qwen2.5-14B-Instruct-AWQ",
    #"OpenGVLab/InternVL2-1B",
    #model_class=Qwen2VLForConditionalGeneration,
    #model_class=AutoModel,
    device="auto",
    #model_kwargs=dict(load_in_4bit=True, bnb_4bit_use_double_quant=True, torch_dtype='auto')
)
mllm_world_model = outlines.models.transformers(
    "/home/fbot/AlphaHome/world_model_checkpoints/checkpoint-149781",
    #"OpenGVLab/InternVL2-1B",
    #model_class=Qwen2VLForConditionalGeneration,
    #model_class=AutoModel,
    device="auto",
    #model_kwargs=dict(load_in_4bit=True, bnb_4bit_use_double_quant=True, torch_dtype='auto')
)
config = generic.load_config()
env_type = config['env']['type'] # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'

# setup environment
env = getattr(environment, env_type)(config, train_eval='eval_in_distribution')
env = env.init_env(batch_size=1)
success = 0.0
n_actions_sampled = 5
with open('action_prompt.md', 'r') as f:
    action_prompt_template = f.read()
with open('final_action_prompt.md', 'r') as f:
    final_action_prompt_template = f.read()
with open('world_model_prompt.md', 'r') as f:
    world_model_prompt_template = f.read()
with open('value_estimation_prompt.md', 'r') as f:
    value_estimation_prompt_template = f.read()
with open('success_detection_prompt.md', 'r') as f:
    success_detection_prompt_template = f.read()
with open('example.json', 'r') as f:
    example = json.load(f)
for k in example.keys():
    example[k] = []#example[k][:10]
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
    world_state = [{
        'think': None,
        'action': None,
        'observation': obs[0]
    }]
    images = []#[Image.fromarray(frame) for frame in env.get_frames()]
    score = 0
    counter = 0
    try:
        while not done and counter < 50:
            #info['admissible_commands'][0] = [c for c in info['admissible_commands'][0] if c not in ['inventory']]
            print('N frames: ', len(images))
            class Action(BaseModel):
                #current_world_state: WorldState = Field()
                #desired_world_state: WorldState = Field()
                #observation_description: str = Field()
                #reflection: str = Field()
                think: str = Field()
                #current_step: str = Field()
                #plan: str = Field()
                #action: str = Field()
                action_choice: Union[Literal[tuple(info['admissible_commands'][0])], str] = Field()
                #next_step: str = Field()
                #action_choice: str = Field()
            class FinalAction(BaseModel):
                action_choice: Literal[tuple(info['admissible_commands'][0])] = Field()
            
            class Observation(BaseModel):
                observation: str = Field()

            class Reward(BaseModel):
                score: int = Field()

            class TaskStatus(BaseModel):
                task_completed: bool = Field()

            montecarlo = MonteCarlo(Node(state=world_state))
            montecarlo.root_node.visits = 1
            action_generator = outlines.generate.json(mllm, Action)
            final_action_generator = outlines.generate.json(mllm, FinalAction)#, outlines.samplers.greedy())
            observation_generator = outlines.generate.json(mllm_world_model, Observation)#, outlines.samplers.greedy())
            reward_generator = outlines.generate.json(mllm, Reward)#, outlines.samplers.greedy())
            success_detector = outlines.generate.json(mllm, TaskStatus)#, outlines.samplers.greedy())

            def child_finder(node: Node, montecarlo: MonteCarlo, start=False, available_actions=[]):
                for n in range(n_actions_sampled):
                        action_prompt = action_prompt_template.format(example=json.dumps(current_example), previous_actions=json.dumps(node.state), task=task)
                        if start == True:
                            action_prompt += f"\n\n### Available Actions\n\n{available_actions}"
                        action: Action = action_generator(action_prompt)
                        observation_prompt = world_model_prompt_template.format(example=json.dumps(current_example), previous_actions=json.dumps(node.state), current_action=action.action_choice)
                        observation: Observation = observation_generator(observation_prompt, max_tokens=128)
                        next_world_state = node.state + [{
                            'think': action.think,
                            'action': action.action_choice,
                            'observation': observation.observation
                        }]
                        child = Node(next_world_state)
                        reward_prompt = value_estimation_prompt_template.format(previous_actions=next_world_state)
                        reward: Reward = reward_generator(reward_prompt)
                        child.policy_value = min(1.0, max(0.0, float(reward.score/10.0)))
                        child.visits = 1
                        node.add_child(child)
                        #print('--------')
                        #print('N: ', n)
                        #print(action)
                        #print(observation)
                        #print(reward)
                        #print(child.policy_value)
                        #print(node.policy_value)
                        #print(node.visits)
                        #print('--------')
                        #time.sleep(5.0)
                success_prompt = success_detection_prompt_template.format(previous_actions=node.state)
                task_status: TaskStatus = success_detector(success_prompt)
                #print('--------')
                #print(task_status)
                #print('--------')
                if task_status.task_completed == True:
                    node.update_win_value(1)
                #while len(threads) > sum([1 for e in events if e.is_set()]):
                #    print(sum([1 for e in events if e.is_set()]))

            child_finder(node=montecarlo.root_node, montecarlo=montecarlo, start=True, available_actions=info['admissible_commands'][0])
            montecarlo.child_finder = child_finder
            montecarlo.simulate(expansion_count=2)
            choosen_child: Node = montecarlo.make_choice()
            action: str = choosen_child.state[-1]['action']
            thought: str = choosen_child.state[-1]['think']
            final_action_prompt = final_action_prompt_template.format(allowed_actions = info['admissible_commands'][0], action=action)
            final_action: FinalAction = final_action_generator(final_action_prompt)
            print('Thought: ', thought)
            print('Action: ', final_action.action_choice)

            '''vision = f"<|vision_start|>{''.join(['<|image_pad|>',]*len(images))}<|vision_end|>"
            prompt = prompt_template.format(task=task, previous_actions=json.dumps(previous_actions), vision=vision, text_obs=obs[0]+"\nReceptacles in the room: "+receptacles+"\nTASK DONE: FALSE", available_actions=json.dumps(info['admissible_commands'][0]), example=json.dumps(current_example))
            #print(prompt)
            #exit()
            generator = outlines.generate.json(mllm, Action, outlines.samplers.greedy())
            actions: List[Action] = []
            for n in range(n_actions_sampled):
                action: Action = generator(prompt)#, images)
                actions.append(action)
            choice_counter: Counter = Counter([action.action_choice for action in actions])
            most_common_choice = choice_counter.most_common(1)[0][0]
            action = [a for a in actions if a.action_choice == most_common_choice][0]
            print(obs[0])
            print(action)'''
            #previous_actions.append({
                #'observation': obs[0],
                #'reflection': action.reflection,
                #'think': action.think,
                #'current_step': action.current_step,
                #'plan': action.plan,
                #'thought': action.thought,
                #'action': action.action,
                #'action_choice': action.action_choice,
                #'next_step': action.next_step
            #})
            obs, scores, dones, infos = env.step([final_action.action_choice])
            world_state.append({
                'think': thought,
                'action': final_action.action_choice,
                'observation': obs[0]
            })
            done = dones[0]
            info = infos
            images = []#[Image.fromarray(frame) for frame in env.get_frames()]
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
    
    