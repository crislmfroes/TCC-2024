import os
import sys
import numpy as np
import alfworld.agents.environment as environment
import alfworld.agents.modules.generic as generic
from alfworld.agents.environment.alfred_thor_env import AlfredThorEnv
import json
import time
from collections import deque
import tqdm
from pydantic import BaseModel, Field
from typing import Optional, Literal, List, Union
from PIL import Image
#from transformers import Qwen2VLForConditionalGeneration, AutoModel, MllamaForConditionalGeneration, LlavaOnevisionForConditionalGeneration, LlavaForConditionalGeneration
import random
from collections import Counter
from montecarlo.montecarlo import MonteCarlo
from montecarlo.node import Node
from copy import deepcopy, copy
import pandas as pd
from llamagym import Agent
from trl import AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
import difflib


config = generic.load_config()
env_type = config['env']['type'] # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'

# setup environment
env = getattr(environment, env_type)(config, train_eval='train')
env = env.init_env(batch_size=1)

class AlfworldAgent(Agent):
    def get_system_prompt(self) -> str:
        return """You are an general purpose service robot operating inside a domestic environment"""
    
    def format_observation(self, observation: str) -> str:
        return observation
    
    def extract_action(self, response: str) -> str:
        return response#response.split('Action: ')[-1].strip()

model_repo = "/home/fbot/AlphaHome/actor_checkpoints/checkpoint-48"
device = 'cuda:0'
model = AutoModelForCausalLMWithValueHead.from_pretrained(model_repo).to(device)
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
agent = AlfworldAgent(model, tokenizer, device)

for i in tqdm.trange(4000):
    done = False
    obs, info = env.reset()
    counter = 0
    while not done:
        observation = f"{obs[0]}\nAvailable Actions: {info['admissible_commands'][0]}"
        action = agent.act(observation)
        action_matches = difflib.get_close_matches(action, info['admissible_commands'][0], cutoff=0.0)
        print(action_matches[0])
        if len(action_matches) > 0:
            obs, scores, dones, infos = env.step([action_matches[0]])
        else:
            obs, scores, dones, infos = [f"Error: Invalid action '{action}'! Choose one of the following actions instead: {', '.join(info['admissible_commands'][0])}"], None, False, info
        done = dones[0]
        info = infos
        reward = 100.0 if done == True and counter < 50 else -1.0
        agent.assign_reward(reward)
        counter += 1
    print(counter)
    train_stats = agent.terminate_episode()
    print(train_stats)