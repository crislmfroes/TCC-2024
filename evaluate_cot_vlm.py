import openai
import pandas as pd
from PIL import Image
import tqdm
import alfworld.agents.environment as environment
import alfworld.agents.modules.generic as generic
import base64
from io import BytesIO
import time

MODEL_TO_EVALUATE = "llama3.2-vision"
BASE_URL = "http://0.0.0.0:11434/v1"
RPM = 2000

with open('cot_generation_prompt.md', 'r') as f:
    cot_generation_prompt_template = f.read()

with open('cot_generation_system_prompt.md', 'r') as f:
    cot_generation_system_prompt = f.read()

config = generic.load_config()
env_type = config['env']['type'] # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'

# setup environment
env = getattr(environment, env_type)(config, train_eval='eval_out_of_distribution')
env = env.init_env(batch_size=1)

client = openai.OpenAI(base_url=BASE_URL)

def convert_image_to_base64(image: Image.Image):
    buffered = BytesIO()
    image.save(buffered, format='JPEG')
    return 'data:image/jpeg;base64,' + base64.b64encode(buffered.getvalue()).decode('utf-8')

def generate_prediction(previous_actions: list[str], available_actions: list[str], current_observation_image: Image.Image, task_goal: str, previous_observations: list[str], previous_thoughts: list[str]):
    prompt = cot_generation_prompt_template.format(task_goal=task_goal, previous_actions=previous_actions, available_actions=available_actions, previous_observations=previous_observations, previous_thoughts=previous_thoughts)
    messages = [
        {
            'role': 'system',
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
                    'type': 'image_url',
                    'image_url': {
                        'url': convert_image_to_base64(image=current_observation_image)
                    }
                },
                {
                    'type': 'text',
                    'text': prompt
                }
            ]
        }
    ]
    completion = client.chat.completions.create(messages=messages, model=MODEL_TO_EVALUATE, temperature=0, max_tokens=2048).choices[0].message.content
    time.sleep(60/RPM)
    return completion

success = 0
for i in tqdm.trange(100):
    done = False
    obs, info = env.reset()
    task_goal = obs[0].split('\n\n')[-1]
    image = Image.fromarray(env.get_frames()[0][:,:,::-1])
    previous_actions = []
    previous_observations = []
    previous_thoughts = []
    step_counter = 0
    for j in range(1):
        prediction = generate_prediction(previous_actions=previous_actions, available_actions=info['admissible_commands'][0], current_observation_image=image, task_goal=task_goal, previous_observations=previous_observations, previous_thoughts=previous_thoughts)
        action = prediction.split('<Action>')[-1].split('</Action>')[0]
        observation_description = prediction.split('<Observation>')[-1].split('</Observation')[0]
        thought = prediction.split('<Thought>')[-1].split('</Thought>')[0]
        if action in info['admissible_commands'][0]:
            break
    print(prediction)
    while not done:
        previous_observations.append(obs[0])
        previous_thoughts.append(thought)
        obs, scores, dones, infos = env.step([action])
        previous_actions.append(action)
        step_counter += 1
        done = dones[0]
        info = infos
        image = Image.fromarray(env.get_frames()[0][:,:,::-1])
        for j in range(1):
            prediction = generate_prediction(previous_actions=previous_actions, available_actions=info['admissible_commands'][0], current_observation_image=image, task_goal=task_goal, previous_observations=previous_observations, previous_thoughts=previous_thoughts)
            action = prediction.split('<Action>')[-1].split('</Action>')[0]
            observation_description = prediction.split('<Observation>')[-1].split('</Observation')[0]
            thought = prediction.split('<Thought>')[-1].split('</Thought>')[0]
            if action in info['admissible_commands'][0]:
                break
        print(prediction)
    if step_counter < 50:
        success += 1
    print('success rate: ', success/(i+1))