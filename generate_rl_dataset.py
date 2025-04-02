import tqdm
from datasets import Dataset
from train_agent_with_rl import ALFWORLD_PROMPT
from alfworld.agents.environment import get_environment
from alfworld.agents.environment.alfred_thor_env import AlfredThorEnv
import alfworld.agents.modules.generic as generic
from PIL import Image

config = generic.load_config()
env_type = config['env']['type']
env = get_environment(env_type)(config, train_eval='train')
env: AlfredThorEnv = env.init_env(batch_size=1)
_, _ = env.reset()
obs, _, _, _ = env.step(["help"])
help_text = obs[0]

def get_rl_dataset(**kwargs):
        conversations = []
        responses = []
        choices = []
        for i in tqdm.trange(3000):
            obs, info = env.reset()
            observation = obs[0]
            done = False
            steps = 0
            actions = []
            observations = []
            admissible_actions = []
            while not done:
                observations.append(observation)
                admissible_actions.append(info['admissible_commands'][0])
                action = info['extra.expert_plan'][0][0]
                actions.append(action)
                obs, reward, done, info = env.step([action])
                observation =  obs[0]
                done = done[0]
                steps += 1
            if steps < 50:
                for j, (action, observation, admissible_action) in enumerate(zip(actions, observations, admissible_actions)):
                    memory = "Memory:"
                    for previous_obs, previous_action in zip(actions[:j], observations[:j]):
                         memory += f"Observation: {previous_obs}\n"
                         memory += f"Thought: I must now {previous_action}\n"
                         memory += f"Action: {previous_action}\n"
                    ad_action_prompt = "Admissible actions: " + ', '.join(admissible_action)
                    prompt = f"Task: '{observations[0]}'\n{memory}\nCurrent observation: {observation}\n{ad_action_prompt}"
                    response = action
                    conversation = []
                    conversation.append({
                        'role': 'system',
                        'content': ALFWORLD_PROMPT + f"\n\n{help_text}"
                    })
                    conversation.append({
                        'role': 'user',
                        'content': prompt
                    })
                    conversations.append(conversation)
                    responses.append(response)
                    choices.append(admissible_action)
        return Dataset.from_dict(mapping=dict(conversations=conversations, responses=responses, choices=choices))


dataset = get_rl_dataset()
dataset.save_to_disk(dataset_path='./datasets/alphahome-rl')