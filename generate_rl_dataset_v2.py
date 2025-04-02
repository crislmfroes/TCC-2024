import tqdm
from datasets import Dataset
from alfworld.agents.environment.alfred_thor_env import AlfredThorEnv
import alfworld.agents.modules.generic as generic
from PIL import Image

config = generic.load_config()
env_type = config['env']['type']
env = AlfredThorEnv(config, train_eval='train')
env: AlfredThorEnv = env.init_env(batch_size=1)
_, info = env.reset()
obs, _, _, _ = env.step(["help"])
help_text = ""

ALFWORLD_PROMPT = """You are a service robot inside a domestic environment.
You must reason inside <think> tags, and then choose an action inside <action> tags.
Example:

<think>some reasoning</think><action>some action</action>

Available Actions:

- go to (location): navigate to some location in the room
- take (object) from (location): takes an object from some location
- put (object) in/on (location): puts a previously taken object at some location
- open (location): open some furniture
- close (location): close some furniture
- heat (object) with (location): heats some object by using an appropriate furniture
- cool (object) with (location): cools some object by using an appropriate furniture
- clean (object) with (location): cleans some object by using an appropriate furniture
- examine (object): examine the physical state of some object
- use (object): use some object
- inventory: reveals the object you are currently carrying
- look: look at your surroundings
"""

def get_rl_dataset(**kwargs):
        conversations = []
        responses = []
        choices = []
        dataset_images = []
        for i in tqdm.trange(4000):
            obs, info = env.reset()
            observation = obs[0]
            done = False
            steps = 0
            actions = []
            observations = []
            admissible_actions = []
            images = []
            while not done:
                observations.append(observation)
                img = Image.fromarray(env.get_frames()[0][:,:,::-1])
                images.append(img)
                admissible_actions.append(info['admissible_commands'][0])
                action = info['extra.expert_plan'][0][0]
                actions.append(action)
                obs, reward, done, info = env.step([action])
                observation =  obs[0]
                done = done[0]
                steps += 1
            #print(info['won'])
            if steps < 50:
                for j, (action, observation, admissible_action, image) in enumerate(zip(actions, observations, admissible_actions, images)):
                    conversation = []
                    conversation.append({
                        'role': 'system',
                        'content': [{'type': 'text', 'text': ALFWORLD_PROMPT}]
                    })
                    for previous_action, previous_obs in zip(actions[:j], observations[:j]):
                         conversation.append({
                              'role': 'user',
                              'content': [{'type': 'text', 'text': previous_obs}]
                         })
                         conversation.append({
                              'role': 'assistant',
                              'content': [{'type': 'text', 'text': f"<action>{previous_action}</action>"}]
                         })
                    #ad_action_prompt = "Admissible actions: " + ', '.join(admissible_action)
                    prompt = observation#\n{ad_action_prompt}"
                    response = action
                    conversation.append({
                        'role': 'user',
                        'content': [{'type': 'text', 'text': prompt}, {'type': 'image'}]
                    })
                    conversations.append(conversation)
                    responses.append(response)
                    choices.append(admissible_action)
                    dataset_images.append(image)
        return Dataset.from_dict(mapping=dict(conversations=conversations, responses=responses, choices=choices, image=dataset_images))


dataset = get_rl_dataset()
dataset.save_to_disk(dataset_path='./datasets/alphahome-rl-vl')