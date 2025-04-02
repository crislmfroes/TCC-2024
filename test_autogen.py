import alfworld.agents.environment as environment
import alfworld.agents.modules.generic as generic
import tqdm
from PIL import Image
from autogen.agentchat import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent
from autogen.agentchat.contrib.capabilities.teachability import Teachability
import difflib
import uuid
from copy import copy
from typing import List
from autogen.coding import CodeBlock, CodeExecutor, CodeExtractor, CodeResult, MarkdownCodeExtractor
from pydantic import BaseModel
from typing import Optional, Literal
import random
import json

random.seed(123)

class Object(BaseModel):
    name: str
    is_hot: bool
    is_cold: bool
    is_clean: bool

class Location(BaseModel):
    name: str
    objects: List[Object]
    can_heat: bool
    can_cool: bool
    can_clean: bool
    is_open: bool

class WorldState(BaseModel):
    locations: List[Location]
    current_location: str
    inventory: str

available_skills = [
    'go to furniture',
    'pick up object',
    'place down object',
    'open furniture',
    'close furniture',
    'use lamp',
    'clean object with sinkbasin',
    'heat object with microwave',
    'cool object with fridge'
]

# NOTE: this ReAct prompt is adapted from Langchain's ReAct agent: https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/agents/react/agent.py#L79
ReAct_prompt = """
Answer the following questions as best you can. You have access to tools provided.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take
Action Input: the input to the action
Observation: the result of the action
... (this process can repeat multiple times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
Question: {input}
"""

# Define the ReAct prompt message. Assuming a "question" field is present in the context


def react_prompt_message(sender, recipient, context):
    return ReAct_prompt.format(input=context["question"])

def camera_capture_func(recipient, messages, sender, config):
    global obs
    global info
    global env
    img_path = f'/tmp/{str(uuid.uuid4())}.jpg'
    img = Image.fromarray(env.get_frames()[0][:,:,::-1])
    img.save(img_path)
    return True, f"Image Observation: <img {img_path}>"

perception_config = {
    'config_list': [
        {
            'model': 'llama3.2-vision',
            'api_type': 'openai',
            'base_url': 'http://0.0.0.0:11434/v1'
        }
    ]
}

planning_config = {
    'config_list': [
        {
            'model': 'marco-o1',
            'api_type': 'openai',
            'base_url': 'http://0.0.0.0:11434/v1',
            'temperature': 0
        }
    ]
}

action_execution_config = {
    'config_list': [
        {
            'model': 'llama3.1',
            'api_type': 'ollama',
            'temperature': 0
        }
    ]
}

orchestrator_config = {
    'config_list': [
        {
            'model': 'llama3.1',
            'api_type': 'ollama'
        }
    ]
}

vlm_config = {
    'config_list': [
        {
            'model': 'llama3.2-vision',
            #'model': 'qwen2.5-coder',
            'api_type': 'openai',
            'base_url': 'http://0.0.0.0:11434/v1',
            #'temperature': 0
        }
    ]
}

llm_config = {
    'config_list': [
        {
            'model': 'qwen2.5',
            'api_type': 'ollama',
            #'base_url': 'http://0.0.0.0:11434/v1',
            'temperature': 0
        }
    ]
}

class PDDLExecutor(CodeExecutor):

    @property
    def code_extractor(self) -> CodeExtractor:
        return MarkdownCodeExtractor()
    
    def execute_code_blocks(self, code_blocks: List[CodeBlock]) -> CodeResult:
        log = ""
        exitcode = 0
        for code_block in code_blocks:
            if code_block.language == 'pddl':
                try:
                    result = plan(pddl_problem_file_content=code_block.code)
                    log += result
                    exitcode = 0
                except BaseException as e:
                    log += f"\n{e}"
                    exitcode = 1
        return CodeResult(exit_code=exitcode, output=log)
            

camera_reader = AssistantAgent(name='camera reader', llm_config=False)

perception_system = MultimodalConversableAgent('perception system', system_message="You are the perception system of a general purpose service robot. You always answer by parsing the contents of the observation into a structured format", llm_config=perception_config)

planning_system = AssistantAgent(name='planning system', system_message=f'You are the task planning system of a general purpose service robot\nUse the fridge for cooling objects\nUse the microwave for heating objects\nUse the sink for cleaning objects', llm_config=planning_config)

planner_user = UserProxyAgent(name='planner user', human_input_mode='NEVER', code_execution_config=False)

navigation_system = AssistantAgent(name='navigation system', system_message='You are the navigation system of a general purpose service robot.', llm_config=llm_config)

manipulation_system = AssistantAgent(name='manipulation system', system_message='You are the manipulation system of a general purpose service robot.', llm_config=llm_config)

robot = AssistantAgent(name='robot', system_message=f"You are a general purpose service robot.\nWhen given a new task, reason through it step-by-step, and then execute your next action with a tool call.\n\nExample:\n\nUser: ...(some task)\nYou (calling 'execute_action' function): {json.dumps({'explanation': 'My task is to ..., therefore my plan is to ..., reasoning about the available actions, the most promising one it $action, I will now execute the action $action and then ...', 'next_action': '$action'})}\nUser: ...(environment feedback)\nYou (calling 'execute_action' function): {{...}}\n...", llm_config=llm_config)

critic = AssistantAgent(name='critic', system_message=f'You are an expert robot critic. Your expertise lies in the ability to provide constructive feedback to a service robot executing tasks inbside a house', llm_config=vlm_config)

exploration_system = AssistantAgent(name='exploration system', system_message='You are the environment exploration system of a general purpose service robot. You are an expert at suggesting locations in the house to explore in order to complete user given tasks', llm_config=llm_config)

def check_terminate(msg):
    global done
    print('Done: ', done)
    return done == True

user_proxy = UserProxyAgent('user proxy', human_input_mode='NEVER', is_termination_msg=check_terminate, code_execution_config=False, llm_config=False)

'''#@user_proxy.register_for_execution()
#@robot.register_for_llm(description='write your step-by-step plan here before doing anything else')
def plan(plan: str):
    return plan'''

'''#@user_proxy.register_for_execution()
#@robot.register_for_llm(description='check your current inventory')
def check_inventory():
    global env
    global info
    global done
    global counter
    obs, score, done, info = env.step(['inventory'])
    return obs[0]'''

#@user_proxy.register_for_execution()
#@robot.register_for_llm(description='go to some furniture')
def go_to(furniture: str):
    global env
    global info
    global done
    global counter
    choices = difflib.get_close_matches(f'go to {furniture}', info['admissible_commands'][0])
    if len(choices) == 0:
        return f'Invalid destination {furniture}'
    else:
        obs, score, done, info = env.step([choices[0]])
        counter += 1
        #opening_actions = [action for action in info['admissible_commands'][0] if action.startswith('open ')]
        #obs, score, done, info = env.step([opening_actions[0]])
        #counter += 1
        done = done[0]
        return obs[0]

#@user_proxy.register_for_execution()
#@robot.register_for_llm(description='pick object from furniture')    
def pick_up(object: str, furniture: str):
    global env
    global info
    global done
    global counter
    go_to(furniture=furniture)
    choices = difflib.get_close_matches(f'take {object} from {furniture}', info['admissible_commands'][0])
    obs, score, done, info = env.step([choices[0]])
    counter += 1
    done = done[0]
    return obs[0]

#@user_proxy.register_for_execution()
#@robot.register_for_llm(description='place object into furniture')
def place_down(object: str, furniture: str):
    global env
    global info
    global done
    global counter
    pick_up(object=object)
    go_to(furniture=furniture)
    choices = difflib.get_close_matches(f'put {object} in/on {furniture}', info['admissible_commands'][0])
    obs, score, done, info = env.step([choices[0]])
    counter += 1
    done = done[0]
    return obs[0]

#@user_proxy.register_for_execution()
#@robot.register_for_llm(description='open cabinet or drawer')    
def open_furniture(furniture: str):
    global env
    global info
    global done
    global counter
    go_to(furniture=furniture)
    choices = difflib.get_close_matches(f'open {furniture}', info['admissible_commands'][0])
    obs, score, done, info = env.step([choices[0]])
    counter += 1
    done = done[0]
    return obs[0]

#@user_proxy.register_for_execution()
#@robot.register_for_llm(description='close cabinet or drawer')
def close(furniture: str):
    global env
    global info
    global done
    global counter
    go_to(furniture=furniture)
    choices = difflib.get_close_matches(f'close {furniture}', info['admissible_commands'][0])
    obs, score, done, info = env.step([choices[0]])
    counter += 1
    done = done[0]
    return obs[0]

#@user_proxy.register_for_execution()
#@robot.register_for_llm(description='use some lamp')    
def use(object: str):
    global env
    global info
    global done
    global counter
    choices = [f'use {object}']
    obs, score, done, info = env.step([choices[0]])
    counter += 1
    done = done[0]
    return obs[0]

#@user_proxy.register_for_execution()
#@robot.register_for_llm(description='clean some object on the sinkbasin')
def clean(object: str, sinkbasin: str):
    global env
    global info
    global done
    global counter
    pick_up(object=object)
    go_to(furniture=sinkbasin)
    choices = difflib.get_close_matches(f'clean {object} with {sinkbasin}', info['admissible_commands'][0])
    obs, score, done, info = env.step([choices[0]])
    counter += 1
    done = done[0]
    return obs[0]

#@user_proxy.register_for_execution()
#@robot.register_for_llm(description='heat some object on the microwave')    
def heat(object: str, microwave: str):
    global env
    global info
    global done
    global counter
    pick_up(object=object)
    go_to(furniture=microwave)
    choices = difflib.get_close_matches(f'heat {object} with {microwave}', info['admissible_commands'][0])
    obs, score, done, info = env.step([choices[0]])
    counter += 1
    done = done[0]
    return obs[0]

#@user_proxy.register_for_execution()
#@robot.register_for_llm(description='cool some object on the fridge')
def cool(object: str, fridge: str):
    global env
    global info
    global done
    global counter
    pick_up(object=object)
    go_to(furniture=fridge)
    choices = difflib.get_close_matches(f'cool {object} with {fridge}', info['admissible_commands'][0])
    obs, score, done, info = env.step([choices[0]])
    counter += 1
    done = done[0]
    return obs[0]

##@user_proxy.register_for_execution()
##@robot.register_for_llm(description='query a planner for a step-by-step plan')
def plan(query: str):
    return planner_user.initiate_chat(recipient=planning_system, message=query, max_turns=1).summary

#@user_proxy.register_for_execution()
#@robot.register_for_llm(description='Think step-by-step before executing any action')
def think(thought: str):
    return thought

@user_proxy.register_for_execution()
@robot.register_for_llm(description='Execute some action in the environment')
def execute_action(explanation: str, next_action: str):
    global env
    global info
    global done
    global counter
    obs, score, done, info = env.step([next_action])
    counter += 1
    done = done[0]
    return f"Observation: {obs[0]}\n\nAdmissible Actions: {info['admissible_commands'][0]}\n\nTask Completed: {done}"

#@planner_user.register_for_execution()
#@planning_system.register_for_llm(description='execute a plan step-by-step. Each step is a string')
def execute_plan_steps(steps: List[str]):
    outcomes = []
    for step in steps:
        outcomes += [user_proxy.initiate_chat(recipient=robot, message=step, max_turns=1).summary]
    return outcomes

#@user_proxy.register_for_execution()
#@robot.register_for_llm(description='put a specified amount of objects, with the given state, on the given piece of furniture')
def put_object_on_furniture(object: str, furniture: str, object_count: int=1, object_must_be_hot: bool=False, object_must_be_cold: bool=False, object_must_be_clean: bool=False):
    global env
    global info
    global done
    global counter
    if done == True:
        return f'TASK COMPLETED: {done}'
    object_count = int(object_count)
    result = []
    for n in range(object_count):
        for action0 in copy(info['admissible_commands'][0]):
            if action0.startswith('go to'):
                obs, score, done, info = env.step([action0])
                counter += 1
                done = done[0]
                for action1 in copy(info['admissible_commands'][0]):
                    if action1.startswith('open'):
                        obs, score, done, info = env.step([action1])
                        counter += 1
                        done = done[0]
                    for action2 in copy(info['admissible_commands'][0]):
                        if action2.startswith(f'take {object}'):
                            obs, score, done, info = env.step([action2])
                            counter += 1
                            done = done[0]
                            if object_must_be_hot:
                                obs, score, done, info = env.step(['go to microwave 1'])
                                counter += 1
                                done = done[0]
                                obs, score, done, info = env.step(['open microwave 1'])
                                counter += 1
                                done = done[0]
                                obs, score, done, info = env.step([difflib.get_close_matches(f'heat {object} with microwave', info['admissible_commands'][0])[0]])
                                counter += 1
                                done = done[0]
                            if object_must_be_cold:
                                obs, score, done, info = env.step(['go to fridge 1'])
                                counter += 1
                                done = done[0]
                                obs, score, done, info = env.step(['open fridge 1'])
                                counter += 1
                                done = done[0]
                                obs, score, done, info = env.step([difflib.get_close_matches(f'cool {object} with fridge', info['admissible_commands'][0])[0]])
                                counter += 1
                                done = done[0]
                            if object_must_be_clean:
                                obs, score, done, info = env.step(['go to sinkbasin 1'])
                                counter += 1
                                done = done[0]
                                obs, score, done, info = env.step([difflib.get_close_matches(f'clean {object} with sinkbasin', info['admissible_commands'][0])[0]])
                                counter += 1
                                done = done[0]
                            for action3 in copy(info['admissible_commands'][0]):
                                if action3.startswith(f'go to {furniture}'):
                                    obs, score, done, info = env.step([action3])
                                    counter += 1
                                    done = done[0]
                                    for action4 in copy(info['admissible_commands'][0]):
                                        if action4.startswith(f'open {furniture}'):
                                            obs, score, done, info = env.step([action4])
                                            counter += 1
                                            done = done[0]
                                        for action5 in copy(info['admissible_commands'][0]):
                                            if action5.startswith(f'put {object}'):
                                                obs, score, done, info = env.step([action5])
                                                counter += 1
                                                done = done[0]
                                                result += [obs[0]]
    return result

groupchat = GroupChat(agents=[user_proxy, robot], select_speaker_auto_llm_config=orchestrator_config, messages=[], max_round=5, allowed_or_disallowed_speaker_transitions={
    user_proxy: [robot],
    robot: [user_proxy]
}, speaker_transitions_type='allowed')
#groupchat = GroupChat(agents=[user_proxy, action_execution_system], select_speaker_auto_llm_config=orchestrator_config, messages=[], speaker_selection_method='round_robin', max_round=200)

#groupchat = GroupChat(agents=[user_proxy, camera_reader, robot], messages=[], speaker_selection_method='round_robin', max_round=50)
#groupchat._raise_exception_on_async_reply_functions = lambda **kwargs: print(kwargs)

groupchat_manager = GroupChatManager(groupchat=groupchat)

camera_reader.register_reply(trigger=groupchat_manager, reply_func=camera_capture_func)

config = generic.load_config()
env_type = config['env']['type'] # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'

# setup environment
env = getattr(environment, env_type)(config, train_eval='eval_out_of_distribution')
env = env.init_env(batch_size=1)
success = 0
total = 0
for i in tqdm.trange(10):
    done = False
    counter = 0
    obs, info = env.reset()
    user_proxy.initiate_chat(recipient=groupchat_manager, message=obs[0]+f'\n\nAdmissible actions: {info["admissible_commands"][0]}', max_turns=50)
    if done == True and counter < 50:
        success += 1
    total += 1
    print('success: ', success)
    print('total: ', total)  