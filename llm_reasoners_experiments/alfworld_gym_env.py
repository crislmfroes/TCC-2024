import gymnasium as gym
from typing import NamedTuple, Optional, Callable, Any
#from reasoners import Environment
from reasoners import WorldModel, LanguageModel
from utils.prompts import build_update_prompt
from utils.response_models import UpdateResponseModel
import copy
import re
import json

ActionGym = Any


class StateGym(NamedTuple):
    step_idx: int
    # action history used to reconstruct the env state for backtracking
    action_history: list[ActionGym]
    observation_history: list[dict]
    # gym observation objects
    last_obs: dict
    # outputs from env.step()
    current_obs: dict
    reward: float
    done: bool
    action_set: list[str]

class AlfworldWorldModel(WorldModel):
    def __init__(self, llm: LanguageModel, max_steps: int=50):
        super().__init__()
        self.max_steps = max_steps
        self.llm = llm

    def init_state(self):
        example = self.example[0]
        return StateGym(
            step_idx=example['step_idx'],
            action_history=example['action_history'],
            observation_history=example['observation_history'],
            last_obs=example['last_obs'],
            current_obs=example['current_obs'],
            reward=example['reward'],
            done=example['done'],
            action_set=example['action_set']
        )
    
    def step(self, state: StateGym, action: ActionGym):
        state = copy.deepcopy(state)
        step_idx = state.step_idx
        #print(state)
        system_msgs, user_msgs, full_prompt_text = build_update_prompt(obs=state.current_obs, action=action['action'], action_set=state.action_set, action_history=state.action_history, observation_history=state.observation_history, use_text_observation=True, use_vision_observation=False)
        response = self.llm.generate(full_prompt_text, reponse_model=UpdateResponseModel)
        observation_proposal = response.text[0]
        try:
            json_string = re.search(r"\{.*\}", observation_proposal, re.DOTALL).group()
            json_object = json.loads(json_string)
            next_text_observation = json_object['observation']
            task_done = json_object['task_done']
            next_action_set = json_object['available_actions']
        except:
            next_text_observation = 'Nothing happens.'
            task_done = False
            next_action_set = state.action_set
        reward = 1.0 if task_done == True else 0.0
        return StateGym(
            step_idx=step_idx+1,
            action_history=state.action_history+[action,],
            observation_history=state.observation_history+[state.current_obs,],
            last_obs=state.current_obs,
            current_obs={
                'text_observation': next_text_observation,
                'task': state.current_obs['task'],
                'last_action': action['action'],
                'last_action_error': None,
                'action_set': next_action_set
            },
            reward=reward,
            done=False,
            action_set=next_action_set
        ), {'env_reward': reward}
    
    def is_terminal(self, state: StateGym) -> bool:
        return state.done


class AlfworldGym:
    """
    WorldModel, but for gym environments. Instead of being based off of a textual example, takes in a gym environment. An LLM will not be used for generating new states. The gym environment's step function takes care of that. 

    Attributes:
    - env (gym.Env): the gym environment
    - env_seed (int): the seed for the gym environment
    - max_steps (int): the maximum number of steps that can be taken until is_terminal cuts off the episode
    - obs_preprocessor (Optional[Callable[[dict], dict]]): optional function to process the observation returned from resetting/stepping the environment before it is stored into the state tuple
    - env_current_obs (dict): the current observation of the environment which is used to check if a passed in state is aligned with the environment's current state
    """

    def __init__(self, env: gym.Env, env_task: str, max_steps=50, obs_preprocessor: Optional[Callable[[dict], dict]] = None):
        self.env = env
        self.env_task = env_task
        self.obs_preprocessor = obs_preprocessor
        self.max_steps = max_steps
        self.env_current_obs: dict = None

    def init_state(self) -> StateGym:
        obs, env_info = self.env.reset(
            task=self.env_task)
        task = obs[0].split('\n\n')[-1]
        obs = {
            'text_observation': obs[0],
            'vision_observation': self.env.get_frames()[0][:,:,::-1],
            'task': task,
            'last_action': '',
            'last_action_error': None,
            'action_set': env_info['admissible_commands'][0]
        }
        if self.obs_preprocessor is not None:
            obs = self.obs_preprocessor(obs)
        self.env_current_obs = obs

        return StateGym(step_idx=0, last_obs={}, current_obs=obs, action_history=[], observation_history=[], reward=0, done=False, action_set=env_info['admissible_commands'][0])

    def step(self, state: StateGym, action: ActionGym) -> tuple[StateGym, dict]:
        """
        Takes in a state and action and steps the environment. Should be noted that the environment may not be aligned with the state passed in. If the environment's current state (self.env_current_obs) is not the same as the state passed in, backtracking is needed. The basic implementation of this is rather naive, as it just resets the environment and replays the actions in the state's action_history list. Depending on the environment, there may be far more efficient ways to do so. 

        Args:
        - state (StateGym): the state to step from
        - action (ActionGym): the action to take from the state

        Returns:
        - next_state (StateGym): the next state after taking the action
        - aux (dict): used to pass the environment's reward to the search algorithm, which then passes it to the SearchConfig's reward function
        """

        if self.env_current_obs != state.current_obs:
            self.env.reset(task=self.env_task)
            for action in state.action_history:
                self.env.step([action['action'],])
        observation_history = state.observation_history.copy()
        observation_history.append(state.current_obs)
        obs, reward, done, step_info = self.env.step([
            action['action'],])
        done = done[0]
        obs = {
            'text_observation': obs[0],
            'vision_observation': self.env.get_frames()[0][:,:,::-1],
            'task': state.current_obs['task'],
            'last_action': action,
            'last_action_error': None,
            'action_set': step_info['admissible_commands'][0]
        }
        reward = 1.0 if done and state.step_idx + 1 < self.max_steps else 0.0
        if self.obs_preprocessor is not None:
            obs = self.obs_preprocessor(obs)
        self.env_current_obs = obs

        next_state = StateGym(step_idx=state.step_idx + 1,
                              last_obs=state.current_obs, current_obs=obs,
                              action_history=state.action_history +
                              [action],
                              observation_history=observation_history,
                              reward=reward, done=done, action_set=step_info['admissible_commands'][0])

        return next_state, {"env_reward": reward}

    def is_terminal(self, state: StateGym) -> bool:
        return state.done or state.step_idx >= self.max_steps
    
    def update_example(self, example: Any, prompt: str=None):
        pass
