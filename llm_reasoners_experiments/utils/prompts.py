"""
Referencing prompt building from. 
https://github.com/ServiceNow/BrowserGym/blob/main/demo_agent/agent.py
"""

import logging
import os
import re
import json
import argparse
import base64
import io
import numpy as np
from PIL import Image

from reasoners import SearchConfig, WorldModel, LanguageModel
from .misc import image_to_jpg_base64_url


def get_user_messages_for_current_state(
    obs: dict,
    action_set: list[str], action_history: list[str], observation_history: list[dict],
    use_text_observation: bool = True, use_vision_observation: bool = False
) -> list[dict]:
    assert obs["task"], "The task is missing."
    user_msgs = []

    user_msgs.append(
        {
            'type': 'text',
            'text': f"""# Your task is: {obs['task']}"""
        }
    )

    user_msgs.append({
        'type': 'text',
        'text': f'# Previous observations: '
    })

    for previous_observation in observation_history:
        #print(previous_observation)
        user_msgs.append({
            'type': 'text',
            'text': f'\n{previous_observation["text_observation"]}\nAction space: {previous_observation["action_set"]}'
        })

    # append page AXTree (if asked)
    if use_text_observation:
        user_msgs.append(
            {
                "type": "text",
                "text": f"""\
# Current environment observation

{obs["text_observation"]}
""",
            }
        )
    '''# append page HTML (if asked)
    if use_html:
        user_msgs.append(
            {
                "type": "text",
                "text": f"""\
# Current page DOM

{obs["pruned_html"]}

""",
            }
        )'''

    # append page screenshot (if asked)
    if use_vision_observation:
        user_msgs.append(
            {
                "type": "text",
                "text": """\
# Current environment observation
    """,
            }
        )
        user_msgs.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_to_jpg_base64_url(obs["vision_observation"]),
                    "detail": "auto",
                },  # Literal["low", "high", "auto"] = "auto"
            }
        )

    # append action space description
    user_msgs.append(
        {
            "type": "text",
            "text": f"""\
# Action Space

{action_set}

Here are examples of actions with chain-of-thought reasoning:

```json
{json.dumps({
    "thought": "My task is to put a hot apple in fridge. I have systematically searched for an apple, and found it on countertop 1. I now need to pick up the apple from the contertop in order to heat it in the microwave. I will use the 'take apple 1 from countertop 1' to pick up the apple, before navigating to the microwave with the 'go to microwave 1' action in order to heat it with the 'heat apple 1 with microwave 1' action. After heating the apple, I will navigate to the fridge with the 'go to fridge 1' action, open the fridge door with 'open fridge 1' and finally 'put apple 1 in/on fridge 1'",
    "action": "take apple 1 from countertop 1"
})}
```

""",
        }
    )

    # append past actions (and last error message) if any
    if action_history:
        user_msgs.append(
            {
                "type": "text",
                "text": f"""\
# History of past actions
""",
            }
        )
        user_msgs.extend(
            [
                {
                    "type": "text",
                    "text": f"""\

{json.dumps(action)}
""",
                }
                for action in action_history
            ]
        )

        if obs["last_action_error"]:
            user_msgs.append(
                {
                    "type": "text",
                    "text": f"""\
# Error message from last action

{obs["last_action_error"]}

""",
                }
            )

    return user_msgs


def build_propose_prompt(
    obs: dict,
    action_set: list[str], action_history: list[str], observation_history: list[dict],
    use_text_observation: bool = True, use_vision_observation: bool = False
    # logger: logging.Logger = None
) -> tuple[list[dict], list[dict], str]:
    system_msgs = []
    user_msgs = []

    assert obs["task"], "The task is missing."
    system_msgs.append(
        {
            "type": "text",
            "text": f"""\
# Instructions

Review the current state of the environment and all other information to find the best
possible next action to accomplish your task. Your answer will be interpreted
and executed by a program, make sure to follow the formatting instructions.
""",
        }
    )

    user_msgs.extend(get_user_messages_for_current_state(
        obs, action_set, action_history, observation_history, use_text_observation, use_vision_observation))

    # ask for the next action
    user_msgs.append(
        {
            "type": "text",
            "text": f"""\
# Next action

You will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, and the current state of the environment before deciding on your next action. Make sure to fill in ALL PARAMETERS of the action. 
""",
        }
    )

    prompt_text_strings = []
    for message in system_msgs + user_msgs:
        match message["type"]:
            case "text":
                prompt_text_strings.append(message["text"])
            case "image_url":
                image_url = message["image_url"]
                if isinstance(message["image_url"], dict):
                    image_url = image_url["url"]
                if image_url.startswith("data:image"):
                    prompt_text_strings.append(
                        "image_url: " + image_url[:30] + "... (truncated)"
                    )
                else:
                    prompt_text_strings.append("image_url: " + image_url)
            case _:
                raise ValueError(
                    f"Unknown message type {repr(message['type'])} in the task goal."
                )
    full_prompt_txt = "\n".join(prompt_text_strings)

    return system_msgs, user_msgs, full_prompt_txt


def build_evaluation_prompt(
    obs: dict,
    action: str, action_set: list[str], action_history: list[str], observation_history: list[dict],
    use_text_observation: bool = True, use_vision_observation: bool = False,
    # logger: logging.Logger = None
) -> tuple[list[dict], list[dict], str]:
    system_msgs = []
    user_msgs = []

    assert obs["task"], "The task is missing."
    system_msgs.append(
        {
            "type": "text",
            "text": """\
# Instructions

Review the current state of the environment along with a proposed action and determine how promising it is towards completing the goal. Provide a score between 0 and 10 along with your reasoning in a json object like so:
{
    "reasoning": [your_reasoning]
    "score": [your_score]
}
""",
        }
    )

    user_msgs.extend(get_user_messages_for_current_state(
        obs, action_set, action_history, observation_history, use_text_observation, use_vision_observation))

    # proposed action
    user_msgs.append(
        {
            "type": "text",
            "text": f"""\
# Proposed action

{action}
""",
        }
    )

    # ask for the evaluation
    user_msgs.append(
        {
            "type": "text",
            "text": """\
# Evaluation of Proposed Action

As mentioned before, considering all the information above in the context of the goal, evaluate the proposed action by providing a score from 0 to 10 along with your reasoning. Use a json object like so:
{
    "reasoning": [your_reasoning]
    "score": [your_score]
}
""",
        }
    )

    prompt_text_strings = []
    for message in system_msgs + user_msgs:
        match message["type"]:
            case "text":
                prompt_text_strings.append(message["text"])
            case "image_url":
                image_url = message["image_url"]
                if isinstance(message["image_url"], dict):
                    image_url = image_url["url"]
                if image_url.startswith("data:image"):
                    prompt_text_strings.append(
                        "image_url: " + image_url[:30] + "... (truncated)"
                    )
                else:
                    prompt_text_strings.append("image_url: " + image_url)
            case _:
                raise ValueError(
                    f"Unknown message type {repr(message['type'])} in the task goal."
                )

    full_prompt_txt = "\n".join(prompt_text_strings)

    return system_msgs, user_msgs, full_prompt_txt

def build_update_prompt(
    obs: dict,
    action: str, action_set: list[str], action_history: list[str], observation_history: list[dict],
    use_text_observation: bool = True, use_vision_observation: bool = False,
    # logger: logging.Logger = None
):
    print(obs.keys())
    system_msgs = []
    user_msgs = []
    assert obs['task'], "The task is missing!"
    system_msgs.append(
        {
            "type": "text",
            "text": """\
# Instructions

Review the current state of the environment along with a proposed action and generate the next observation. Provide the next observation and action space, along with your reasoning and whether the environment task has been completed in a json object like so:
{
    "reasoning": [your_reasoning]
    "observation": [next_observation]
    "available_actions": [next_action_space]
    "task_done": [true or false]
}
""",
        }
    )
    user_msgs.extend(get_user_messages_for_current_state(
        obs, action_set, action_history, observation_history, use_text_observation, use_vision_observation))

    # proposed action
    user_msgs.append(
        {
            "type": "text",
            "text": f"""\
# Proposed action

{action}
""",
        }
    )
    user_msgs.append(
        {
            "type": "text",
            "text": """\
# Simulation of Proposed Action

As mentioned before, considering all the information above in the context of the goal, generate the next observation resulting from executing the proposed action. Use a json object like so:
{
    "reasoning": [your_reasoning]
    "observation": [next_observation]
    "available_actions": [next_action_space]
    "task_done": [true or false]
}
""",
        }
    )
    prompt_text_strings = []
    for message in system_msgs + user_msgs:
        match message["type"]:
            case "text":
                prompt_text_strings.append(message["text"])
            case "image_url":
                image_url = message["image_url"]
                if isinstance(message["image_url"], dict):
                    image_url = image_url["url"]
                if image_url.startswith("data:image"):
                    prompt_text_strings.append(
                        "image_url: " + image_url[:30] + "... (truncated)"
                    )
                else:
                    prompt_text_strings.append("image_url: " + image_url)
            case _:
                raise ValueError(
                    f"Unknown message type {repr(message['type'])} in the task goal."
                )

    full_prompt_txt = "\n".join(prompt_text_strings)

    return system_msgs, user_msgs, full_prompt_txt