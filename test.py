import os
import sys
sys.path.append(os.environ['ALFRED_ROOT'])
sys.path.append(os.environ['ALFRED_ROOT']+'/gen')
from env.thor_env import ThorEnv
import json
import time
import tqdm
from ultralytics import YOLOWorld, SAM
from pydantic import BaseModel, Field
from typing import Optional, Literal
import outlines
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoModel

SPLIT = 'tests_unseen'

class Action(BaseModel):
    #action_type: Literal["RotateLeft","RotateRight","MoveAhead"] = Field(description="The type of action to take")
    action_type: Literal["RotateLeft","RotateRight","MoveAhead","LookUp","LookDown","OpenObject","CloseObject","PickupObject","PutObject","ToggleObjectOn","ToggleObjectOff","SliceObject","Terminate"] = Field(description="The type of action to take")
    action_parameters: Optional[str] = Field(description="The target object upon which to take the action, if any.")

with open(os.environ['ALFRED_ROOT']+'/data/splits/oct21.json', 'r') as f:
    dataset = json.load(f)
env = ThorEnv()
#yolo_world = YOLOWorld()
#sam = SAM()
mllm = outlines.models.transformers_vision(
    "Qwen/Qwen2-VL-2B-Instruct",
    #"OpenGVLab/InternVL2-1B",
    model_class=Qwen2VLForConditionalGeneration,
    #model_class=AutoModel,
    device="auto",
    model_kwargs=dict(load_in_4bit=True, bnb_4bit_use_double_quant=True)
)
policy = outlines.generate.json(mllm, Action)
for i in tqdm.trange(len(dataset[SPLIT])):
    task = dataset[SPLIT][i]['task']
    with open(os.environ['ALFRED_ROOT']+'/data/json_2.1.0/'+SPLIT+'/'+task+'/traj_data.json', 'r') as f:
        traj_data = json.load(f)
    scene_num = traj_data['scene']['scene_num']
    object_poses = traj_data['scene']['object_poses']
    dirty_and_empty = traj_data['scene']['dirty_and_empty']
    object_toggles = traj_data['scene']['object_toggles']
    scene_name = 'FloorPlan%d' % scene_num
    env.reset(scene_name)
    env.restore_scene(object_poses, object_toggles, dirty_and_empty)
    goal = traj_data['turk_annotations']['anns'][dataset[SPLIT][i]['repeat_idx']]['task_desc']
    print("Task: %s" % goal)
    previous_actions = []
    for j in range(10):
        prompt = f"Goal: {goal}\nAction Schema: {Action.model_json_schema()}\nPrevious actions: {previous_actions}\nObservation: <image>\nNext Action:"
        action: Action = policy(prompt, [Image.fromarray(env.last_event.frame)])
        print(action)
        env.va_interact(action=action.action_type)
        previous_actions.append(action)
    break
    
    