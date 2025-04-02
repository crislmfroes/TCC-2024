from open_r1.trainer.grpo_trainer import Qwen2VLGRPOTrainer
import torch
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url
from transformers import PreTrainedModel, Qwen2_5_VLForConditionalGeneration
from alfworld.agents.environment.alfred_thor_env import AlfredThorEnv
from PIL import Image
from datasets import Dataset, DatasetDict


class Qwen2VLAlfredGRPOTrainer(Qwen2VLGRPOTrainer):
    def __init__(self, model, args = None, train_env=None, processing_class = None, reward_processing_classes = None, callbacks = None, peft_config = None, max_pixels = 12845056, min_pixels = 3136, attn_implementation = "flash_attention_2"):
        super().__init__(model, reward_funcs=[self.step_environment], args=args, train_dataset=Dataset.from_dict(dict(prompt=['dummy',]*50*4000)), eval_dataset=None, processing_class=processing_class, reward_processing_classes=reward_processing_classes, callbacks=callbacks, peft_config=peft_config, max_pixels=max_pixels, min_pixels=min_pixels, attn_implementation=attn_implementation)
        self.train_env: AlfredThorEnv = train_env
        self.train_info = []
        self.train_obs = []
        self.train_actions = []
        self.train_thoughts = []
        self.train_done = []
        self.help_text = self.compute_help_text()
        self.reset_environment()

    def compute_help_text(self):
        _, _ = self.train_env.reset()
        obs, _, _, _ = self.train_env.step(['help'])
        return obs[0]

    def get_train_prompt(self):
        prompt = []
        prompt.append({
            'role': 'system',
            'content': [{'type': 'text', 'text': 'You are a general purpose service robot inside a house.\nTo execute some action, reason inside <think> and </think> tags, and answer with the action enclosed in <action> and </action> tags.\nExample:\n\n<think>some reasoning</think><action>some action</action>'}]
        })
        for obs, thought, action in zip(self.train_obs[:-1], self.train_thoughts, self.train_actions):
            prompt.append({
                'role': 'user',
                'content': [{'type': 'text', 'text': obs[0]}]
            })
            prompt.append({
                'role': 'assistant',
                'content': [{'type': 'text', 'text': f"<think>{thought}</think><action>{action}</action>"}]
            })
        prompt.append({
            'role': 'user',
            'content': [
                {'type': 'text', 'text': self.train_obs[-1][0] + "\n\nAvailable actions: " + ', '.join(self.train_info[-1]['admissible_commands'][0])},
                {'type': 'image'}
            ]
        })
        return prompt
    
    def step_environment(self, completions, **kwargs):
        assert len(completions) == 1
        completion = completions[0][0]['content']
        #print(completion)
        reward = 0.0
        warning = ''
        if '<action>' in completion and '</action>' in completion:
            reward += 0.1
            action = completion.split('<action>')[1].split('</action>')[0].strip()
        else:
            action = completion
        if '<think>' in completion and '</think>' in completion:
            reward += 0.1
            thought = completion.split('<think>')[1].split('</think>')[0].strip()
        else:
            thought = completion
        print('Thought:', thought)
        print('Action:', action)
        if action in self.train_info[-1]['admissible_commands'][0]:
            reward += 0.1
        else:
            warning += f"\n\nInvalid action: {action}!"
        if action == self.train_info[-1]['extra.expert_plan'][0][0]:
            reward += 0.2
        obs, score, done, info = self.train_env.step([action,])
        obs[0] += warning
        if info['won'][0] == True:
            reward += 1.0
        if done[0] == True or len(self.train_obs) > 10:
            self.reset_environment()
        else:
            self.train_obs.append(obs)
            self.train_thoughts.append(thought)
            self.train_actions.append(action)
            self.train_info.append(info)
            self.train_done.append(done)
        #print(reward)
        return [reward,]

    def check_env_done(self):
        return self.train_done[-1][0] == True
    
    def check_env_success(self):
        return self.train_info[-1]['won'][0] == True
    
    def reset_environment(self):
        self.train_obs = []
        self.train_info = []
        self.train_done = []
        self.train_actions = []
        self.train_thoughts = []
        obs, info = self.train_env.reset()
        self.train_obs.append(obs)
        self.train_info.append(info)
        self.train_done.append([False,])

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):    
        assert len(inputs) == 1
        new_inputs = Dataset.from_dict({'prompt': [self.get_train_prompt()], 'image': [Image.fromarray(self.train_env.get_frames()[0][:,:,::-1])]})
        return super(Qwen2VLAlfredGRPOTrainer, self).compute_loss(model, new_inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)