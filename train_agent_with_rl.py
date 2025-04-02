from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
from alfworld.agents.environment import get_environment
import alfworld.agents.modules.generic as generic
import json
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset
import verifiers as vf
from verifiers.envs.multistep_env import MultiStepEnv
from verifiers.rubrics.base import BaseRubric
import tqdm
from copy import deepcopy
from collections import deque


def get_default_grpo_config(run_name: str, num_gpus: int = 1) -> GRPOConfig:
    return GRPOConfig(
        output_dir=f"outputs/{run_name}",
        run_name=run_name,
        learning_rate=5e-6,
        warmup_steps=50,
        num_train_epochs=1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        bf16=True,
        adam_beta1=0.9,
        adam_beta2=0.99,
        max_grad_norm=0.1,
        beta=0.04,
        max_prompt_length=512,
        max_completion_length=512,
        per_device_train_batch_size=1,
        num_generations=2,
        gradient_accumulation_steps=int(1 / num_gpus),
        gradient_checkpointing=True,
        save_strategy="epoch",
        save_only_model=True,
        use_vllm=True,
        vllm_device=f"cuda:{num_gpus-1}",
        vllm_gpu_memory_utilization=0.1,
        logging_steps=1,
        log_on_each_node=False,
        report_to="wandb",
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

class AlfworldRubric(BaseRubric):
    def __init__(self, env, **kwargs):
        super().__init__(**kwargs)
        self.env = env

    def check_valid_action(self, completions, **kwargs):
        rewards = []
        for completion in completions:
            reward = 0.0
            if isinstance(completion, dict):
                messages_to_verify = [completion,]
            elif isinstance(completion, list):
                messages_to_verify = [c for c in completion if c['role'] == 'assistant']
            for msg in messages_to_verify:
                individual_messages = [m.split('\nassistant\n')[0] for m in msg['content'].split('\nuser\n')]
                for im in individual_messages:
                    if im.count('Nothing happens.') > 0:
                        reward -= 0.01
            rewards.append(reward)
        return rewards

    def check_format(self, completions, **kwargs):
        rewards = []
        for completion in completions:
            reward = 0.0
            if isinstance(completion, dict):
                messages_to_verify = [completion,]
            elif isinstance(completion, list):
                messages_to_verify = [c for c in completion if c['role'] == 'assistant']
            for msg in messages_to_verify:
                individual_messages = [m.split('\nuser\n')[0] for m in msg['content'].split('\nassistant\n')]
                for im in individual_messages:
                    for tag in ['think', 'action']:
                        if im.count(f'<{tag}>') == 1 and im.count(f'</{tag}>') == 1:
                            reward += 0.01
            rewards.append(reward)
        return rewards

    def check_env(self, completions, **kwargs):
        completions = completions.copy()
        prompts = kwargs['prompts'].copy()
        rewards = [10.0 * (len(d) < 50 and True in d) for d in self.env.dones]
        return rewards

    def get_reward_funcs(self):
        return [self.check_env, self.check_format, self.check_valid_action]
    
ALFWORLD_PROMPT = """You are a service robot inside a domestic environment.
You must choose an action inside <action> tags.
Example:

<action>
go to garbagecan 1
</action>

"""

ALFWORLD_FEWSHOT = []

class AlfworldEnv(MultiStepEnv):
    def __init__(self, system_prompt = ALFWORLD_PROMPT, few_shot = ALFWORLD_FEWSHOT, dataset_size = 10000, **kwargs):
        super().__init__(system_prompt, few_shot, **kwargs)
        config = generic.load_config()
        env_type = config['env']['type']
        env = get_environment(env_type)(config, train_eval='train')
        env = env.init_env(batch_size=1)
        self.dataset_size = dataset_size
        self.env = env
        self.new_available_actions = []
        self.help_text = self.get_help_text()

    def get_help_text(self):
        _, _ = self.env.reset()
        return self.env.step(['help'])[0][0]

    def reset(self):
        obs, info = self.env.reset()
        self.dones.append([False])
        self.rewards.append([0.0])
        self.all_available_actions = info['admissible_commands'][0]
        return obs[0]

    def env_response(self, messages, **kwargs):
        last_message = messages[-1]['content']
        last_action = last_message.split('</thinking>')[-1].split('<action>')[-1].split('</action')[0].strip()
        observation, reward, done, info = self.env.step([last_action,])
        new_available_actions = [a for a in info['admissible_commands'][0] if a not in self.all_available_actions]
        observation = observation[0]
        reward = reward[0]
        done = done[0]
        self.rewards[-1].append(reward)
        self.dones[-1].append(done)
        self.all_available_actions += new_available_actions
        return {
            'content': observation,
            'role': 'user',
        }

    def get_dataset(self, **kwargs):
        initial_observations = []
        for i in tqdm.trange(self.dataset_size):
            initial_observations.append([{
                'content': self.system_prompt + f"\n\n{self.help_text}",
                'role': 'system',
            },
            *self.few_shot,
            {
                'content': '',
                'role': 'user'
            }])
        return Dataset.from_dict(mapping=dict(prompt=initial_observations))
    
    def get_sft_dataset(self, **kwargs):
        conversations = []
        for i in tqdm.trange(self.dataset_size):
            conversation = []
            obs, info = self.env.reset()
            observation = obs[0]
            conversation.append({
                'role': 'system',
                'content': self.system_prompt + f"\n\n{self.help_text}"
            })
            conversation.append({
                'role': 'user',
                'content': observation
            })
            done = False
            steps = 0
            actions = []
            observations = []
            while not done:
                action = info['extra.expert_plan'][0][0]
                actions.append(action)
                obs, reward, done, info = self.env.step([action])
                observation =  obs[0]
                done = done[0]
                observations.append(observation)
                steps += 1
            if steps < 50:
                for j, (action, observation) in enumerate(zip(actions, observations)):
                    plan = f"I will {', '.join(actions[j:])}."
                    response = f"<think>\n{plan}\n</think>\n<action>\n{action}\n</action>"
                    conversation.append({
                        'role': 'assistant',
                        'content': response
                    })
                    conversation.append({
                        'role': 'user',
                        'content': observation
                    })
                conversations.append(conversation[:-1])
        return Dataset.from_dict(mapping=dict(conversations=conversations))
            
    def get_rl_dataset(self, **kwargs):
        conversations = []
        for i in tqdm.trange(self.dataset_size):
            conversation = []
            obs, info = self.env.reset()
            observation = obs[0]
            done = False
            steps = 0
            actions = []
            observations = []
            while not done:
                action = info['extra.expert_plan'][0][0]
                actions.append(action)
                obs, reward, done, info = self.env.step([action])
                observation =  obs[0]
                done = done[0]
                observations.append(observation)
                steps += 1
            if steps < 50:
                for j, (action, observation) in enumerate(zip(actions, observations)):
                    plan = f"I will {', '.join(actions[j:])}."
                    response = f"<think>\n{plan}\n</think>\n<action>\n{action}\n</action>"
                    conversation.append({
                        'role': 'system',
                        'content': self.system_prompt + f"\n\n{self.help_text}"
                    })
                    conversation.append({
                        'role': 'user',
                        'content': observation
                    })
                    conversation.append({
                        'role': 'assistant',
                        'content': response
                    })
                conversations.append(conversation[:-1])
        return Dataset.from_dict(mapping=dict(conversations=conversations))
    
    def get_rubric(self, **kwargs):
        return AlfworldRubric(env=self).get_reward_funcs()
    
    def generate(self, prompts, llm, sampling_params, output_type = "ids", **kwargs):
        completions = []
        self.rewards = deque([], maxlen=len(prompts))
        self.dones = deque([], maxlen=len(prompts))
        for prompt in prompts.copy():
            new_prompt = deepcopy(prompt)
            obs = self.reset()
            new_prompt[-1] = {
                'role': 'user',
                'content': obs
            }
            completions += super().generate([new_prompt,], llm, sampling_params, output_type, **kwargs)
        return completions
    
    def is_completed(self, messages, **kwargs):
        return self.dones[-1][-1]

if __name__ == '__main__':
    model_name = "unsloth/SmolLM2-1.7B-Instruct-bnb-4bit"
    max_seq_length = 8192 # Can increase for longer reasoning traces
    lora_rank = 64 # Larger rank = smarter, but slower
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        load_in_4bit = True, # False for LoRA 16bit
        fast_inference = True, # Enable vLLM fast inference
        max_lora_rank = lora_rank,
        gpu_memory_utilization = 0.5, # Reduce if out of memory
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = [
            #"q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ], # Remove QKVO if out of memory
        lora_alpha = lora_rank,
        use_gradient_checkpointing = "unsloth", # Enable long context finetuning
        random_state = 3407,
    )
    model = FastLanguageModel.for_training(model=model, use_gradient_checkpointing=True)
    vf_env = AlfworldEnv(dataset_size=2000, sampling_args={
        'max_tokens': 512
    })
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        env=vf_env,
        reward_funcs=vf_env.get_rubric(),
        args=get_default_grpo_config(run_name='alphahome-rl-3B', num_gpus=1),
        train_dataset=vf_env.get_dataset(),
    )

    trainer.train()