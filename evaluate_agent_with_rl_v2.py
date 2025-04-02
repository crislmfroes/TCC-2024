import tqdm
from datasets import Dataset
from train_agent_with_rl import ALFWORLD_PROMPT
from alfworld.agents.environment import get_environment
from alfworld.agents.environment.alfred_thor_env import AlfredThorEnv
import alfworld.agents.modules.generic as generic
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from collections import Counter
import random

random.seed(123)


config = generic.load_config()
env_type = config['env']['type']
env = get_environment(env_type)(config, train_eval='eval_out_of_distribution')
env: AlfredThorEnv = env.init_env(batch_size=1)
_, _ = env.reset()
obs, _, _, _ = env.step(["help"])
help_text = obs[0]

def evaluate_model(**kwargs):
        success = 0.0
        for i in tqdm.trange(10):
            try:
                obs, info = env.reset()
                observation = obs[0]
                done = False
                steps = 0
                observations = [observation]
                actions = []
                thoughts = []
                while not done:
                    print('Observation: ', observations[-1])
                    memory = "Memory:"
                    for previous_obs, previous_action, previous_thought in zip(actions[:steps], observations[:steps], thoughts[:steps]):
                        memory += f"Observation: {previous_obs}\n"
                        memory += f"Thought: {previous_thought}\n"
                        memory += f"Action: {previous_action}\n"
                    admissible_actions = 'Admissible actions: ' + ','.join(info['admissible_commands'][0])
                    prompt = f"Task: '{observations[0]}'\n{memory}\nCurrent observation: {observation}\n{admissible_actions}"
                    conversation = []
                    conversation.append({
                        'role': 'system',
                        'content': ALFWORLD_PROMPT + f"\n\n{help_text}"
                    })
                    conversation.append({
                        'role': 'user',
                        'content': prompt
                    })
                    inputs = tokenizer.apply_chat_template(
                        [conversation,]*4,
                        tokenize = True,
                        add_generation_prompt = True, # Must add for generation
                        return_tensors = "pt",
                    ).to("cuda")

                    outputs = model.generate(
                        input_ids = inputs, max_new_tokens = 3000, use_cache = True, temperature = 0.8
                    )
                    completions = tokenizer.batch_decode(outputs[:,inputs.shape[-1]:], skip_special_tokens=True)
                    sampled_actions = []
                    action2thought = {}
                    for completion in completions:
                        action = completion.split('<action>')[-1].split('</action>')[0].strip()
                        sampled_actions.append(action)
                        if action not in action2thought.keys():
                             action2thought[action] = []
                        thought = completion.split('<think>')[-1].split('</think>')[0].strip()
                        action2thought[action].append(thought)
                    action_counter = Counter(sampled_actions)
                    action = action_counter.most_common(1)[0][0]
                    actions.append(action)
                    thought = sorted(action2thought[action], key=lambda e: len(e))[-1]
                    thoughts.append(thought)
                    print('Thought: ', thoughts[-1])
                    print('Action: ', actions[-1])
                    admissible_commands = info['admissible_commands'][0]
                    obs, reward, done, info = env.step([action])
                    if action not in admissible_commands:
                         observation = f"Invalid action '{action}'!\n{help_text}"
                    else:
                        observation =  obs[0]
                    observations.append(observation)
                    done = done[0]
                    steps += 1
                if steps < 50:
                    success += 1.0
                    print('Success rate: ', success/float(i+1))
            except BaseException as e:
                 if isinstance(e, KeyboardInterrupt):
                      raise e
                 print(e)
        return success/float(i+1)


if __name__ == '__main__':
    model_name = "./outputs/alphahome-rl-1.5B/checkpoint-9237"
    #model_name = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
    max_seq_length = 8192 # Can increase for longer reasoning traces
    lora_rank = 64 # Larger rank = smarter, but slower
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        load_in_4bit = True, # False for LoRA 16bit
        fast_inference = True, # Enable vLLM fast inference
        max_lora_rank = lora_rank,
        gpu_memory_utilization = 0.7, # Reduce if out of memory
    )
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "qwen-2.5",
    )
    FastLanguageModel.for_inference(model)
    success_rate = evaluate_model()
    print(f'{model_name} success rate: ', success_rate)