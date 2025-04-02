import argparse
import os
import pickle
import time

from pydantic import BaseModel

from reasoners import Reasoner
from reasoners.algorithm import MCTS, MCTSResult, MCTSNode
from reasoners.lm import OpenAIModel
from reasoners.lm.openai_model import GenerateOutput, PROMPT_TEMPLATE_ANSWER, PROMPT_TEMPLATE_CONTINUE
import alfworld.agents.environment as environment_alfworld
import alfworld.agents.modules.generic as generic

from alfworld_gym_env import AlfworldGym, AlfworldWorldModel
from search_config import SearchConfigAlfworld
from utils.misc import obs_preprocessor
from utils.parse import parse_common_arguments

import random

import tqdm

random.seed(123)

class MyOpenAIModel(OpenAIModel):
    def generate(self, prompt, max_tokens = None, top_p = 1, num_return_sequences = 1, rate_limit_per_min = 2000, stop = None, logprobs = None, temperature=None, additional_prompt=None, retry=64, response_model: BaseModel=None, **kwargs):
        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        temperature = self.temperature if temperature is None else temperature
        logprobs = 0 if logprobs is None else logprobs

        if isinstance(prompt, list):
            assert len(prompt) == 1  # @zj: why can't we pass a list of prompts?
            prompt = prompt[0]
        if additional_prompt is None and self.additional_prompt is not None:
            additional_prompt = self.additional_prompt
        elif additional_prompt is not None and self.additional_prompt is not None:
            print("Warning: additional_prompt set in constructor is overridden.")
        if additional_prompt == "ANSWER":
            prompt = PROMPT_TEMPLATE_ANSWER + prompt
        elif additional_prompt == "CONTINUE":
            prompt = PROMPT_TEMPLATE_CONTINUE + prompt

        is_instruct_model = self.is_instruct_model
        if not is_instruct_model:
            # Recheck if the model is an instruct model with model name
            model_name = self.model.lower()
            if (
                ("gpt-3.5" in model_name)
                or ("gpt-4" in model_name)
                or ("instruct" in model_name)
            ):
                is_instruct_model = True

        for i in range(1, retry + 1):
            try:
                # sleep several seconds to avoid rate limit
                if rate_limit_per_min is not None:
                    time.sleep(60 / rate_limit_per_min)
                ### GPT 3.5 and higher use a different API
                if is_instruct_model:
                    messages = [{"role": "user", "content": prompt}]
                    response = self.client.beta.chat.completions.parse(
                        model=self.model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        n=num_return_sequences,
                        stop=stop,
                        response_format=response_model,
                    )
                    return GenerateOutput(
                        text=[choice.message.content for choice in response.choices],
                        log_prob=None,
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        n=num_return_sequences,
                        stop=stop,
                        logprobs=0,
                        **kwargs,
                    )
                    return GenerateOutput(
                        text=[choice["text"] for choice in response.choices],
                        log_prob=[choice["logprobs"] for choice in response["choices"]],
                    )

            except Exception as e:
                print(f"An Error Occured: {e}, sleeping for {i} seconds")
                time.sleep(i)

        # after 64 tries, still no luck
        raise RuntimeError(
            "GPTCompletionModel failed to generate output, even after 64 tries"
        )

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run a task with specified parameters."
    )
    parse_common_arguments(parser)

    # MCTS parameters
    parser.add_argument(
        "--n_iters", type=int, default=2, help="Number of iterations for MCTS."
    )
    parser.add_argument(
        "--depth_limit", type=int, default=10, help="Depth limit for MCTS."
    )
    parser.add_argument(
        "--w_exp",
        type=float,
        default=10**0.5,
        help="Exploration weight of the UCT score for MCTS.",
    )

    return parser.parse_known_args()[0]


def run_task(args):

    exp_dir = os.path.join(args.exp_dir, args.task_name)
    os.makedirs(exp_dir, exist_ok=True)

    config = generic.load_config()
    env_type = config['env']['type']
    # setup environment
    env = getattr(environment_alfworld, env_type)(config, train_eval='eval_out_of_distribution')
    #env = getattr(environment, env_type)(config, train_eval='train')
    env = env.init_env(batch_size=1)
    for i in tqdm.trange(10):
        _, _ = env.reset()
        task = env.envs[0].task_file

        llm = MyOpenAIModel(
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            is_instruct_model=True,
        )
        llm.client.base_url = 'http://0.0.0.0:8000/v1'
        #print(llm.generate(prompt='hello world!', num_return_sequences=3))
        #exit()
        environment = AlfworldGym(env=env, env_task=task, obs_preprocessor=obs_preprocessor)
        state = environment.init_state()
        #print(state)
        #exit()
        success = 0
        episodes = 0
        task_start_time = time.time()
        while (not state.done) and (time.time() - task_start_time < 30*60):
            world_model = AlfworldWorldModel(llm=llm, max_steps=environment.max_steps)
            search_config = SearchConfigAlfworld(
                n_proposals=10,
                llm=llm,
                use_text_observation=True,
                use_vision_observation=False
            )
            algorithm = MCTS(
                n_iters=args.n_iters,
                depth_limit=args.depth_limit,
                w_exp=args.w_exp,
                uct_with_fast_reward=True,
                disable_tqdm=False,
                output_trace_in_each_iter=True,
            )

            reasoner = Reasoner(world_model, search_config, algorithm)

            plan_result: MCTSResult = reasoner(example=[state._asdict()])

            #with open(f"{exp_dir}/result.pkl", "wb") as f:
            #    pickle.dump(plan_result, f)

            sorted_childs = sorted(plan_result.tree_state.children, key=lambda n: n.Q, reverse=True)
            try:
                choosen_node: MCTSNode = sorted_childs[0]
                action_choice = choosen_node.action
                print(action_choice)
                state, _ = environment.step(state=state, action=action_choice)
            except BaseException as e:
                print(e)

        print('episode len: ', state.step_idx)

        if state.reward == 1.0:
            success += 1

        episodes += 1
        
        if episodes > 0:
            print('success rate: ', success/episodes)


if __name__ == "__main__":
    args = parse_arguments()

    start_time = time.time()
    run_task(args)

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
