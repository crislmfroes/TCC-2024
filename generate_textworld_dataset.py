import textworld
from textworld.generator import make_game, compile_game
options = textworld.GameOptions()
options.seeds = 1234
game = make_game(options)
#game.extras["more"] = "This is extra information."
gamefile = compile_game(game)
import gym
import textworld.gym
from textworld import EnvInfos
from textworld.agents.walkthrough import WalkthroughAgent

request_infos = EnvInfos(description=True, inventory=True, extras=["more"])
env_id = textworld.gym.register_game(gamefile, request_infos)
env = textworld.gym.make(env_id)
obs, infos = env.reset()
done = False
reward = 0
agent = WalkthroughAgent()
while not done:
    print('obs: ', obs)
    print('infos: ', infos)
    action = agent.act(game_state=obs, reward=reward, done=done)
    print('action: ', action)
    obs, reward, done, infos = env.step(action)