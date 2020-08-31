import gym
import numpy as np

from dqn import DQN
from absl import flags, app


def main():
    # env_name = 'CartPole-v0'
    env_name = 'LunarLander-v2'
    env = gym.make(env_name)

    agent = DQN(env, load_path=f'train/{env_name}/')

    for episodes in range(100):
        done = False
        obs = env.reset()
        episode_reward = 0
        while not done:
            env.render()
            action = agent.act(np.expand_dims(obs, axis=0))
            obs, rew, done, info = env.step(action)
            episode_reward += rew
        print(f'Episode Reward:{episode_reward}')


if __name__ == '__main__':
    main()
