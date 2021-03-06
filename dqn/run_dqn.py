import gym
import numpy as np

from dqn import DQN
from absl import flags, app

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_episodes', 1, 'Number of episodes to run the trained agent')


def main(argv):
    env_name = FLAGS.env_name
    env = gym.make(env_name)
    agent = DQN(env, load_path=f'train/{env_name}/')

    for episodes in range(FLAGS.num_episodes):
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
    app.run(main)
