import gym
import time

from hiro import HIRO  # Note that flags from hiro.py are imported
from absl import flags, app

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_episodes', 10, 'Number of episodes to run the trained agent')


def main(argv):
    env_name = FLAGS.env_name
    env = gym.make(env_name)
    agent = HIRO(env, load_path=FLAGS.load_path, training=False)

    for episodes in range(FLAGS.num_episodes):
        done = False
        obs = env.reset()
        episode_reward = 0
        while not done:
            env.render()
            goal = agent.get_goal(obs.reshape(1, -1), noise=False)
            action = agent.act(obs.reshape(1, -1), goal, noise=False)
            obs, rew, done, info = env.step(action)
            obs = obs.flatten()
            episode_reward += rew
            time.sleep(0.01)
        print(f'Episode Reward:{episode_reward}')
    env.close()


if __name__ == '__main__':
    app.run(main)
