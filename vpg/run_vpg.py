import gym

from vpg import VPG
from absl import flags, app

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_episodes', 10, 'Number of episodes to run the trained agent')


def main(argv):
    env_name = FLAGS.env_name
    env = gym.make(env_name)
    agent = VPG(env, load_path=FLAGS.load_path, training=False)

    for episodes in range(FLAGS.num_episodes):
        done = False
        obs = env.reset()
        episode_reward = 0
        while not done:
            env.render()
            action = agent.act(obs.reshape(1, -1))
            obs, rew, done, info = env.step(action)
            episode_reward += rew
        print(f'Episode Reward:{episode_reward}')
    env.close()


if __name__ == '__main__':
    app.run(main)
