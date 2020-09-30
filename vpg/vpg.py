import gym
import numpy as np
import tensorflow as tf

from absl import flags, app
from tensorflow.keras import layers, Model

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'CartPole-v0', 'Environment name')
flags.DEFINE_string('log_path', './logs', 'Directory to log training data')
flags.DEFINE_string('save_path', 'train/', 'Directory to save the model')
flags.DEFINE_string('load_path', None, 'Directory to load the model from')
flags.DEFINE_float('lr', 1e-3, 'q network learning rate')
flags.DEFINE_integer('max_episodes', 100, 'Maximum number of training episodes')
flags.DEFINE_integer('num_updates', 1, 'Number of gradient descent updates to perform on a collected episode')
flags.DEFINE_integer('print_freq', 1, 'Episodes between displaying log info')
flags.DEFINE_boolean('render', False, 'Render the environment during training')
flags.DEFINE_integer('seed', 1234, 'Random seed for reproducible results')


class Actor(Model):
    def __init__(self, action_dim, **kwargs):
        """
        Model for the actor in VPG.
        Args:
            action_dim (int): Number of dimensions in the action vector
            **kwargs (dict): arbitrary keyword arguments
        """
        super(Actor, self).__init__(kwargs)
        self.layer_1 = layers.Dense(32, activation='relu', name='actor_1')
        self.layer_2 = layers.Dense(32, activation='relu', name='actor_2')
        self.layer_3 = layers.Dense(action_dim, name='linear_action_pred')
        self.action_probabilities = layers.Softmax()

    def call(self, states, **kwargs):
        """
        Runs a forward pass of the model.
        Args:
            states: State vectors
            **kwargs (dict): arbitrary keyword arguments

        Returns:
            Action predictions
        """
        out = self.layer_1(states)
        out = self.layer_2(out)
        out = self.layer_3(out)
        return self.action_probabilities(out)


def create_actor(state_dim, action_dim):
    inputs = layers.Input(shape=(state_dim,))
    outputs = Actor(action_dim=action_dim)(inputs)
    return Model(inputs=inputs, outputs=outputs)


class Trajectory:
    def __init__(self, states, actions, rewards, total_reward):
        """
        Represents a single trajectory sampled from our policy
        Args:
            states: list of states visited during the trajectory (excluding final state)
            actions: list of actions taken from each state
            rewards: list of rewards received in the trajectory
        """

        self.states = np.concatenate(states)
        self.actions = actions
        self.advantage = np.zeros(shape=len(self.actions))
        self.total_reward = total_reward
        mean_reward = np.mean(rewards)
        std = np.std(rewards)
        std = 1 if std == 0 else std

        # Compute Reward-to-go
        reward_to_go = total_reward
        for i in range(len(states)):
            self.advantage[i] = (reward_to_go - mean_reward) / std
            reward_to_go -= rewards[i]


class VPG:

    def __init__(self,
                 env,
                 render=False,
                 max_episodes=1000,
                 print_freq=20,
                 load_path=None,
                 save_path=None,
                 num_updates=1,
                 lr=1e-3,
                 log_path='logs/train',
                 training=True,
                 ):
        """Trains an agent via vanilla policy gradient with average reward baseline.
        Args:
            env: gym.Env where our agent resides.
            sess: tensorflow session
            render: True to render the environment, else False
            max_episodes: maximum number of episodes to train for
            print_freq: Displays logging information every 'print_freq' episodes
            load_path: (str) Path to load existing model from
            save_path: (str) Path to save model during training
        """
        self.max_episodes = max_episodes
        self.print_freq = print_freq
        self.env = env
        self.input_dim = env.observation_space.shape[0]
        self.output_dim = env.action_space.n
        self.render = render
        self.num_updates = num_updates
        self.save_path = save_path
        self.rewards = []
        self.actor = create_actor(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
        if training:
            self.ce_loss = tf.keras.losses.SparseCategoricalCrossentropy()
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            self.summary_writer = tf.summary.create_file_writer(log_path)
        if load_path is not None:
            self.actor.load_weights(f'{load_path}/actor')
            print(f'Successfully loaded model from {load_path}')

    def act(self, observation):
        pred = self.actor(observation)
        return np.random.choice(range(self.output_dim), p=pred.numpy().flatten())

    def learn(self):
        """Trains an agent via vanilla policy gradient"""
        total_reward = 0
        total_timesteps = 0
        mean_reward = None
        for e in range(self.max_episodes):
            if e % self.print_freq == 0 and e > 0:
                new_mean_reward = total_reward / self.print_freq
                total_reward = 0
                print(f"-------------------------------------------------------")
                print(f"Mean {self.print_freq} Episode Reward: {new_mean_reward}")
                print(f"Total Episodes: {e}")
                print(f"Total timesteps: {total_timesteps}")
                print(f"-------------------------------------------------------")

                with tf.name_scope('VPG'):
                    with tf.name_scope('Episodic Information'):
                        with self.summary_writer.as_default():
                            tf.summary.scalar(f'Mean {self.print_freq} Episode Reward', new_mean_reward, step=e)

                # Model saving inspired by Open AI Baseline implementation
                if (mean_reward is None or new_mean_reward >= mean_reward) and self.save_path is not None:
                    print(f"Saving model due to mean reward increase:{mean_reward} -> {new_mean_reward}")
                    print(f'Location: {self.save_path}')
                    self.actor.save_weights(f'{self.save_path}/actor')
                    mean_reward = new_mean_reward

            traj = self.sample()
            total_reward += traj.total_reward
            total_timesteps += len(traj.states)
            self.rewards.append(traj.total_reward)

            # Compute and apply the gradient of the log policy multiplied by our baseline
            for u in range(self.num_updates):
                self.train_step(traj.states, traj.actions, traj.advantage)

    @tf.function
    def train_step(self, states, actions, advantage):
        with tf.GradientTape() as tape:
            action_probabilities = self.actor(states)
            loss = self.ce_loss(actions, action_probabilities, sample_weight=advantage)
        gradients = tape.gradient(loss, self.actor.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables))

        with tf.name_scope('VPG'):
            with self.summary_writer.as_default():
                tf.summary.scalar('Policy Loss', loss, step=self.optimizer.iterations)
                tf.summary.scalar('Average Advantage',
                                  tf.reduce_mean(advantage),
                                  step=self.optimizer.iterations)

    def sample(self):
        """Samples a single trajectory under the current policy."""
        done = False
        actions = []
        states = []
        rewards = []

        obs = self.env.reset()
        total_reward = 0
        while not done:
            obs = obs.reshape(1, -1)
            states.append(obs)
            if self.render:
                self.env.render()

            action = self.act(obs)
            obs, rew, done, _ = self.env.step(action)

            total_reward += rew
            rewards.append(rew)
            actions.append(action)
        return Trajectory(states, actions, rewards, total_reward)


def main(argv):
    env = gym.make(FLAGS.env_name)
    vpg = VPG(
        env=env,
        render=FLAGS.render,
        max_episodes=FLAGS.max_episodes,
        num_updates=FLAGS.num_updates,
        print_freq=FLAGS.print_freq,
        load_path=FLAGS.load_path,
        save_path=FLAGS.save_path,
        lr=FLAGS.lr,
        log_path=FLAGS.log_path,
    )
    vpg.learn()


if __name__ == '__main__':
    app.run(main)
