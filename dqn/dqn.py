import gym
import numpy as np
import tensorflow as tf

from absl import flags, app
from util.replay_buffer import ReplayBuffer
from tensorflow.keras import layers, Model

FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'CartPole-v0', 'Environment name')
flags.DEFINE_string('log_path', './logs', 'Location to log training data')
flags.DEFINE_string('save_path', 'train/', 'Save  location for the model')
flags.DEFINE_string('load_path', None, 'Load location for the model')
flags.DEFINE_float('learning_rate', 1e-3, 'network learning rate')
flags.DEFINE_float('gamma', 0.99, 'discount factor')
flags.DEFINE_integer('target_update', 10000, 'Steps before we update target network')
flags.DEFINE_integer('batch_size', 32, 'Number of training examples in the batch')
flags.DEFINE_integer('buffer_capacity', 10000, 'Number of transitions to keep in replay buffer')
flags.DEFINE_integer('max_steps', 100000, 'Maximum number of training steps')
flags.DEFINE_integer('max_episode_len', None, 'Maximum length of each episode')
flags.DEFINE_integer('print_freq', 1, 'Episodes between displaying log info')
flags.DEFINE_float('min_eps', 0.1, 'minimum for epsilon greedy exploration')
flags.DEFINE_float('max_eps', 1.0, 'maximum for epsilon greedy exploration')
flags.DEFINE_float('eps_decay', -1e-4, 'decay schedule for epsilon')
flags.DEFINE_boolean('render', False, 'Render the environment during training')
flags.DEFINE_integer('seed', 1234, 'Random seed for reproducible results')


@tf.function
def copy_weights(vars1, vars2):
    """
    Copies the values of tensors in vars1 to the corresponding tensors in vars2
    Args:
        vars1: list of tensors
        vars2: list of tensors
    """
    for a, b in zip(vars1, vars2):
        tf.compat.v1.assign(b, a)


class QNetwork(Model):
    def __init__(self, out_dim, name='q_network'):
        """
        Creates a Q network
        Args:
            out_dim (int): number of possible actions
            name (str): name of the model
        """
        super(QNetwork, self).__init__(name=name)
        self.q1 = layers.Dense(32, activation='relu', name='q_pred_1')
        self.q2 = layers.Dense(32, activation='relu', name='q_pred_2')
        self.q_value = layers.Dense(out_dim, activation='linear', name='q_pred')

    def call(self, state, **kwargs):
        """
        Runs a forward pass of the model
        Args:
            state (np.ndarray): model input (a single state)
            **kwargs:

        Returns:
            action_pred: output of the model
        """
        action_pred = self.q1(state)
        action_pred = self.q2(action_pred)
        action_pred = self.q_value(action_pred)
        return action_pred


class DQN:
    def __init__(self,
                 env,
                 learning_rate=1e-3,
                 seed=1234,
                 gamma=0.99,
                 max_eps=1.0,
                 min_eps=0.1,
                 render=False,
                 print_freq=1,
                 load_path=None,
                 save_path=None,
                 batch_size=32,
                 log_dir='logs/train',
                 max_steps=100000,
                 buffer_capacity=None,
                 max_episode_len=None,
                 eps_decay_rate=-1e-4,
                 target_update_freq=1000,
                 ):
        tf.random.set_seed(seed)
        np.random.seed(seed)
        self.gamma = gamma
        self.render = render
        self.batch_size = batch_size
        self.print_freq = print_freq
        self.q_lr = learning_rate
        self.max_eps = max_eps
        self.min_eps = min_eps
        self.eps_decay_rate = eps_decay_rate
        self.buffer = ReplayBuffer(buffer_capacity)
        self.max_steps = max_steps
        self.target_update = target_update_freq
        self.model = QNetwork(env.action_space.n, name='q_network')
        self.target = QNetwork(env.action_space.n, name='target_network')
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.q_lr)
        self.summary_writer = tf.summary.create_file_writer(log_dir)
        self.env = env
        self.max_episode_len = max_episode_len if max_episode_len else self.env.spec.max_episode_steps
        self.rewards = []
        self.save_path = save_path

        if load_path is not None:
            self.model.load_weights(load_path)

    def act(self, state):
        return np.argmax(self.model(state))

    @tf.function
    def train_step(self, states, indices, targets):
        """
        Performs a single step of gradient descent on the Q network

        Args:
            states: numpy array of states with shape (batch size, state dim)
            indices: list indices of the selected actions
            targets: targets for computing the MSE loss

        """
        with tf.GradientTape() as tape:
            action_values = tf.gather_nd(self.model(states), indices)
            mse_loss = tf.keras.losses.MeanSquaredError()(action_values, targets)

        gradients = tape.gradient(mse_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Log training information
        with self.summary_writer.as_default():
            tf.summary.scalar('MSE Loss', mse_loss, step=self.optimizer.iterations)
            tf.summary.scalar('Estimated Q Value', tf.reduce_mean(action_values), step=self.optimizer.iterations)

    def update(self):
        """
        Computes the target for the MSE loss and calls the tf.function for gradient descent
        """
        if len(self.buffer) >= self.batch_size:
            # Sample random minibatch of N transitions
            states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

            # Adjust the targets for non-terminal states
            next_state_pred = self.target(next_states)
            targets = rewards + self.gamma * next_state_pred.numpy().max(axis=1) * (1 - dones)
            batch_range = tf.range(start=0, limit=actions.shape[0])
            indices = tf.stack((batch_range, actions), axis=1)

            # update critic by minimizing the MSE loss
            self.train_step(states, indices, targets)

    def learn(self):
        """Learns via Deep-Q-Networks (DQN)"""
        obs = self.env.reset()
        total_reward = 0
        ep = 0
        ep_len = 0
        rand_actions = 0
        mean_reward = None
        for t in range(self.max_steps):

            if t % self.target_update == 0:
                copy_weights(self.model.variables, self.target.variables)

            # weight decay from https://jaromiru.com/2016/10/03/lets-make-a-dqn-implementation/
            eps = self.min_eps + (self.max_eps - self.min_eps) * np.exp(
                self.eps_decay_rate * t)
            if self.render:
                self.env.render()

            # Take exploratory action with probability epsilon
            if np.random.uniform() < eps:
                action = self.env.action_space.sample()
                rand_actions += 1
            else:
                action = self.act(np.expand_dims(obs, axis=0))

            # Execute action in emulator and observe reward and next state
            new_obs, reward, done, info = self.env.step(action)
            total_reward += reward

            # Store transition s_t, a_t, r_t, s_t+1 in replay buffer
            self.buffer.add((obs, action, reward, new_obs, done))

            # Perform learning step
            self.update()

            obs = new_obs
            ep_len += 1
            if done or ep_len >= self.max_episode_len:
                with self.summary_writer.as_default():
                    ep += 1
                    self.rewards.append(total_reward)
                    total_reward = 0
                    obs = self.env.reset()

                    if ep % self.print_freq == 0 and ep > 0:
                        new_mean_reward = np.mean(self.rewards[-self.print_freq - 1:])

                        print(f"-------------------------------------------------------")
                        print(f"Mean {self.print_freq} Episode Reward: {new_mean_reward}")
                        print(f"Exploration fraction: {rand_actions / ep_len}")
                        print(f"Total Episodes: {ep}")
                        print(f"Total timesteps: {t}")
                        print(f"-------------------------------------------------------")

                        tf.summary.scalar(f'Mean {self.print_freq} Episode Reward', new_mean_reward, step=t)
                        tf.summary.scalar(f'Epsilon', eps, step=t)

                        # Model saving inspired by Open AI Baseline implementation
                        if (mean_reward is None or new_mean_reward >= mean_reward) and self.save_path is not None:
                            print(f"Saving model due to mean reward increase:{mean_reward} -> {new_mean_reward}")
                            print(f'Location: {self.save_path}')
                            mean_reward = new_mean_reward
                            self.model.save_weights(self.save_path)

                    ep_len = 0
                    rand_actions = 0


def main(argv):
    env = gym.make(FLAGS.env_name)
    dqn = DQN(
        env=env,
        learning_rate=FLAGS.learning_rate,
        gamma=FLAGS.gamma,
        print_freq=FLAGS.print_freq,
        target_update_freq=FLAGS.target_update,
        batch_size=FLAGS.batch_size,
        seed=FLAGS.seed,
        buffer_capacity=FLAGS.buffer_capacity,
        render=FLAGS.render,
        max_steps=FLAGS.max_steps,
        min_eps=FLAGS.min_eps,
        max_eps=FLAGS.max_eps,
        eps_decay_rate=FLAGS.eps_decay,
        max_episode_len=FLAGS.max_episode_len,
        log_dir=FLAGS.log_path,
        save_path=FLAGS.save_path,
        load_path=FLAGS.load_path)
    dqn.learn()


if __name__ == '__main__':
    app.run(main)
