import gym
import numpy as np
import tensorflow as tf

from absl import flags, app
from util import ReplayBuffer
from tensorflow.keras import layers, Model

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'Pendulum-v0', 'Environment name')
flags.DEFINE_string('log_path', './logs', 'Location to log training data')
flags.DEFINE_string('save_path', 'train/', 'Save  location for the model')
flags.DEFINE_string('load_path', None, 'Load location for the model')
flags.DEFINE_float('q_lr', 1e-3, 'q network learning rate')
flags.DEFINE_float('p_lr', 1e-4, 'policy network learning rate')
flags.DEFINE_float('act_noise', 0.1, 'standard deviation of actor noise')
flags.DEFINE_float('gamma', 0.99, 'discount factor')
flags.DEFINE_float('polyak', 0.995, 'polyak averaging coefficient')
flags.DEFINE_integer('batch_size', 32, 'Number of training examples in the batch')
flags.DEFINE_integer('buffer_capacity', 5000, 'Number of transitions to keep in replay buffer')
flags.DEFINE_integer('max_episodes', 100, 'Maximum number of training episodes')
flags.DEFINE_integer('start_steps', 10000, 'Number of steps for random action selection before running the real policy')
flags.DEFINE_integer('max_episode_len', None, 'Maximum length of each episode')
flags.DEFINE_integer('print_freq', 1, 'Episodes between displaying log info')
flags.DEFINE_boolean('render', False, 'Render the environment during training')
flags.DEFINE_integer('seed', 1234, 'Random seed for reproducible results')


@tf.function
def polyak_average(vars1, vars2, polyak):
    for a, b in zip(vars1, vars2):
        tf.compat.v1.assign(b, polyak * b + (1 - polyak) * a)


class DDPGModel(Model):
    def __init__(self, out_dim):
        super(DDPGModel, self).__init__()
        # Actor variables
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
        self.a1 = layers.Dense(512, activation='relu', name='actor_1')
        self.a2 = layers.Dense(512, activation='relu', name='actor_2')
        self.action_prediction = layers.Dense(out_dim,
                                              activation='tanh',
                                              kernel_initializer=last_init,
                                              name='actor_action_pred')

        self.q1 = layers.Dense(512, activation='relu', name='critic_1')
        self.q2 = layers.Dense(512, activation='relu', name='critic_2')
        self.q_pred = layers.Dense(1, name='critic_q_pred')

    def calculate_q_value(self, states, action_predictions):
        q_in = layers.Concatenate()([states, action_predictions])
        out = self.q1(q_in)
        out = layers.BatchNormalization()(out)
        out = self.q2(out)
        out = layers.BatchNormalization()(out)
        q_pred = self.q_pred(out)
        return action_predictions, q_pred

    def calculate_action_predictions(self, states):
        out = self.a1(states)
        out = layers.BatchNormalization()(out)
        out = self.a2(out)
        out = layers.BatchNormalization()(out)
        return 2 * self.action_prediction(out)

    def call(self, states, **kwargs):
        return self.calculate_q_value(states=states, action_predictions=self.calculate_action_predictions(states))


class DDPG:
    def __init__(self, env,
                 gamma=0.99,
                 polyak=0.995,
                 act_noise=0.1,
                 render=False,
                 batch_size=32,
                 q_lr=1e-3,
                 p_lr=1e-4,
                 buffer_capacity=5000,
                 max_episodes=100,
                 save_path=None,
                 load_path=None,
                 print_freq=1,
                 start_steps=10000,
                 log_dir='logs/train',
                 ):
        self.first_update = False
        self.gamma = gamma
        self.polyak = polyak
        self.act_noise = act_noise
        self.render = render
        self.batch_size = batch_size
        self.p_lr = p_lr
        self.q_lr = q_lr
        self.max_episodes = max_episodes
        self.start_steps = start_steps
        self.model = DDPGModel(env.action_space.shape[0])
        self.target = DDPGModel(env.action_space.shape[0])
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.p_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.q_lr)
        self.buffer = ReplayBuffer(buffer_capacity)
        self.summary_writer = tf.summary.create_file_writer(log_dir)
        self.env = env
        self.rewards = []
        self.print_freq = print_freq
        self.save_path = save_path
        if load_path is not None:
            self.model.load_weights(load_path)

    # @tf.function
    def train_step(self, states, actions, targets):
        with tf.GradientTape() as tape1:
            action_predictions, q_values = self.model(states)
            policy_loss = -tf.reduce_mean(q_values)
        actor_vars = [v for v in self.model.variables if 'actor' in v.name]
        actor_gradients = tape1.gradient(policy_loss, actor_vars)
        self.actor_optimizer.apply_gradients(zip(actor_gradients, actor_vars))

        with tf.GradientTape() as tape2:
            action_predictions, q_values = self.model.calculate_q_value(states, actions)
            mse_loss = tf.keras.losses.MeanSquaredError()(q_values, targets)
        critic_gradients = tape2.gradient(mse_loss, self.model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.model.trainable_variables))

        # Log training information
        with self.summary_writer.as_default():
            tf.summary.scalar('MSE Loss', mse_loss, step=self.critic_optimizer.iterations)
            tf.summary.scalar('Policy Loss', policy_loss, step=self.critic_optimizer.iterations)
            tf.summary.scalar('Estimated Q Value', tf.reduce_mean(q_values), step=self.critic_optimizer.iterations)

    def update(self):
        if len(self.buffer) >= self.batch_size:
            # Sample random minibatch of N transitions
            states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
            dones = dones.reshape(-1, 1)
            rewards = rewards.reshape(-1, 1)

            # Weights are not initialized until used, so we have to initialize the target weights here
            if len(self.buffer) == self.batch_size + 1:
                self.target.set_weights(self.model.get_weights())

            # Set the target for learning and initialize layers for target network if not already
            _, target_q_value = self.target(next_states)
            targets = rewards + self.gamma * target_q_value * (1 - dones)

            # update critic by minimizing the MSE loss
            # update the actor policy using the sampled policy gradient
            self.train_step(states, actions, targets)

            # Update target networks
            polyak_average(self.model.variables, self.target.variables, self.polyak)

    def act(self, obs, noise=False):
        # Initialize a random process N for action exploration
        norm_dist = tf.random.normal(self.env.action_space.shape, stddev=self.act_noise)

        action, q_value = self.model(np.expand_dims(obs, axis=0))
        action = np.clip(action.numpy() + norm_dist.numpy() if noise else 0,
                         a_min=self.env.action_space.low,
                         a_max=self.env.action_space.high)
        return action

    def learn(self):
        mean_reward = None
        total_steps = 0
        for ep in range(self.max_episodes):
            if ep % self.print_freq == 0 and ep > 0:
                new_mean_reward = np.mean(self.rewards[-self.print_freq - 1:])

                print(f"-------------------------------------------------------")
                print(f"Mean {self.print_freq} Episode Reward: {new_mean_reward}")
                print(f"Total Episodes: {ep}")
                print(f"Total Steps: {total_steps}")
                print(f"-------------------------------------------------------")

                total_steps = 0
                with self.summary_writer.as_default():
                    tf.summary.scalar(f'Mean {self.print_freq} Episode Reward', new_mean_reward, step=ep)

                # Model saving inspired by Open AI Baseline implementation
                if (mean_reward is None or new_mean_reward >= mean_reward) and self.save_path is not None:
                    print(f"Saving model due to mean reward increase:{mean_reward} -> {new_mean_reward}")
                    print(f'Location: {self.save_path}')
                    mean_reward = new_mean_reward

                    self.model.save_weights(self.save_path)

            # Receive initial observation state s_1
            obs = self.env.reset()
            done = False
            episode_reward = 0
            ep_len = 0
            while not done:
                # Display the environment
                if self.render:
                    self.env.render()

                # Execute action and observe reward and observe new state
                if self.start_steps > 0:
                    self.start_steps -= 1
                    action = self.env.action_space.sample()
                else:
                    # Select action according to policy and exploration noise
                    action = self.act(obs, noise=True).flatten()
                new_obs, rew, done, info = self.env.step(action)
                new_obs = new_obs.flatten()
                episode_reward += rew

                # Store transition in R
                self.buffer.add((obs, action, rew, new_obs, done))

                # Perform a single learning step
                self.update()

                obs = new_obs
                ep_len += 1

            with self.summary_writer.as_default():
                tf.summary.scalar(f'Episode Reward', episode_reward, step=ep)

            self.rewards.append(episode_reward)
            total_steps += ep_len


def main(argv):
    env = gym.make(FLAGS.env_name)
    ddpg = DDPG(env,
                gamma=FLAGS.gamma,
                polyak=FLAGS.polyak,
                act_noise=FLAGS.act_noise,
                render=FLAGS.render,
                batch_size=FLAGS.batch_size,
                q_lr=FLAGS.q_lr,
                p_lr=FLAGS.p_lr,
                buffer_capacity=FLAGS.buffer_capacity,
                save_path=FLAGS.save_path,
                load_path=FLAGS.load_path,
                log_dir=FLAGS.log_path,
                max_episodes=FLAGS.max_episodes,
                print_freq=FLAGS.print_freq,
                start_steps=FLAGS.start_steps,
                )
    ddpg.learn()


if __name__ == '__main__':
    app.run(main)
