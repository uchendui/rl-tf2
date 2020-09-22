import gym
import numpy as np
import tensorflow as tf
import multiprocessing as mp

from absl import flags, app
from util import ReplayBuffer
from util.tf import polyak_average
from tensorflow.keras import layers, Model

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.config.experimental_run_functions_eagerly(False)

FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'Pendulum-v0', 'Environment name')
flags.DEFINE_string('log_path', 'logs/', 'Location to log training data')
flags.DEFINE_string('save_path', 'train/', 'Save  location for the model')
flags.DEFINE_string('load_path', None, 'Load location for the model')
flags.DEFINE_float('q_lr', 1e-3, 'q network learning rate')
flags.DEFINE_float('p_lr', 1e-4, 'policy network learning rate')
flags.DEFINE_float('high_act_noise', 0.1, 'standard deviation of high level actor noise')
flags.DEFINE_float('low_act_noise', 0.1, 'standard deviation of low level actor noise')
flags.DEFINE_float('high_rew_scale', 0.1, 'reward scaling for high level policy')
flags.DEFINE_float('low_rew_scale', 1.0, 'reward scaling for low level policy')
flags.DEFINE_float('gamma', 0.99, 'discount factor')
flags.DEFINE_float('polyak', 0.995, 'polyak averaging coefficient')
flags.DEFINE_integer('batch_size', 32, 'Number of training examples in the batch')
flags.DEFINE_integer('buffer_capacity', 5000, 'Number of transitions to keep in replay buffer')
flags.DEFINE_integer('max_episodes', 100, 'Maximum number of training episodes')
flags.DEFINE_integer('max_episode_len', None, 'Maximum length of each episode')
flags.DEFINE_integer('print_freq', 1, 'Episodes between displaying log info')
flags.DEFINE_integer('c', 10, 'Number of environment steps between high level policy updates')
flags.DEFINE_boolean('render', False, 'Render the environment during training')
flags.DEFINE_integer('seed', 1234, 'Random seed for reproducible results')


class Actor(Model):
    def __init__(self, action_dim, action_range, **kwargs):
        super(Actor, self).__init__(kwargs)
        # Model architecture from https://keras.io/examples/rl/ddpg_pendulum/
        self.a1 = layers.Dense(512, activation='relu', name='actor_1')
        self.a2 = layers.Dense(512, activation='relu', name='actor_2')
        self.action_prediction = layers.Dense(action_dim,
                                              activation='tanh',
                                              name='actor_action_pred')
        self.batch_norm_1 = layers.BatchNormalization()
        self.batch_norm_2 = layers.BatchNormalization()
        self.action_range = action_range

    def call(self, states, **kwargs):
        out = self.a1(states)
        out = self.batch_norm_1(out)
        out = self.a2(out)
        out = self.batch_norm_2(out)
        return self.action_range * self.action_prediction(out)


class Critic(Model):
    def __init__(self, **kwargs):
        super(Critic, self).__init__(kwargs)
        self.q1 = layers.Dense(512, activation='relu', name='critic_1')
        self.q2 = layers.Dense(512, activation='relu', name='critic_2')
        self.q_pred = layers.Dense(1, name='critic_q_pred')
        self.concat = layers.Concatenate()
        self.batch_norm_1 = layers.BatchNormalization()
        self.batch_norm_2 = layers.BatchNormalization()

    def call(self, inputs, **kwargs):
        out = self.concat(inputs)
        out = self.q1(out)
        out = self.batch_norm_1(out)
        out = self.q2(out)
        out = self.batch_norm_2(out)
        q_pred = self.q_pred(out)
        return q_pred


def create_actor_critic(state_dim, action_dim, action_range):
    state_in = layers.Input(shape=(state_dim,), )
    action_in = layers.Input(shape=(action_dim,), )

    actor_output = Actor(action_dim=action_dim, action_range=action_range)(state_in)
    critic_output = Critic()([state_in, action_in])

    actor = Model(inputs=state_in, outputs=actor_output)
    critic = Model(inputs=[state_in, action_in], outputs=critic_output)
    return actor, critic


class HIRO:
    def __init__(self,
                 env,
                 gamma=0.99,
                 polyak=0.995,
                 c=10,
                 high_act_noise=0.1,
                 low_act_noise=0.1,
                 high_rew_scale=0.1,
                 low_rew_scale=1.0,
                 render=False,
                 batch_size=32,
                 q_lr=1e-3,
                 p_lr=1e-4,
                 buffer_capacity=5000,
                 max_episodes=100,
                 save_path=None,
                 load_path=None,
                 print_freq=1,
                 log_dir='logs/train',
                 training=True
                 ):
        self.gamma = gamma
        self.polyak = polyak
        self.low_act_noise = low_act_noise
        self.high_act_noise = high_act_noise
        self.low_rew_scale = low_rew_scale
        self.high_rew_scale = high_rew_scale
        self.render = render
        self.batch_size = batch_size
        self.p_lr = p_lr
        self.q_lr = q_lr
        self.max_episodes = max_episodes
        self.env = env
        self.rewards = []
        self.print_freq = print_freq
        self.save_path = save_path
        self.c = c
        self.higher_buffer = ReplayBuffer(buffer_capacity, tuple_length=5)
        self.lower_buffer = ReplayBuffer(buffer_capacity, tuple_length=4)

        self.strategy = tf.distribute.MirroredStrategy()

        # with self.strategy.scope():
        self.low_actor, self.low_critic = create_actor_critic(state_dim=2 * env.observation_space.shape[0],
                                                              action_dim=env.action_space.shape[0],
                                                              action_range=env.action_space.high)
        self.low_target_actor, self.low_target_critic = create_actor_critic(
            state_dim=2 * env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            action_range=env.action_space.high)

        self.high_actor, self.high_critic = create_actor_critic(state_dim=env.observation_space.shape[0],
                                                                action_dim=env.observation_space.shape[0],
                                                                action_range=env.observation_space.high
                                                                )
        self.high_target_actor, self.high_target_critic = create_actor_critic(state_dim=env.observation_space.shape[0],
                                                                              action_dim=env.observation_space.shape[0],
                                                                              action_range=env.observation_space.high)
        self.low_target_actor.set_weights(self.low_actor.get_weights())
        self.low_target_critic.set_weights(self.low_critic.get_weights())
        self.high_target_actor.set_weights(self.high_actor.get_weights())
        self.high_target_critic.set_weights(self.high_critic.get_weights())

        if training:
            self.low_actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.p_lr)
            self.low_critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.q_lr)
            self.high_actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.p_lr)
            self.high_critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.q_lr)
            self.mse = tf.keras.losses.MeanSquaredError()
            self.summary_writer = tf.summary.create_file_writer(log_dir)
        if load_path is not None:
            self.low_actor.load_weights(f'{load_path}/low/actor')
            self.low_critic.load_weights(f'{load_path}/low/critic')
            self.high_actor.load_weights(f'{load_path}/high/actor')
            self.high_critic.load_weights(f'{load_path}/high/critic')

    @staticmethod
    def goal_transition(state, goal, next_state):
        return state + goal - next_state

    @staticmethod
    def intrinsic_reward(state, goal, next_state):
        return - np.linalg.norm(state + goal - next_state)

    def act(self, obs, goal, noise=False):
        norm_dist = tf.random.normal(self.env.action_space.shape, stddev=0.1 * self.env.action_space.high)
        action = self.low_actor(np.concatenate((obs, goal), axis=1)).numpy()
        action = np.clip(action + (norm_dist.numpy() if noise else 0),
                         a_min=self.env.action_space.low,
                         a_max=self.env.action_space.high)
        return action

    def get_goal(self, obs, noise=False):
        norm_dist = tf.random.normal(self.env.observation_space.shape, stddev=0.1 * self.env.observation_space.high)
        action = self.high_actor(obs).numpy()
        action = np.clip(action + (norm_dist.numpy() if noise else 0),
                         a_min=self.env.observation_space.low,
                         a_max=self.env.observation_space.high)
        return action

    @tf.function
    def log_probability(self, states, actions, candidate_goal):
        goals = tf.reshape(candidate_goal, (1, -1))

        # If a state-action pair is all zero, then the episode ended before an entire sequence of length c was recorded.
        # We must remove these empty states and actions from the log probability calculation, as they could skew the
        #   argmax computation
        i = 1
        while i < states.shape[0] and not (
                tf.equal(tf.math.count_nonzero(states[i]), 0) and tf.equal(tf.math.count_nonzero(
            actions[i]), 0)):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[(goals, tf.TensorShape([None, goals.shape[1]]))])
            goals = tf.concat(
                (goals, tf.reshape(self.goal_transition(states[i - 1], goals[i - 1], states[i]), (1, -1))), axis=0)
            i += 1
        states = states[:i, :]
        actions = actions[:i, :]

        action_predictions = self.low_actor(tf.concat((states, goals), axis=1))
        return -(1 / 2) * tf.reduce_sum(tf.linalg.norm(actions - action_predictions, axis=1))

    @tf.function
    def off_policy_correct(self, states, goals, actions, new_states):
        first_states = tf.reshape(states, (self.batch_size, -1))[:, :new_states[0].shape[0]]
        means = new_states - first_states
        std_dev = 0.5 * (1 / 2) * tf.convert_to_tensor(self.env.observation_space.high)

        for i in range(states.shape[0]):
            # Sample eight candidate goals sampled randomly from a Gaussian centered at s_{t+c} - s_t
            # Include the original goal and a goal corresponding to the difference s_{t+c} - s_t
            # TODO: clip the random actions to lie within the high-level action range
            candidate_goals = tf.concat(
                (tf.random.normal(shape=(8, self.env.observation_space.shape[0]), mean=means[i], stddev=std_dev),
                 tf.reshape(goals[i], (1, -1)), tf.reshape(means[i], (1, -1))),
                axis=0)

            chosen_goal = tf.argmax(
                [self.log_probability(states[i], actions[i], candidate_goals[g]) for g in
                 range(candidate_goals.shape[0])])
            goals = tf.tensor_scatter_nd_update(goals, [[i]], [candidate_goals[chosen_goal]])

        return first_states, goals

    @tf.function
    def train_step_high(self, states, actions, targets):
        # Update actor
        with tf.GradientTape() as tape:
            goal_predictions = self.high_actor(states)
            q_values = self.high_critic([states, goal_predictions])
            policy_loss = -tf.reduce_mean(q_values)
        gradients = tape.gradient(policy_loss, self.high_actor.trainable_variables)
        self.high_actor_optimizer.apply_gradients(zip(gradients, self.high_actor.trainable_variables))

        # Update critic
        with tf.GradientTape() as tape:
            q_values = self.high_critic([states, actions])
            mse_loss = self.mse(q_values, targets)
        gradients = tape.gradient(mse_loss, self.high_critic.trainable_variables)
        self.high_critic_optimizer.apply_gradients(zip(gradients, self.high_critic.trainable_variables))

        with tf.name_scope("Higher_Policy"):
            with self.summary_writer.as_default():
                tf.summary.scalar('MSE Loss', mse_loss, step=self.high_critic_optimizer.iterations)
                tf.summary.scalar('Policy Loss', policy_loss, step=self.high_critic_optimizer.iterations)
                tf.summary.scalar('Estimated Q Value', tf.reduce_mean(q_values),
                                  step=self.high_critic_optimizer.iterations)

    @tf.function
    def train_step_low(self, states, actions, targets):
        # Update actor
        with tf.GradientTape() as tape:
            action_predictions = self.low_actor(states)
            q_values = self.low_critic([states, action_predictions])
            policy_loss = -tf.reduce_mean(q_values)
        actor_gradients = tape.gradient(policy_loss, self.low_actor.trainable_variables)
        self.low_actor_optimizer.apply_gradients(zip(actor_gradients, self.low_actor.trainable_variables))

        # Update Critic
        with tf.GradientTape() as tape:
            q_values = self.low_critic([states, actions])
            mse_loss = self.mse(q_values, targets)
        gradients = tape.gradient(mse_loss, self.low_critic.trainable_variables)
        self.low_critic_optimizer.apply_gradients(zip(gradients, self.low_critic.trainable_variables))

        with tf.name_scope("Lower_Policy"):
            with self.summary_writer.as_default():
                tf.summary.scalar('MSE Loss', mse_loss, step=self.low_critic_optimizer.iterations)
                tf.summary.scalar('Policy Loss', policy_loss, step=self.low_critic_optimizer.iterations)
                tf.summary.scalar('Estimated Q Value', tf.reduce_mean(q_values),
                                  step=self.low_critic_optimizer.iterations)

    def update_lower(self):
        if len(self.lower_buffer) >= self.batch_size:
            # Sample random minibatch of N transitions
            states, actions, rewards, next_states = self.lower_buffer.sample(self.batch_size)
            rewards = rewards.reshape(-1, 1)

            # Set the target for learning
            target_action_preds = self.low_target_actor(next_states)
            target_q_values = self.low_target_critic([next_states, target_action_preds])
            targets = rewards + self.gamma * target_q_values

            # Update actor and critic networks
            self.train_step_low(states, actions, targets)

            # Update target networks
            polyak_average(self.low_actor.variables, self.low_target_actor.variables, self.polyak)
            polyak_average(self.low_critic.variables, self.low_target_critic.variables, self.polyak)

    def update_higher(self):
        if len(self.higher_buffer) >= self.batch_size:
            states, goals, actions, rewards, next_states = self.higher_buffer.sample(self.batch_size)
            rewards = rewards.reshape((-1, 1))

            states, goals = self.off_policy_correct(states=tf.convert_to_tensor(states, dtype=tf.float32),
                                                    goals=tf.convert_to_tensor(goals, dtype=tf.float32),
                                                    actions=tf.convert_to_tensor(actions, dtype=tf.float32),
                                                    new_states=tf.convert_to_tensor(next_states, dtype=tf.float32))

            # Set the target for learning
            target_goal_preds = self.high_target_actor(next_states)
            target_q_values = self.high_target_critic([next_states, target_goal_preds])
            targets = rewards + self.gamma * target_q_values

            # Update actor and critic networks. (goals are actions for the higher policy)
            self.train_step_high(states, goals, targets)

            # Update target networks
            polyak_average(self.high_actor.variables, self.high_target_actor.variables, self.polyak)
            polyak_average(self.high_critic.variables, self.high_target_critic.variables, self.polyak)

    def learn(self):
        # Collect experiences s_t, g_t, a_t, R_t
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

                    self.low_actor.save_weights(f'{self.save_path}/low/actor')
                    self.low_critic.save_weights(f'{self.save_path}/low/critic')
                    self.high_actor.save_weights(f'{self.save_path}/high/actor')
                    self.high_critic.save_weights(f'{self.save_path}/high/critic')

            obs = self.env.reset()
            goal = self.get_goal(obs.reshape((1, -1)), noise=True).flatten()
            higher_goal = goal
            higher_obs = []
            higher_actions = []
            higher_reward = 0
            episode_reward = 0
            episode_intrinsic_rewards = 0
            ep_len = 0
            c = 0

            done = False
            while not done:
                if self.render:
                    self.env.render()
                action = self.act(obs.reshape((1, -1)), goal.reshape((1, -1)), noise=True).flatten()
                new_obs, rew, done, info = self.env.step(action)
                new_obs = new_obs.flatten()
                new_goal = self.goal_transition(obs, goal, new_obs)
                episode_reward += rew

                # Goals are treated as additional state information for the low level
                # policy. Store transitions in respective replay buffers
                intrinsic_reward = self.intrinsic_reward(obs, goal, new_obs) * self.low_rew_scale
                self.lower_buffer.add((np.concatenate((obs, goal)), action,
                                       intrinsic_reward,
                                       np.concatenate((new_obs, new_goal)),))
                episode_intrinsic_rewards += intrinsic_reward

                self.update_lower()

                # Fill lists for single higher level transition
                higher_obs.append(obs)
                higher_actions.append(action)
                higher_reward += self.high_rew_scale * rew

                # Only add transitions to the high level replay buffer every c steps
                c += 1
                if c == self.c or done:
                    # Need all higher level transitions to be the same length
                    # fill the rest of this transition with zeros
                    while c < self.c:
                        higher_obs.append(np.full(self.env.observation_space.shape, 0))
                        higher_actions.append(np.full(self.env.action_space.shape, 0))
                        c += 1
                    self.higher_buffer.add((higher_obs, higher_goal, higher_actions, higher_reward, new_obs))

                    self.update_higher()
                    c = 0
                    higher_obs = []
                    higher_actions = []
                    higher_reward = 0
                    goal = self.get_goal(new_obs.reshape((1, -1)), noise=True).flatten()
                    higher_goal = goal

                obs = new_obs
                goal = new_goal

            with tf.name_scope('Episodic Information'):
                with self.summary_writer.as_default():
                    tf.summary.scalar(f'Episode Environment Reward', episode_reward, step=ep)
                    tf.summary.scalar(f'Episode Intrinsic Reward', episode_intrinsic_rewards, step=ep)

            self.rewards.append(episode_reward)
            total_steps += ep_len


def main(argv):
    env = gym.make(FLAGS.env_name)
    hiro = HIRO(env,
                gamma=FLAGS.gamma,
                polyak=FLAGS.polyak,
                high_act_noise=FLAGS.high_act_noise,
                low_act_noise=FLAGS.low_act_noise,
                high_rew_scale=FLAGS.high_rew_scale,
                low_rew_scale=FLAGS.low_rew_scale,
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
                c=FLAGS.c
                )
    hiro.learn()


if __name__ == '__main__':
    app.run(main)