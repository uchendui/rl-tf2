# TODO: make a whole bunch of functions that are deocration with tf.function for training the actor and critic models
#   right now we have a problem where the tf.function can only act on one model and optimizer 😢😭
import gym
import numpy as np
import tensorflow as tf

from absl import flags, app
from util import ReplayBuffer
from util.tf import polyak_average
from td3.model import create_actor_critic

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
flags.DEFINE_integer('d', 2, 'The actor and target networks will be updated every d steps')
flags.DEFINE_boolean('render', False, 'Render the environment during training')
flags.DEFINE_integer('seed', 1234, 'Random seed for reproducible results')


class HIRO:
    def __init__(self,
                 env,
                 gamma=0.99,
                 polyak=0.995,
                 c=10,
                 d=2,
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
        self.d = d
        self.higher_buffer = ReplayBuffer(buffer_capacity, tuple_length=5)
        self.lower_buffer = ReplayBuffer(buffer_capacity, tuple_length=4)

        self.low_actor, self.low_critic_1, self.low_critic_2 = create_actor_critic(
            state_dim=2 * env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            action_range=env.action_space.high)

        self.low_target_actor, self.low_target_critic_1, self.low_target_critic_2 = create_actor_critic(
            state_dim=2 * env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            action_range=env.action_space.high)

        self.high_actor, self.high_critic_1, self.high_critic_2 = create_actor_critic(
            state_dim=env.observation_space.shape[0],
            action_dim=env.observation_space.shape[0],
            action_range=env.observation_space.high)

        self.high_target_actor, self.high_target_critic_1, self.high_target_critic_2 = create_actor_critic(
            state_dim=env.observation_space.shape[0],
            action_dim=env.observation_space.shape[0],
            action_range=env.observation_space.high)
        self.low_target_actor.set_weights(self.low_actor.get_weights())
        self.low_target_critic_1.set_weights(self.low_critic_1.get_weights())
        self.low_target_critic_2.set_weights(self.low_critic_2.get_weights())
        self.high_target_actor.set_weights(self.high_actor.get_weights())
        self.high_target_critic_1.set_weights(self.high_critic_1.get_weights())
        self.high_target_critic_2.set_weights(self.high_critic_2.get_weights())

        if training:
            self.low_actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.p_lr)
            self.low_critic_1_optimizer = tf.keras.optimizers.Adam(learning_rate=self.q_lr)
            self.low_critic_2_optimizer = tf.keras.optimizers.Adam(learning_rate=self.q_lr)
            self.high_actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.p_lr)
            self.high_critic_1_optimizer = tf.keras.optimizers.Adam(learning_rate=self.q_lr)
            self.high_critic_2_optimizer = tf.keras.optimizers.Adam(learning_rate=self.q_lr)
            self.mse = tf.keras.losses.MeanSquaredError()
            self.summary_writer = tf.summary.create_file_writer(log_dir)

            self.low_actor_train_fn = self.create_train_step_actor_fn(self.low_actor, self.low_critic_1,
                                                                      self.low_actor_optimizer)
            self.low_critic_train_fns = [self.create_train_step_critic_fn(critic=c, optimizer=o) for c, o in
                                         [(self.low_critic_1, self.low_critic_1_optimizer),
                                          (self.low_critic_2, self.low_critic_2_optimizer)]]

            self.high_actor_train_fn = self.create_train_step_actor_fn(self.high_actor, self.high_critic_1,
                                                                       self.high_actor_optimizer)
            self.high_critic_train_fns = [self.create_train_step_critic_fn(critic=c, optimizer=o) for c, o in
                                          [(self.high_critic_1, self.high_critic_1_optimizer),
                                           (self.high_critic_2, self.high_critic_2_optimizer)]]
        if load_path is not None:
            self.low_actor.load_weights(f'{load_path}/low/actor')
            self.low_critic_1.load_weights(f'{load_path}/low/critic_1')
            self.low_critic_2.load_weights(f'{load_path}/low/critic_2')
            self.high_actor.load_weights(f'{load_path}/high/actor')
            self.high_critic_1.load_weights(f'{load_path}/high/critic_1')
            self.high_critic_2.load_weights(f'{load_path}/high/critic_2')

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

        def body(curr_i, curr_goals, s):
            new_goals = tf.concat(
                (curr_goals,
                 tf.reshape(self.goal_transition(s[curr_i - 1], curr_goals[curr_i - 1], s[curr_i]), (1, -1))), axis=0)
            curr_i += 1
            return [curr_i, new_goals, s]

        def condition(curr_i, curr_goals, s):
            return curr_i < s.shape[0] and not (
                    tf.equal(tf.math.count_nonzero(s[curr_i]), 0) and tf.equal(tf.math.count_nonzero(actions[curr_i]),
                                                                               0))

        # If a state-action pair is all zero, then the episode ended before an entire sequence of length c was recorded.
        # We must remove these empty states and actions from the log probability calculation, as they could skew the
        #   argmax computation
        i = tf.constant(1)
        i, goals, states = tf.while_loop(condition, body, [i, goals, states],
                                         shape_invariants=[tf.TensorShape(None), tf.TensorShape([None, goals.shape[1]]),
                                                           states.shape])
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
    def train_step_critics(self, states, actions, rewards, next_states, actor, target_critic_1,
                           target_critic_2, critic_trains_fns, target_noise,
                           scope='Policy'):
        target_goal_preds = actor(next_states)
        target_goal_preds += target_noise

        target_q_values_1 = target_critic_1([next_states, target_goal_preds])
        target_q_values_2 = target_critic_2([next_states, target_goal_preds])

        target_q_values = tf.concat((target_q_values_1, target_q_values_2), axis=1)
        target_q_values = tf.reshape(tf.reduce_min(target_q_values, axis=1), (self.batch_size, -1))
        targets = rewards + self.gamma * target_q_values

        critic_trains_fns[0](states, actions, targets, scope=scope, label='Critic 1')
        critic_trains_fns[1](states, actions, targets, scope=scope, label='Critic 2')

    def create_train_step_actor_fn(self, actor, critic, optimizer):
        @tf.function
        def train_step_actor(states, scope='policy', label='actor'):
            with tf.GradientTape() as tape:
                action_predictions = actor(states)
                q_values = critic([states, action_predictions])
                policy_loss = -tf.reduce_mean(q_values)
            gradients = tape.gradient(policy_loss, actor.trainable_variables)
            optimizer.apply_gradients(zip(gradients, actor.trainable_variables))

            with tf.name_scope(scope):
                with self.summary_writer.as_default():
                    tf.summary.scalar(f'{label} Policy Loss', policy_loss, step=optimizer.iterations)

        return train_step_actor

    def create_train_step_critic_fn(self, critic, optimizer):
        @tf.function
        def train_step_critic(states, actions, targets, scope='Policy', label='Critic'):
            with tf.GradientTape() as tape:
                q_values = critic([states, actions])
                mse_loss = self.mse(q_values, targets)
            gradients = tape.gradient(mse_loss, critic.trainable_variables)
            optimizer.apply_gradients(zip(gradients, critic.trainable_variables))

            with tf.name_scope(scope):
                with self.summary_writer.as_default():
                    tf.summary.scalar(f'{label} MSE Loss', mse_loss, step=optimizer.iterations)
                    tf.summary.scalar(f'{label} Mean Q Values', tf.reduce_mean(q_values), step=optimizer.iterations)

        return train_step_critic

    def update_lower(self):
        if len(self.lower_buffer) >= self.batch_size:
            states, actions, rewards, next_states = self.lower_buffer.sample(self.batch_size)
            rewards = rewards.reshape(-1, 1).astype(np.float32)

            self.train_step_critics(states, actions, rewards, next_states, self.low_actor, self.low_target_critic_1,
                                    self.low_target_critic_2,
                                    self.low_critic_train_fns,
                                    target_noise=tf.random.normal(actions.shape,
                                                                  stddev=0.1 * self.env.action_space.high),
                                    scope='Lower_Policy')

            if self.low_critic_1_optimizer.iterations % self.d == 0:
                self.low_actor_train_fn(states, scope='Lower_Policy', label='Actor')

                # Update target networks
                polyak_average(self.low_actor.variables, self.low_target_actor.variables, self.polyak)
                polyak_average(self.low_critic_1.variables, self.low_target_critic_1.variables, self.polyak)
                polyak_average(self.low_critic_2.variables, self.low_target_critic_2.variables, self.polyak)

    def update_higher(self):
        if len(self.higher_buffer) >= self.batch_size:
            states, goals, actions, rewards, next_states = self.higher_buffer.sample(self.batch_size)
            rewards = rewards.reshape((-1, 1))

            states, goals, actions, rewards, next_states = (tf.convert_to_tensor(states, dtype=tf.float32),
                                                            tf.convert_to_tensor(goals, dtype=tf.float32),
                                                            tf.convert_to_tensor(actions, dtype=tf.float32),
                                                            tf.convert_to_tensor(rewards, dtype=tf.float32),
                                                            tf.convert_to_tensor(next_states, dtype=tf.float32))

            states, goals = self.off_policy_correct(states=states, goals=goals, actions=actions, new_states=next_states)

            self.train_step_critics(states, goals, rewards, next_states, self.high_actor, self.high_target_critic_1,
                                    self.high_target_critic_2,
                                    self.high_critic_train_fns,
                                    target_noise=tf.random.normal(next_states.shape,
                                                                  stddev=0.1 * self.env.observation_space.high),
                                    scope='Higher_Policy')

            if self.high_critic_1_optimizer.iterations % self.d == 0:
                self.high_actor_train_fn(states, scope='Higher_Policy', label='Actor')

                # Update target networks
                polyak_average(self.high_actor.variables, self.high_target_actor.variables, self.polyak)
                polyak_average(self.high_critic_1.variables, self.high_target_critic_1.variables, self.polyak)
                polyak_average(self.high_critic_2.variables, self.high_target_critic_2.variables, self.polyak)

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
                with tf.name_scope('Episodic Information'):
                    with self.summary_writer.as_default():
                        tf.summary.scalar(f'Mean {self.print_freq} Episode Reward', new_mean_reward, step=ep)

                # Model saving inspired by Open AI Baseline implementation
                if (mean_reward is None or new_mean_reward >= mean_reward) and self.save_path is not None:
                    print(f"Saving model due to mean reward increase:{mean_reward} -> {new_mean_reward}")
                    print(f'Location: {self.save_path}')
                    mean_reward = new_mean_reward

                    self.low_actor.save_weights(f'{self.save_path}/low/actor')
                    self.low_critic_1.save_weights(f'{self.save_path}/low/critic_1')
                    self.low_critic_2.save_weights(f'{self.save_path}/low/critic_2')
                    self.high_actor.save_weights(f'{self.save_path}/high/actor')
                    self.high_critic_1.save_weights(f'{self.save_path}/high/critic_1')
                    self.high_critic_2.save_weights(f'{self.save_path}/high/critic_2')

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
                c=FLAGS.c,
                d=FLAGS.d
                )
    hiro.learn()


if __name__ == '__main__':
    app.run(main)
