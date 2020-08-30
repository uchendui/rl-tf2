import gym
import numpy as np
import tensorflow as tf
from util.replay_buffer import ReplayBuffer
from tensorflow.keras import layers, Model


# tf.config.experimental_functions_run_eagerly()


@tf.function
def copy_weights(vars1, vars2):
    for a, b in zip(vars1, vars2):
        tf.compat.v1.assign(b, a)


class QNetwork(Model):
    def __init__(self, out_dim, name='q_network'):
        super(QNetwork, self).__init__()
        self.q1 = layers.Dense(32, activation='relu', name='q_pred_1')
        self.q2 = layers.Dense(32, activation='relu', name='q_pred_2')
        self.q_value = layers.Dense(out_dim, activation='linear', name='q_pred')

    def call(self, state, **kwargs):
        action_pred = self.q1(state)
        action_pred = self.q2(action_pred)
        action_pred = self.q_value(action_pred)
        return action_pred


class DQN:
    def __init__(self, env):
        self.gamma = 0.99
        self.render = False
        self.batch_size = 32
        self.print_freq = 1
        self.q_lr = 1e-3
        self.max_eps = 1.0
        self.min_eps = 0.1
        self.eps_decay_rate = -1e-3
        self.buffer = ReplayBuffer(5000)
        self.max_steps = 50000
        self.target_update = 1000
        self.model = QNetwork(env.action_space.n, name='q_network')
        self.target = QNetwork(env.action_space.n, name='target_network')
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.q_lr)
        self.summary_writer = tf.summary.create_file_writer('logs/')
        self.env = env
        self.max_episode_len = self.env.spec.max_episode_steps
        self.rewards = []

    def act(self, state):
        return np.argmax(self.model(state))

    @tf.function
    def train_step(self, states, indices, targets):
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

                    ep_len = 0
                    rand_actions = 0


def main():
    env = gym.make('CartPole-v0')
    dqn = DQN(env)
    dqn.learn()


if __name__ == '__main__':
    main()
