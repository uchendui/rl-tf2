from tensorflow.keras import layers, Model


class Actor(Model):
    def __init__(self, action_dim, action_range, **kwargs):
        """
        Model for the actor in TD3. The architecture is specified in the paper.
        Args:
            action_dim (int): Number of dimensions in the actor vector
            action_range (np.array): Sequence of ranges for each dimension in the action vector
            **kwargs (dict): arbitrary keyword arguments
        """
        super(Actor, self).__init__(kwargs)
        self.layer_1 = layers.Dense(400, activation='relu', name='actor_1')
        self.layer_2 = layers.Dense(300, activation='relu', name='actor_2')
        self.action_prediction = layers.Dense(action_dim,
                                              activation='tanh',
                                              name='actor_action_pred')
        self.action_range = action_range

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
        return self.action_range * self.action_prediction(out)


class Critic(Model):
    def __init__(self, **kwargs):
        """
        Model for the critic (Q function) in TD3. The architecture is specified in the paper.
        Args:
            **kwargs (dict): arbitrary keyword arguments.
        """
        super(Critic, self).__init__(kwargs)
        self.q1 = layers.Dense(400, activation='relu', name='critic_1')
        self.q2 = layers.Dense(300, activation='relu', name='critic_2')
        self.q_pred = layers.Dense(1, name='critic_q_pred')
        self.concat = layers.Concatenate()

    def call(self, inputs, **kwargs):
        """
        Runs a forward pass of the model.
        Args:
            inputs (list): A list of two elements with structure [states, actions] for estimating the Q-value
            **kwargs (dict): arbitrary keyword arguments

        Returns:
            Q-value prediction
        """
        out = self.concat(inputs)
        out = self.q1(out)
        out = self.q2(out)
        q_pred = self.q_pred(out)
        return q_pred


def create_actor_critic(state_dim, action_dim, action_range):
    state_in = layers.Input(shape=(state_dim,), )
    action_in = layers.Input(shape=(action_dim,), )

    actor_output = Actor(action_dim=action_dim, action_range=action_range)(state_in)
    critic_1_out = Critic()([state_in, action_in])
    critic_2_out = Critic()([state_in, action_in])

    actor = Model(inputs=state_in, outputs=actor_output)
    critic_1 = Model(inputs=[state_in, action_in], outputs=critic_1_out)
    critic_2 = Model(inputs=[state_in, action_in], outputs=critic_2_out)
    return actor, critic_1, critic_2
