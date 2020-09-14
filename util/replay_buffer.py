import pickle
import random
import numpy as np

from collections import deque


class Buffer:
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.buffer = []

    def __len__(self):
        return self.size()

    def size(self):
        """Returns the number of transitions in the buffer"""
        return len(self.buffer)

    def sample(self, n):
        raise NotImplementedError

    def save(self, path):
        pickle.dump(self, open(path, 'wb'))

    def load(self, path):
        bf = np.load(path)
        self.buffer = bf


class ReplayBuffer(Buffer):
    def __init__(self, capacity, tuple_length=5):
        """Represents a replay buffer that stores transitions. (prev_state, action, reward, next_state)

        Args:
            capacity: Number of samples the replay buffer can hold.
            tuple_length: Length of transition tuples. In some cases, we will need to
                store more than just (s, a, r, s', d)
        """
        super().__init__(capacity=capacity)
        self.tuple_length = tuple_length
        self.buffer = deque()

    def sample(self, n):
        """Samples transitions from the replay buffer.

        Args:
            n: Number of transitions to sample

        Returns:
            samples: array of (states, actions, rewards, next_states, done)
        """
        assert (n <= len(self.buffer)), "Sample amount larger than replay buffer size"
        sample = random.sample(self.buffer, n)
        return list(map(np.array, zip(*sample)))

    def add(self, transition):
        """ Adds a transition of the form (prev_state, action, reward, next_state) to the replay buffer

        Args:
            transition:  array containing transition (obs, action, reward, new_obs, done)
        """
        assert (len(
            transition) == self.tuple_length), (f'Transitions passed to replay buffer '
                                                f'must be of length {self.tuple_length}')
        if len(self.buffer) >= self.capacity:
            self.buffer.pop()
        self.buffer.appendleft(transition)
