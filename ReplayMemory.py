from collections import namedtuple
import random

Sample = namedtuple("Sample", ("state", "action", "reward", "next_state",))


class ReplayMemory:

    def __init__(self, size, batch_size):
        self.size = size
        self.memory = [None for _ in range(self.size)]
        self.pos, self.effect_size = 0, 0
        self.batch_size = batch_size

    def push(self, *sample_args):
        self.memory[self.pos] = Sample(*sample_args)
        self.pos = (self.pos + 1) % self.size
        if self.effect_size < self.size:
            self.effect_size += 1
        return

    def create_batch(self):
        if self.effect_size < self.batch_size:
            return None
        if self.effect_size < self.size:
            return random.sample(self.memory[:self.effect_size], self.batch_size)
        return random.sample(self.memory, self.batch_size)

