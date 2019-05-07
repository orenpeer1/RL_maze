import random


def random_throw_agent(self):
    start_cell = random.choice(self.environment.cells)
    state = self.environment.reset(start_cell)
    return state

# def random_throw_agent_MA(self):
#     start_cell = random.choice(self.environment.cells)
#     state = self.environment.reset(start_cell, self.agent_idx)
#     return state
#
