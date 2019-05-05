import random


def random_throw_agent(self, start_list):
    if not start_list:
        start_list = self.environment.cells.copy()
    start_cell = random.choice(start_list)
    start_list.remove(start_cell)
    state = self.environment.reset(start_cell)
    return state
