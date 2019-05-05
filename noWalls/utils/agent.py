import random
from datetime import datetime
import numpy as np

# all actions the agent can choose, plus a dictionary for textual representation
MOVE_LEFT = 0
MOVE_RIGHT = 1
MOVE_UP = 2
MOVE_DOWN = 3

actions = {
    MOVE_LEFT: "move left",
    MOVE_RIGHT: "move right",
    MOVE_UP: "move up",
    MOVE_DOWN: "move down"
}


class agent():
    def __init__(self, game, **kwargs):
        self.environment = game
        # self.name = kwargs.get("name")
        self.learner = kwargs.get("learner")
        self.Q = self.learner.get_Q()
        self.state = random_throw_agent(self, start_list)

    def update_Q(self, Q):
        self.Q = Q

    def make_action(self):
        action = self.predict(state)
        next_state, reward, status = self.environment.step(action)


