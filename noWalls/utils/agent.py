import random
from datetime import datetime
import numpy as np
from utils.functions import *

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
        """
        init an agent
        :param game: maze object
        :param kwargs: learner: qTable learner to send samples to.
        """
        self.environment = game
        # self.name = kwargs.get("name")
        self.learner = kwargs.get("learner")
        self.agent_idx = kwargs.get("agent_idx")
        self.Q = self.learner.get_Q()
        # self.random_throw_agent()
        self.state = (None, None)

    def update_Q(self, Q):
        self.Q = np.copy(Q)

    def make_action(self):
        """
        perform an action according to current Q table.
        :return: next_state, reward, status
        """
        action = self.predict()
        next_state, reward, status = self.environment.step(action, self.agent_idx)
        if status is "win":
            self.random_throw_agent()
        else:
            self.state = next_state
        return action, next_state, reward, status

    def predict(self):
        return np.argmax(self.Q[self.state])

    def random_throw_agent(self):
        start_cell = random.choice(self.environment.cells)
        self.state = self.environment.reset(start_cell, self.agent_idx)

    def get_state(self):
        return self.state


