# import logging
import random
from datetime import datetime
import numpy as np
from utils.functions import *
from utils.agent import agent as Agent



# all actions the agent can choose, plus a dictionary for textual representation
MOVE_LEFT = 0
MOVE_RIGHT = 1
MOVE_UP = 2
MOVE_DOWN = 3

possible_actions = {
    MOVE_LEFT: "move left",
    MOVE_RIGHT: "move right",
    MOVE_UP: "move up",
    MOVE_DOWN: "move down"
}


class QTableModel_medianMA():

    def __init__(self, game, **kwargs):
        self.environment = game
        self.name = kwargs.get("name", "model")
        self.num_agents = kwargs.get("num_agents")

    def train(self, stop_at_convergence=False, **kwargs):

        def init_U(self):
            U = {}
            for col in range(self.environment.ncols):
                for row in range(self.environment.nrows):
                    state = (col, row)
                    for action in range(4):
                        U[(state, action)] = []
            return U

        self.discount = kwargs.get("discount", 0.98)
        episodes = kwargs.get("episodes", 10)
        num_steps = 500
        k = kwargs.get("k")
        km = kwargs.get("km")
        # Median params:
        self.Q_max = self.environment.max_reward / (1 - self.discount)
        # eps_b = eps_a * np.sqrt(k)
        eps_b = 0.01 * self.Q_max
        eps_a = (1/np.sqrt(k)) * eps_b     # try few values.
        ny, nx = self.environment.maze_map.shape

        # initilize Q table, and U, U_max sets - Alg. line 2-3 # Q is 3D ndarray: (nx, ny, A) or (16, 16, 4)
        U, U_new = init_U(self), init_U(self)
        self.Q = np.ones((nx, ny, len(self.environment.actions))) * self.Q_max
        self.agents = []
        for agent_idx in range(self.num_agents):
            self.agents.append(Agent(game=self.environment, learner=self, agent_idx=agent_idx))

        # variables for reporting purposes
        cumulative_reward = 0
        self.cumulative_reward_history = []
        self.episode_reward_history = []
        self.win_history = []

        start_time = datetime.now()
        episode = 1
        update_stops = np.array(range(0, k + km, km))
        def BQ(self):
            bq = np.ones(self.Q.shape) * self.Q_max
            for col in range(self.environment.ncols):
                for row in range(self.environment.nrows):
                    for action in range(4):
                        state = (col, row)
                        U_sa = U[(state, action)]
                        size_U_sa = len(U_sa)
                        if size_U_sa > km:
                            chunk_size = int(size_U_sa/km)
                            km_chunks = [U_sa[sample: sample + chunk_size] for sample in range(0, size_U_sa, chunk_size)]

                            g = []
                            for chunk in km_chunks:
                                gj = []
                                for samp in chunk:
                                    gj.append(samp[0] + self.discount * max(self.Q[samp[1]]))
                                g.append(sum(gj))
                            bq[state][action] = min(self.Q_max, ((eps_b/np.sqrt(size_U_sa)) + km * np.median(g)/size_U_sa))
            return bq

        def iterateQ(self):
            max_num_iters = 50
            iter = 0
            old_Q = self.Q
            self.Q = BQ(self)
            while ((np.max(self.Q - old_Q) > eps_a) or (np.max(old_Q - self.Q) > eps_a)) and (iter < max_num_iters):
                old_Q = self.Q.copy()
                self.Q = BQ(self)
                iter += 1

        while episode < episodes:
            episode_reward = 0
            steps = 0
            # we can start from all possible cells
            self.random_throw_agents()
            # self.get_agents_states()
            while steps < num_steps:
                states = self.get_agents_states()
                # lines 5+6 (Alg.)
                actions, next_states, rewards, statuses = self.do_action_all_agents()

                for state, action, next_state, reward, status in zip(states, actions, next_states, rewards, statuses):
                    if status is not "FAKE_REWARD":
                        episode_reward += reward
                        cumulative_reward += reward

                    if len(U[(state, action)]) < k:
                        # line 8 in Alg. - add to U_new
                        U_new[(state, action)].append((reward, next_state))
                        if len(U_new[(state, action)]) > k or len(U[(state, action)]) > k:
                            a = 1
                        # if (len(U_new[(state, action)]) > len(U[(state, action)])) and \
                        #         (np.log2(len(U_new[(state, action)])/km).is_integer()):     # Or, number of samples?!
                        if (len(U_new[(state, action)]) > len(U[(state, action)])) and \
                            len(U_new[(state, action)]) in update_stops:     # Or, number of samples?!

                            if len(U_new[(state, action)]) > k or len(U[(state, action)]) > k:
                                a = 1

                            U[(state, action)] = U_new[(state, action)]
                            U_new[(state, action)] = []     # OR - what about the old samples we had??? why we don't append the new samples to U?
                        iterateQ(self)   # stuff in lines 13-15 : OR - for few agents ONE TIME!
                        self.update_agents_Q()

                steps += 1

            self.cumulative_reward_history.append(cumulative_reward)
            self.episode_reward_history.append(episode_reward)

            # if episode % 1 == 0:
            # check if the current model wins from all starting cells
            # can only do this if there is a finite number of starting states
            # w_all, win_rate = self.environment.win_all(self)
            print("episode " + str(episode) + " is done. episode_reward = " + str(episode_reward))
            # self.win_history.append((episode, win_rate))
            # if w_all is True and stop_at_convergence is True:
            #     break
            episode += 1

        # return self.cumulative_reward_history, self.win_history, episode, datetime.now() - start_time
        return self.cumulative_reward_history, self.episode_reward_history, datetime.now() - start_time

    def q(self, state):
        """ Get q values for all actions for a certain state. """
        return self.Q[state]

    def predict(self, state):
        """ Policy: choose the action with the highest value from the Q-table.

            :param np.ndarray state: Game state.
            :return int: Chosen action.
        """

        return np.argmax(self.Q[state])


    def restore_train_results(self):
        return self.cumulative_reward_history, self.win_history

    def random_throw_agents(self):
        for agent in self.agents:
            agent.random_throw_agent()

    def do_action_all_agents(self):
        actions, next_states, rewards, statuses = [], [], [], []
        for agent in self.agents:
            action, next_state, reward, status = agent.make_action()
            next_states.append(next_state)
            rewards.append(reward)
            statuses.append(status)
            actions.append(action)
        return actions, next_states, rewards, statuses

    def get_Q(self):
        return np.copy(self.Q)

    def update_agents_Q(self):
        for agent in self.agents:
            agent.update_Q(self.Q)

    def get_agents_states(self):
        states = []
        for agent in self.agents:
            states.append(agent.get_state())
        return states

    def save_model(self, path=".//", model_name="model"):
        model_data = {"model_name": self.name, "environmant": self.environment, "Q": self.Q, "win_history": self.win_history,
                      "cumulative_reward_history": self.cumulative_reward_history}
        np.save(path + model_name, model_data)

    def load_model(self, path=".//", model_name="model"):
        model_data = np.load(path + model_name)


