# # import logging
# import random
# from datetime import datetime
# import numpy as np
#
#
#
# class QTableModel_regular():
#     """ Tabular Q-learning prediction model.
#
#         For every state (here: the agents current location ) the value for each of the actions is stored in a table.
#         The key for this table is (state + action). Initially all values are 0. When playing training games
#         after every move the value in the table is updated based on the reward gained after making the move. Training
#         ends after a fixed number of games, or earlier if a stopping criterion is reached (here: a 100% win rate).
#
#         :param class Maze game: Maze game object.
#     """
#
#     def __init__(self, game, **kwargs):
#         self.environment = game
#         self.name = kwargs.get("name", "model")
#         self.Q = dict()  # table with value for (state, action) combination
#
#     def train(self, stop_at_convergence=False, **kwargs):
#         """ Hyperparameters:
#
#             :keyword float discount: (gamma) preference for future rewards (0 = not at all, 1 = only)
#             :keyword float exploration_rate: (epsilon) 0 = preference for exploring (0 = not at all, 1 = only)
#             :keyword float exploration_decay: exploration rate reduction after each random step (<= 1, 1 = no at all)
#             :keyword float learning_rate: (alpha) preference for using new knowledge (0 = not at all, 1 = only)
#             :keyword int episodes: number of training games to play
#             :return int, datetime: number of training episodes, total time spent
#         """
#         discount = kwargs.get("discount", 0.90)
#         exploration_rate = kwargs.get("exploration_rate", 0.10)
#         exploration_decay = kwargs.get("exploration_decay", 0.995)  # % reduction per step = 100 - exploration decay
#         learning_rate = kwargs.get("learning_rate", 0.10)
#         episodes = kwargs.get("episodes", 1000)
#
#         # variables for reporting purposes
#         cumulative_reward = 0
#         self.cumulative_reward_history = []
#         self.win_history = []
#
#         start_list = list()
#         start_time = datetime.now()
#
#         # for episode in range(1, episodes + 1):
#         win_rate = 0
#         episode = 0
#         # for episode in range(1, episodes + 1)
#         while (win_rate != 1):
#             episode += 1
#             # optimization: make sure to start from all possible cells
#             if not start_list:
#                 start_list = self.environment.empty.copy()
#             start_cell = random.choice(start_list)
#             start_list.remove(start_cell)
#
#             state = self.environment.reset(start_cell)
#             state = tuple(state.flatten())
#
#             while True:
#                 # choose action epsilon greedy (off-policy, instead of only using the learned policy)
#                 if np.random.random() < exploration_rate:
#                     action = random.choice(self.environment.actions)
#                 else:
#                     action = self.predict(state)
#
#                 next_state, reward, status = self.environment.step(action)
#                 next_state = tuple(next_state.flatten())
#
#                 cumulative_reward += reward
#
#                 if (state, action) not in self.Q.keys():  # ensure value exists for (state, action) to avoid a KeyError
#                     self.Q[(state, action)] = 0.0
#
#                 max_next_Q = max([self.Q.get((next_state, a), 0.0) for a in self.environment.actions])
#
#                 self.Q[(state, action)] += learning_rate * (reward + discount * max_next_Q - self.Q[(state, action)])
#
#                 if status in ("win", "lose"):  # terminal state reached, stop training episode
#                     break
#
#                 state = next_state
#
#                 self.environment.render_q(self)
#
#             self.cumulative_reward_history.append(cumulative_reward)
#
#
#             if episode % 20 == 0:
#                 # check if the current model wins from all starting cells
#                 # can only do this if there is a finite number of starting states
#                 print("episode " + str(episode) + " is done.")
#                 w_all, win_rate = self.environment.win_all(self)
#                 self.win_history.append((episode, win_rate))
#                 if w_all is True and stop_at_convergence is True:
#                     break
#
#             exploration_rate *= exploration_decay  # explore less as training progresses
#
#         return self.cumulative_reward_history, self.win_history, episode, datetime.now() - start_time
#
#     def q(self, state):
#         """ Get q values for all actions for a certain state. """
#         if type(state) == np.ndarray:
#             state = tuple(state.flatten())
#
#         return np.array([self.Q.get((state, action), 0.0) for action in self.environment.actions])
#
#     def predict(self, state):
#         """ Policy: choose the action with the highest value from the Q-table.
#             Random choice if multiple actions have the same (max) value.
#
#             :param np.ndarray state: Game state.
#             :return int: Chosen action.
#         """
#         q = self.q(state)
#
#         # logging.debug("q[] = {}".format(q))
#
#         # actions = np.nonzero(q == np.max(q))[0]  # get index of the action(s) with the max value
#         # return random.choice(actions)
#         actions = np.argmax(q) # get index of the action(s) with the max value
#         return actions
#
#
#     def save_model(self, path=".//", model_name="model"):
#         model_data = {"model_name": self.name, "environmant": self.environment, "Q": self.Q, "win_history": self.win_history,
#                       "cumulative_reward_history": self.cumulative_reward_history}
#         np.save(path + model_name, model_data)
#
#     def load_model(self, path=".//", model_name="model"):
#         model_data = np.load(path + model_name)
#
#
#     def restore_train_results(self):
#         return self.cumulative_reward_history, self.win_history
#
