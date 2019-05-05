import matplotlib.pyplot as plt
import numpy as np
import random
CELL_EMPTY = 0  # indicates empty cell where the agent can move to
CELL_OCCUPIED = 1  # indicates cell which contains a wall and cannot be entered
CELL_CURRENT = 2  # indicates current cell of the agent

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


class Maze:

    def __init__(self, maze, start_cell=(0, 0), exit_cell=None, num_agents=1):

        self.maze_map = maze.maze_map
        self.__minimum_reward = -0.5 * self.maze_map.size  # stop game if accumulated reward is below this threshold
        self.action_rand = 0.01
        self.max_reward = 1
        self.wall_reward = -0.0

        self.actions = [MOVE_LEFT, MOVE_RIGHT, MOVE_UP, MOVE_DOWN]

        self.nrows, self.ncols = self.maze_map.shape
        exit_cell = (self.ncols - 1, self.nrows - 1) if exit_cell is None else exit_cell

        self.__exit_cell = exit_cell
        self.__previous_cell = self.current_cell = start_cell
        self.cells = [(col, row) for col in range(self.ncols) for row in range(self.nrows)]
        # self.empty = [(col, row) for col in range(ncols) for row in range(nrows) if self.maze[row, col] == CELL_EMPTY]
        self.cells.remove(exit_cell)


        self.__render = "nothing"
        self.ax1 = None  # axes for rendering the moves
        self.ax2 = None  # axes for rendering the best action per cell

        self.reset(start_cell)

    def plot_walls(self):
        nrows, ncols = self.maze_map.shape
        for row in range(nrows):
            for col in range(ncols):
                curr_cell = self.maze_map[col][row]
                for wall in curr_cell.walls.items():
                    if wall[1]:
                        if wall[0] is "E":
                            self.ax1.plot(*zip(*[(col + 0.5, row - 0.5), (col + 0.5, row + 0.5)]), "k-", linewidth="6")
                        if wall[0] is "W":
                            self.ax1.plot(*zip(*[(col - 0.5, row - 0.5), (col - 0.5, row + 0.5)]), "k-", linewidth="6")
                        if wall[0] is "N":
                            self.ax1.plot(*zip(*[(col - 0.5, row - 0.5), (col + 0.5, row - 0.5)]), "k-", linewidth="6")
                        if wall[0] is "S":
                            self.ax1.plot(*zip(*[(col - 0.5, row + 0.5), (col + 0.5, row + 0.5)]), "k-", linewidth="6")

    def reset(self, start_cell=(0, 0)):
        """ Reset the maze to its initial state and place the agent at start_cell.

            :param tuple start_cell: Here the agent starts its journey through the maze (optional, else upper left).
            :return: New state after reset.
        """

        self.__previous_cell = self.current_cell = start_cell
        self.__total_reward = 0.0  # accumulated reward
        self.__visited = set()  # a set() only stores unique values

        if self.__render in ("training", "moves"):
        # render the maze
            nrows, ncols = self.maze_map.shape
            self.ax1.clear()
            self.ax1.set_xticks(np.arange(0.5, nrows, step=1))
            self.ax1.set_xticklabels([])
            self.ax1.set_yticks(np.arange(0.5, ncols, step=1))
            self.ax1.set_yticklabels([])
            self.ax1.grid(True)
            self.ax1.plot(*self.current_cell, "rs", markersize=25)  # start is a big red square
            self.ax1.plot(*self.__exit_cell, "gs", markersize=25)  # exit is a big green square
            self.ax1.imshow(np.zeros(self.maze_map.shape), cmap="binary")
            # plt.pause(0.001)  # replaced by the two lines below
            self.ax1.get_figure().canvas.draw()
            self.ax1.get_figure().canvas.flush_events()
            self.plot_walls()

        return self.__observe()

    def __draw(self):
        """ Draw a line from the agents previous to its current cell. """
        self.ax1.plot(*zip(*[self.__previous_cell, self.current_cell]), "bo-")  # previous cells are blue dots
        self.ax1.plot(*self.current_cell, "ro")  # current cell is a red dot
        # plt.pause(0.001)  # replaced by the two lines below
        self.ax1.get_figure().canvas.draw()
        self.ax1.get_figure().canvas.flush_events()

    def render(self, content="nothing"):
        """ Define what will be rendered during play and/or training.

            :param str content: "nothing", "training" (moves and q), "moves"
        """
        if content not in ("nothing", "training", "moves"):
            raise ValueError("unexpected content: {}".format(content))

        self.__render = content
        if content == "nothing":
            if self.ax1:
                self.ax1.get_figure().close()
                self.ax1 = None
            if self.ax2:
                self.ax2.get_figure().close()
                self.ax2 = None
        if content == "training":
            if self.ax2 is None:
                fig, self.ax2 = plt.subplots(1, 1, tight_layout=True)
                fig.canvas.set_window_title("Best move")
                self.ax2.set_axis_off()
                self.render_q(None)
        if content in ("moves", "training"):
            # if self.__ax1 is None:
                fig, self.ax1 = plt.subplots(1, 1, tight_layout=True)
                fig.canvas.set_window_title("Maze")

        plt.show(block=False)

    def step(self, action):
        """ Move the agent according to 'action' and return the new state, reward and game status.

            :param int action: The agent will move in this direction.
            :return: state, reward, status
        """
        # if np.random.random() < self.action_rand:
        #     action = random.choice(self.actions)
        if np.random.random() < self.action_rand:
            reward = 100
            state = self.__observe()
            status = "FAKE_REWARD"
            return state, reward, status

        reward = self.__execute(action)
        self.__total_reward += reward
        status = self.__status()
        state = self.__observe()
        return state, reward, status

    def __execute(self, action):
        """ Execute action and collect the reward or penalty.

            :param int action: The agent will move in this direction.
            :return float: Reward or penalty after the action is done.
        """
        possible_actions = self.possible_actions(self.current_cell)

        if not possible_actions:
            reward = self.__minimum_reward - 1  # cannot move anywhere, force end of game
        elif action in possible_actions:
            col, row = self.current_cell
            if action == MOVE_LEFT:
                col -= 1
            elif action == MOVE_UP:
                row -= 1
            if action == MOVE_RIGHT:
                col += 1
            elif action == MOVE_DOWN:
                row += 1

            self.__previous_cell = self.current_cell
            self.current_cell = (col, row)

            if self.__render != "nothing":
                self.__draw()

            if self.current_cell == self.__exit_cell:
                reward = self.max_reward  # maximum reward for reaching the exit cell
            elif self.current_cell in self.__visited:
                reward = -0.0  # penalty for returning to a cell which was visited earlier
            else:
                reward = -0.0  # penalty for a move which did not result in finding the exit cell

            self.__visited.add(self.current_cell)
        else:
            reward = self.wall_reward  # penalty for trying to enter an occupied cell (= a wall) or moving out of the maze

        return reward

    def possible_actions(self, cell=None):
        """ Create a list with possible actions, avoiding the maze's edges and walls.

            :param tuple cell: Location of the agent (optional, else current cell).
            :return list: All possible actions.
        """
        if cell is None:
            col, row = self.current_cell
        else:
            col, row = cell

        possible_actions = self.actions.copy()  # initially allow all

        # now restrict the initial list by removing impossible actions
        nrows, ncols = self.maze_map.shape
        if row == 0 or (row > 0 and self.maze_map[cell].walls["N"]):
            possible_actions.remove(MOVE_UP)
        if row == nrows - 1 or (row < nrows - 1 and self.maze_map[cell].walls["S"]):
            possible_actions.remove(MOVE_DOWN)

        if col == 0 or (col > 0 and self.maze_map[cell].walls["W"]):
            possible_actions.remove(MOVE_LEFT)
        if col == ncols - 1 or (col < ncols - 1 and self.maze_map[cell].walls["E"]):
            possible_actions.remove(MOVE_RIGHT)

        return possible_actions

    def __status(self):
        """ Determine the game status.

            :return str: Current game status (win/lose/playing).
        """
        if self.current_cell == self.__exit_cell:
            return "win"

        if self.__total_reward < self.__minimum_reward:  # force end of game after to much loss
            return "lose"

        return "playing"

    def __observe(self):
        """ Return the state of the maze - in this example the agents current location.

            :return numpy.array [1][2]: Agents current location.
        """
        return self.current_cell

    def play(self, model, start_cell=(0, 0)):
        """ Play a single game, choosing the next move based a prediction from 'model'.

            :param class AbstractModel model: The prediction model to use.
            :param tuple start_cell: Agents initial cell (optional, else upper left).
            :return str: "win" or "lose"
        """
        self.reset(start_cell)

        state = self.__observe()

        while True:
            action = model.predict(state=state)
            state, reward, status = self.step(action)
            if status in ("win", "lose"):
                return status

    def win_all(self, model):
        """ Check if the model wins from all possible starting cells. """
        previous = self.__render
        self.__render = "nothing"  # avoid rendering anything during execution of win_all()

        win = 0
        lose = 0

        for cell in self.cells:
            if self.play(model, cell) == "win":
                win += 1
            else:
                lose += 1


        self.__render = previous

        result = True if lose == 0 else False
        return result, win / (win + lose)

    def render_q(self, model):
        """ Render the recommended action for each cell. """
        if self.__render != "training":
            return

        nrows, ncols = self.maze_map.shape

        self.ax2.clear()
        self.ax2.set_xticks(np.arange(0.5, nrows, step=1))
        self.ax2.set_xticklabels([])
        self.ax2.set_yticks(np.arange(0.5, ncols, step=1))
        self.ax2.set_yticklabels([])
        self.ax2.grid(True)
        self.ax2.plot(*self.__exit_cell, "gs", markersize=25)  # exit is a big green square

        for cell in self.cells:
            state = cell
            q = model.q(state) if model is not None else [0, 0, 0, 0]
            a = np.nonzero(q == np.max(q))[0]

            for action in a:
                dx = 0
                dy = 0
                if action == 0:  # left
                    dx = -0.2
                if action == 1:  # right
                    dx = +0.2
                if action == 2:  # up
                    dy = -0.2
                if action == 3:  # down
                    dy = 0.2

                self.ax2.arrow(*cell, dx, dy, head_width=0.2, head_length=0.1)

        self.ax2.imshow(self.maze, cmap="binary")
        self.ax2.get_figure().canvas.draw()
        # plt.pause(0.001)

    def plot_walls(self):
        nrows, ncols = self.maze_map.shape
        for row in range(nrows):
            for col in range(ncols):
                curr_cell = self.maze_map[col][row]
                for wall in curr_cell.walls.items():
                    if wall[1]:
                        if wall[0] is "E":
                            self.ax1.plot(*zip(*[(col + 0.5, row - 0.5), (col + 0.5, row + 0.5)]), "k-", linewidth="6")
                        if wall[0] is "W":
                            self.ax1.plot(*zip(*[(col - 0.5, row - 0.5), (col - 0.5, row + 0.5)]), "k-", linewidth="6")
                        if wall[0] is "N":
                            self.ax1.plot(*zip(*[(col - 0.5, row - 0.5), (col + 0.5, row - 0.5)]), "k-", linewidth="6")
                        if wall[0] is "S":
                            self.ax1.plot(*zip(*[(col - 0.5, row + 0.5), (col + 0.5, row + 0.5)]), "k-", linewidth="6")

    def draw(self):
        nrows, ncols = self.maze_map.shape
        fig, self.ax1 = plt.subplots(1, 1, tight_layout=True)
        fig.canvas.set_window_title("Maze")
        self.ax1.clear()
        self.ax1.set_xticks(np.arange(0.5, nrows, step=1))
        self.ax1.set_xticklabels([])
        self.ax1.set_yticks(np.arange(0.5, ncols, step=1))
        self.ax1.set_yticklabels([])
        self.ax1.grid(True)
        self.ax1.imshow(np.zeros(self.maze_map.shape), cmap="binary")
        # plt.pause(0.001)  # replaced by the two lines below
        self.ax1.get_figure().canvas.draw()
        self.ax1.get_figure().canvas.flush_events()
        self.plot_walls()