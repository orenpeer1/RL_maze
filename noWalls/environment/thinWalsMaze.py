import random
import numpy as np
import matplotlib.pyplot as plt


class Cell:
    """A cell in the maze.

    A maze "Cell" is a point in the grid which may be surrounded by walls to
    the north, east, south or west.

    """

    # A wall separates a pair of cells in the N-S or W-E directions.
    wall_pairs = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}

    def __init__(self, x, y):
        """Initialize the cell at (x,y). At first it is surrounded by walls."""

        self.x, self.y = x, y
        self.walls = {'N': True, 'S': True, 'E': True, 'W': True}

    def has_all_walls(self):
        """Does this cell still have all its walls?"""

        return all(self.walls.values())

    def knock_down_wall(self, other, wall):
        """Knock down the wall between cells self and other."""

        self.walls[wall] = False
        other.walls[Cell.wall_pairs[wall]] = False


class Maze_no_walls:
    """A Maze, represented as a grid of cells."""

    def __init__(self, nx, ny, ix=0, iy=0):
        """Initialize the maze grid.
        The maze consists of nx x ny cells and will be constructed starting
        at the cell indexed at (ix, iy).

        """
        self.size = nx * ny
        self.nx, self.ny = nx, ny
        self.ix, self.iy = ix, iy
        self.maze_map = [[Cell(x, y) for y in range(ny)] for x in range(nx)]
        # self.maze_map = [[Cell(x, y) for x in range(nx)] for y in range(ny)]
        self.maze_map = np.array(self.maze_map)

    def cell_at(self, x, y):
        """Return the Cell object at (x,y)."""

        return self.maze_map[x][y]

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


    def find_valid_neighbours(self, cell):
        """Return a list of unvisited neighbours to cell."""

        delta = [('W', (-1,0)),
                 ('E', (1,0)),
                 ('S', (0,1)),
                 ('N', (0,-1))]
        neighbours = []
        for direction, (dx,dy) in delta:
            x2, y2 = cell.x + dx, cell.y + dy
            if (0 <= x2 < self.nx) and (0 <= y2 < self.ny):
                neighbour = self.cell_at(x2, y2)
                if neighbour.has_all_walls():
                    neighbours.append((direction, neighbour))
        return neighbours

    def make_maze(self):
        # Total number of cells.
        n = self.nx * self.ny
        cell_stack = []
        current_cell = self.cell_at(self.ix, self.iy)
        # Total number of visited cells during maze construction.
        nv = 1

        while nv < n:
            neighbours = self.find_valid_neighbours(current_cell)

            if not neighbours:
                # We've reached a dead end: backtrack.
                random.shuffle(cell_stack)
                current_cell = cell_stack.pop()
                continue

            # Choose a random neighbouring cell and move to it.
            direction, next_cell = random.choice(neighbours)
            current_cell.knock_down_wall(next_cell, direction)
            cell_stack.append(current_cell)
            current_cell = next_cell
            nv += 1


# Maze dimensions (ncols, nrows)
# nx, ny = 10, 10
# Maze entry position
# ix, iy = 0, 0

# maze = Maze_no_walls(nx, ny, ix, iy)
# maze.make_maze()

# print(maze)
# maze.write_svg('maze.svg')