import matplotlib.pyplot as plt
from environment.maze import Maze as Maze
from models import *
from models.qtable_regular import *


maze = np.load('maze_16_16.npy')
game = Maze(maze)


model = QTableModel_regular(game, name="QTableModel")
h, w, _, _ = model.train(discount=0.90, exploration_rate=0.10, learning_rate=0.10, episodes=200)
model.save_model(model_name="last")

fig, (ax1, ax2) = plt.subplots(2, 1, tight_layout=True)
fig.canvas.set_window_title(model.name)
ax1.plot(*zip(*w))
ax1.set_xlabel("episode")
ax1.set_ylabel("win rate")
ax2.plot(h)
ax2.set_xlabel("episode")
ax2.set_ylabel("cumulative reward")
plt.show()

# h, w = model.restore_train_results()
game.render(content="moves")
game.play(model, start_cell=(0, 0))
# game.play(model, start_cell=(2, 5))
# game.play(model, start_cell=(4, 1))
plt.show()
