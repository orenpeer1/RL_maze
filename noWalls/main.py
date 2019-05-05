import matplotlib.pyplot as plt
from environment.thinWalsMaze import Maze_no_walls as Maze_no_walls
from environment.maze import Maze as Maze
from models import *
from models.qtable_regular import *
from models.qtable_median import *


# Maze dimensions (ncols, nrows)
# nx, ny = 10, 10
# Maze entry position
# ix, iy = 0, 0

# maze = Maze_no_walls(nx, ny, ix, iy)
# maze.make_maze()
# maze = np.save('maze_6_6.npy', maze)
# maze = np.load('maze_6_6.npy').item()
maze = np.load('maze_10_10.npy').item()
game = Maze(maze)
# maze.draw()

# model = QTableModel_regular(game, name="QTableModel")
ks, kms = [9, 30, 60, 9, 105], [3, 10, 20, 1, 1]
episodes_rewards = []
cum_reward = []
for k, km in zip(ks, kms):
    print("k = " + str(k))
    print("km = " + str(km))
    model = QTableModel_median(game, name="QTableModel")
    k_cum_reward, k_episodes_rewards, _ = model.train(discount=0.98, episodes=50, k=k, km=km)
    episodes_rewards.append(k_episodes_rewards)
    cum_reward.append(k_cum_reward)
# np.save("results", {"ks": ks, "kms": kms, "cum_reward": cum_reward, "episodes_rewards": episodes_rewards})
results = np.load("results.npy").item()
# model.save_model(model_name="last")

ks = results["ks"]; kms = results["kms"]; cum_reward = results["cum_reward"]; episodes_rewards = results["episodes_rewards"]


fig = plt.figure()
ax1 = fig.add_subplot(121)
for episode_res, k, km in zip(episodes_rewards, ks, kms):
    ax1.plot(episode_res, label="k_m="+str(km)+", k="+str(k))
ax1.set_xlabel("episode")
ax1.set_ylabel("episodes_rewards")
ax1.legend()
# ax2.plot(cum_reward)
ax2 = fig.add_subplot(122)
ax2.set_xlabel("episode")
ax2.set_ylabel("cumulative reward")
for cum_reward, k, km in zip(episodes_rewards, ks, kms):
    ax2.plot(cum_reward, label="k_m="+str(km)+", k="+str(k))
ax2.legend()
plt.show()

# h, w = model.restore_train_results()
game.render(content="moves")
game.play(model, start_cell=(9, 0))
# game.play(model, start_cell=(2, 5))
# game.play(model, start_cell=(4, 1))
plt.show()
