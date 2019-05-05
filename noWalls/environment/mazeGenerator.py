
import numpy as np
import random
import matplotlib.pyplot as plt

imgx = 16; imgy = 16
image = np.zeros((imgy,imgx))
pixels = np.zeros((imgy,imgx))

mx = 16; my = 16 # width and height of the maze

maze = [[0 for x in range(mx)] for y in range(my)]
dx = [0, 1, 0, -1]; dy = [-1, 0, 1, 0] # 4 directions to move in the maze
color = [(0,0, 0), (255, 255, 255)] # RGB colors of the maze
# start the maze from a random cell
stack = [(0,0)]

while len(stack) > 0:
    (cx, cy) = stack[-1]
    maze[cy][cx] = 1
    # find a new cell to add
    nlst = [] # list of available neighbors
    for i in range(4):
        nx = cx + dx[i]
        ny = cy + dy[i]
        if nx >= 0 and nx < mx and ny >= 0 and ny < my:
            if maze[ny][nx] == 0:
                # of occupied neighbors must be 1
                ctr = 0
                for j in range(4):
                    ex = nx + dx[j]; ey = ny + dy[j]
                    if ex >= 0 and ex < mx and ey >= 0 and ey < my:
                        if maze[ey][ex] == 1: ctr += 1
                if ctr == 1: nlst.append(i)
    # if 1 or more neighbors available then randomly select one and move
    if len(nlst) > 0:
        ir = nlst[random.randint(0, len(nlst) - 1)]
        cx += dx[ir]; cy += dy[ir]
        stack.append((cx, cy))
    else: stack.pop()

# paint the maze
for ky in range(imgy):
    for kx in range(imgx):
        pixels[kx, ky] = maze[int(my * ky / imgy)][int(mx * kx / imgx)]

pixels = np.array(pixels, dtype=int)
plt.imshow(pixels)
pixels = 1-pixels
np.save("maze_16_16", pixels)