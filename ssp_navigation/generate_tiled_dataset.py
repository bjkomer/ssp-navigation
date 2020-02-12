import argparse
import numpy as np
import os

parser = argparse.ArgumentParser(
    'Generate a dataset with a giant maze consisting of a tiling of smaller mazes'
)

parser.add_argument('--dataset-dir', type=str, default='datasets/mixed_style_100mazes_100goals_64res_13size_13seed')
parser.add_argument('--side-len', type=int, default=10, choices=[2, 3, 4, 5, 6, 7, 8, 9, 10])
parser.add_argument('--output-name', type=str, required=True)

args = parser.parse_args()

dataset_file = os.path.join(args.dataset_dir, 'maze_dataset.npz')
data = np.load(dataset_file)

n_mazes = args.side_len ** 2

# n_mazes by res by res
fine_mazes = data['fine_mazes'][:n_mazes, :, :]

# n_mazes by size by size
coarse_mazes = data['coarse_mazes'][:n_mazes, :, :]

# n_mazes by n_goals by res by res by 2
solved_mazes = data['solved_mazes'][:n_mazes, :, :, :, :]

# n_mazes by n_goals by 2
goals = data['goals'][:n_mazes, :, :]

n_goals = goals.shape[1]
res = fine_mazes.shape[1]

full_maze = np.zeros((args.side_len*res, args.side_len*res))
full_solved_mazes = np.zeros((n_goals*n_mazes, res, res, 2))
# connectivity of the tiles
global_connectivity = np.zeros((n_mazes, n_mazes))


# convert coord to index in connectivity graph
def ci(x, y):
    return x*args.side_len + y


for x in range(args.side_len):
    for y in range(args.side_len):
        full_maze[x*res:(x+1)*res, y*res:(y+1)*res] = fine_mazes[x*args.side_len + y, :, :]

# constants for getting the offsets for connecting mazes correct
to_edge = res - 3
to_center = int(res/2)

# create connections between mazes
# left-right connections
for i in range(args.side_len - 1):
    for j in range(args.side_len):
        x = i*res + to_edge
        y = j*res + to_center - 1
        # check if a connection will be valid
        # if np.all(full_maze[x:x+6, y] == np.array([0, 1, 1, 1, 1, 0])):
        if full_maze[x-6, y] == 0 and full_maze[x+6, y] == 0:
            # set pathway to open
            full_maze[x-6:x+7, y-3:y] = 0
            # full_maze[x - 6-res//2:x + 7+res//2, y-3:y] = .5  # for viewing/debugging
            # mark in connectivity graph
            global_connectivity[ci(i, j), ci(i + 1, j)] = 1
            global_connectivity[ci(i + 1, j), ci(i, j)] = 1

# up-down connections
for j in range(args.side_len - 1):
    for i in range(args.side_len):
        x = i*res + to_center
        y = j*res + to_edge
        # check if a connection will be valid
        # if np.all(full_maze[x, y:y+6] == np.array([0, 1, 1, 1, 1, 0])):
        if full_maze[x, y - 6] == 0 and full_maze[x, y + 6] == 0:
            # set pathway to open
            full_maze[x:x+3, y-6:y+7] = 0
            # full_maze[x:x + 3, y - 6 - res//2:y + 7 + res//2] = .5  # for viewing/debugging
            # mark in connectivity graph
            global_connectivity[ci(i, j + 1), ci(i, j)] = 1
            global_connectivity[ci(i, j), ci(i, j + 1)] = 1



import matplotlib.pyplot as plt
plt.figure()
plt.imshow(1-full_maze, cmap='gray')
plt.figure()
plt.imshow(global_connectivity)
plt.show()


# np.savez(
#     args.output_name,
# )
