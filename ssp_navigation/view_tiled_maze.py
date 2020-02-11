import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    'View a full tiled maze'
)

parser.add_argument('--n-mazes', type=int, default=100, help='Number of mazes from the dataset to train with')
parser.add_argument('--dataset-dir', type=str, default='datasets/mixed_style_100mazes_100goals_64res_13size_13seed')

args = parser.parse_args()

dataset_file = os.path.join(args.dataset_dir, 'maze_dataset.npz')

data = np.load(dataset_file)

# n_mazes by res by res
fine_mazes = data['fine_mazes'][:args.n_mazes, :, :]

res = fine_mazes.shape[1]
side_len = int(np.sqrt(args.n_mazes))

# the +1 is an offset to make the image look nicer
full_maze = np.ones((1+side_len*res, 1+side_len*res))

for x in range(side_len):
    for y in range(side_len):
        full_maze[1+x*res:1+(x+1)*res, 1+y*res:1+(y+1)*res] = fine_mazes[x*side_len + y, :, :]


# the :-4 is an offset to make the image look nicer
plt.imshow(1 - full_maze[:-4, :-4], cmap='gray')
# plt.figure()
# plt.imshow(1 - full_maze, cmap='gray')

plt.show()
