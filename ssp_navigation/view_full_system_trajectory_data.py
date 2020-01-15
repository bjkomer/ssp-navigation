import argparse
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser('Run the full system on a maze task and record the trajectories taken')

parser.add_argument('--fname', type=str, default='fs_traj_data.npz', help='file to save the trajectory data to')
parser.add_argument('--maze-index', type=int, default=0, help='index within the dataset for the maze to use')
parser.add_argument('--dataset', type=str,
                    default='datasets/mixed_style_100mazes_100goals_64res_13size_13seed/maze_dataset.npz',
                    help='dataset to get the maze layout from')

parser.add_argument('--start-x', type=float, default=0, help='x-coord of the agent start location')
parser.add_argument('--start-y', type=float, default=0, help='y-coord of the agent start location')
parser.add_argument('--goal-x', type=float, default=0, help='x-coord of the goal location')
parser.add_argument('--goal-y', type=float, default=0, help='y-coord of the goal location')

args = parser.parse_args()

dataset = np.load(args.dataset)

limit_low = 0
limit_high = dataset['coarse_mazes'].shape[2]

map_array = dataset['coarse_mazes'][args.maze_index, :, :]


data = np.load(args.fname)

# shape is n_trajectories, n_steps, 2
locations = data['locations']

n_trajectories = locations.shape[0]
n_steps = locations.shape[1]

# print(np.max(locations))
# print(np.min(locations))

plt.imshow(1 - map_array.T, cmap='gray')

for i in range(n_trajectories):
    indices = np.where((locations[i, :, 0] != 0) | (locations[i, :, 1] != 0))
    plt.plot(locations[i, indices, 0][0], locations[i, indices, 1][0])
    # plt.plot(locations[i, :, 0], locations[i, :, 1])



plt.show()
