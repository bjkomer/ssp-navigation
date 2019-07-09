import torch
import torch.nn as nn
import numpy as np
import argparse

from ssp_navigation.utils.models import MLP, LearnedEncoding
from ssp_navigation.utils.datasets import MazeDataset
import nengo.spa as spa
import matplotlib.pyplot as plt
from ssp_navigation.utils.path import plot_path_predictions_image

# TODO: make it so the user can click on the image, and the location clicked is the goal

parser = argparse.ArgumentParser(
    'View a policy for an interactive goal location on a particular map'
)

parser.add_argument('--map-index', type=int, default=0, help='Index of the map in the dataset to use')
parser.add_argument('--seed', type=int, default=13, help='Seed for training and generating axis SSPs')
parser.add_argument('--limit-low', type=float, default=-5, help='lowest coordinate value')
parser.add_argument('--limit-high', type=float, default=5, help='highest coordinate value')
parser.add_argument('--hidden-size', type=int, default=512)
parser.add_argument('--n-hidden-layers', type=int, default=1)
# parser.add_argument('--view-activations', action='store_true', help='view spatial activations of each neuron')
parser.add_argument('--dim', type=int, default=512, help='Dimensionality of the SSPs')
parser.add_argument('--spatial-encoding', type=str, default='ssp',
                    choices=['ssp', 'random', '2d', '2d-normalized', 'one-hot', 'trig', 'random-proj', 'random-trig', 'learned'],
                    help='coordinate encoding for agent location and goal')
parser.add_argument('--maze-id-type', type=str, choices=['ssp', 'one-hot', 'random-sp'], default='one-hot',
                    help='ssp: region corresponding to maze layout.'
                         'one-hot: each maze given a one-hot vector.'
                         'random-sp: each maze given a unique random SP as an ID')
parser.add_argument('--subsample', type=int, default=1, help='amount to subsample for the visualization validation')
parser.add_argument(
    '--dataset', type=str,
    default='maze_datasets/maze_dataset_maze_style_10mazes_25goals_64res_13size_13seed.npz',
)

parser.add_argument('--load-saved-model', type=str, default='', help='Saved model to load from')

args = parser.parse_args()

data = np.load(args.dataset)

rng = np.random.RandomState(seed=args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# n_mazes by res by res
fine_mazes = data['fine_mazes']

# n_mazes by res by res by 2
solved_mazes = data['solved_mazes']

# n_mazes by dim
maze_sps = data['maze_sps']

# n_mazes by n_goals by dim
goal_sps = data['goal_sps']

# n_mazes by n_goals by 2
goals = data['goals']

n_goals = goals.shape[1]
n_mazes = fine_mazes.shape[0]



if args.maze_id_type == 'ssp':
    id_size = args.dim
elif args.maze_id_type == 'one-hot':
    id_size = n_mazes
    # overwrite data
    maze_sps = np.eye(n_mazes)
else:
    raise NotImplementedError


# Dimension of location representation is dependent on the encoding used
if args.spatial_encoding == 'ssp':
    repr_dim = args.dim
elif args.spatial_encoding == 'random':
    repr_dim = args.dim
elif args.spatial_encoding == '2d':
    repr_dim = 2
elif args.spatial_encoding == 'learned':
    repr_dim = 2
elif args.spatial_encoding == 'frozen-learned':  # use a pretrained learned representation
    repr_dim = 2
elif args.spatial_encoding == '2d-normalized':
    repr_dim = 2
elif args.spatial_encoding == 'one-hot':
    repr_dim = int(np.sqrt(args.dim))**2
elif args.spatial_encoding == 'trig':
    repr_dim = args.dim
elif args.spatial_encoding == 'random-trig':
    repr_dim = args.dim
elif args.spatial_encoding == 'random-proj':
    repr_dim = args.dim

# input is maze, loc, goal ssps, output is 2D direction to move
if 'learned' in args.spatial_encoding:
    model = LearnedEncoding(
        input_size=repr_dim,
        maze_id_size=id_size,
        hidden_size=args.hidden_size,
        output_size=2,
        n_layers=args.n_hidden_layers
    )
else:
    model = MLP(
        input_size=id_size + repr_dim * 2,
        hidden_size=args.hidden_size,
        output_size=2,
        n_layers=args.n_hidden_layers
    )

if args.load_saved_model:
    model.load_state_dict(torch.load(args.load_saved_model), strict=False)

model.eval()




fig = plt.figure()
ax = fig.add_subplot(111)

# initialize image with correct dimensions
ax.imshow(np.zeros((64, 64)), cmap='hsv', interpolation=None)

def on_click(event):

    # TODO: need to make sure these coordinates are in the right reference frame, and translate them if they are not
    goal = (event.xdata, event.ydata)
    print(goal)

    # for i, data in enumerate(self.vizloader):
    #
    #     maze_loc_goal_ssps, directions, locs, goals = data
    #
    #     outputs = model(maze_loc_goal_ssps)
    #
    #     # loss = criterion(outputs, directions)
    #
    #     wall_overlay = (directions.detach().numpy()[:, 0] == 0) & (directions.detach().numpy()[:, 1] == 0)
    #
    #     fig_pred = plot_path_predictions_image(
    #         ax=ax,
    #         directions=outputs.detach().numpy(), coords=locs.detach().numpy(), wall_overlay=wall_overlay
    #     )

fig.canvas.mpl_connect('button_press_event', on_click)
plt.show()
