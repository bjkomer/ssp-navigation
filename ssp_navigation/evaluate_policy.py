from ssp_navigation.utils.training import PolicyValidationSet, create_policy_dataloader
import torch
import numpy as np
import argparse
import os
from ssp_navigation.utils.models import MLP, LearnedEncoding
from utils.encodings import get_ssp_encode_func, encode_trig, encode_hex_trig, encode_random_trig, \
    encode_projection, get_one_hot_encode_func
from spatial_semantic_pointers.utils import encode_random
from functools import partial
import nengo.spa as spa
import pandas as pd

parser = argparse.ArgumentParser(
    'Compute the RMSE of a policy on a dataset'
)

parser.add_argument('--spatial-encoding', type=str, default='ssp',
                    choices=[
                        'ssp', 'random', '2d', '2d-normalized', 'one-hot', 'hex-trig',
                        'trig', 'random-trig', 'random-proj', 'learned', 'frozen-learned',
                    ],
                    help='coordinate encoding for agent location and goal')
parser.add_argument('--hex-freq-coef', type=float, default=2.5, help='constant to scale frequencies by for hex-trig')
parser.add_argument('--maze-id-type', type=str, choices=['ssp', 'one-hot', 'random-sp'], default='one-hot',
                    help='ssp: region corresponding to maze layout.'
                         'one-hot: each maze given a one-hot vector.'
                         'random-sp: each maze given a unique random SP as an ID')
parser.add_argument('--seed', type=int, default=13, help='Seed for training and generating axis SSPs')
parser.add_argument('--dim', type=int, default=512, help='Dimensionality of the SSPs')
parser.add_argument('--hidden-size', type=int, default=512, help='Size of the hidden layer in the model')
parser.add_argument('--n-hidden-layers', type=int, default=1, help='Number of hidden layers in the model')
parser.add_argument('--dataset-dir', type=str, default='datasets/mixed_style_20mazes_50goals_64res_13size_13seed')
parser.add_argument('--model', type=str, default='', help='Saved model to use')

parser.add_argument('--gpu', type=int, default=-1,
                    help="set to an integer corresponding to the gpu to use. Set to -1 to use the CPU")

parser.add_argument('--out-file', type=str, default="", help='Output file name')

args = parser.parse_args()

# an output file must be specified
assert args.out_file != ""
# only support npz or pandas csv
assert ('.npz' in args.out_file) or ('.csv' in args.out_file)

dataset_file = os.path.join(args.dataset_dir, 'maze_dataset.npz')

data = np.load(dataset_file)

rng = np.random.RandomState(seed=args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# n_mazes by res by res
fine_mazes = data['fine_mazes']

# n_mazes by res by res by 2
solved_mazes = data['solved_mazes']

# n_mazes by dim
maze_sps = data['maze_sps']

# n_mazes by n_goals by 2
goals = data['goals']

n_goals = goals.shape[1]
n_mazes = fine_mazes.shape[0]

if args.gpu == -1:
    device = torch.device('cpu:0')
    pin_memory = False
else:
    device = torch.device('cuda:{}'.format(int(args.gpu)))
    pin_memory = True

if args.maze_id_type == 'ssp':
    id_size = args.dim
elif args.maze_id_type == 'one-hot':
    id_size = n_mazes
    # overwrite data
    maze_sps = np.eye(n_mazes)
elif args.maze_id_type == 'random-sp':
    id_size = args.dim
    maze_sps = np.zeros((n_mazes, args.dim))
    # overwrite data
    for mi in range(n_mazes):
        maze_sps[mi, :] = spa.SemanticPointer(args.dim).v
else:
    raise NotImplementedError

limit_low = 0
limit_high = data['coarse_mazes'].shape[2]

# Dimension of location representation is dependent on the encoding used
repr_dim = args.dim

# Generate the encoding function
if args.spatial_encoding == '2d' or args.spatial_encoding == 'learned' or args.spatial_encoding == 'frozen-learned':
    repr_dim = 2
    # no special encoding required for these cases
    def encoding_func(x, y):
        return np.array([x, y])
elif args.spatial_encoding == '2d-normalized':
    repr_dim = 2
    def encoding_func(x, y):
        return ((np.array([x, y]) - limit_low) * 2 / (limit_high - limit_low)) - 1
elif args.spatial_encoding == 'ssp':
    encoding_func = get_ssp_encode_func(args.dim, args.seed)
elif args.spatial_encoding == 'one-hot':
    repr_dim = int(np.sqrt(args.dim)) ** 2
    encoding_func = get_one_hot_encode_func(dim=args.dim, limit_low=limit_low, limit_high=limit_high)
elif args.spatial_encoding == 'trig':
    encoding_func = partial(encode_trig, dim=args.dim)
elif args.spatial_encoding == 'random-trig':
    encoding_func = partial(encode_random_trig, dim=args.dim, seed=args.seed)
elif args.spatial_encoding == 'hex-trig':
    encoding_func = partial(
        encode_hex_trig,
        dim=args.dim, seed=args.seed,
        frequencies=(args.hex_freq_coef, args.hex_freq_coef*1.4, args.hex_freq_coef*1.4*1.4)
    )
elif args.spatial_encoding == 'random-proj':
    encoding_func = partial(encode_projection, dim=args.dim, seed=args.seed)
elif args.spatial_encoding == 'random':
    encoding_func = partial(encode_random, dim=args.dim)
else:
    raise NotImplementedError


# Create a validation/visualization set to run periodically while training and at the end
# validation_set = ValidationSet(data=data, maze_indices=np.arange(n_mazes), goal_indices=[0])

# Set up number of mazes/goals to view in the viz set based on how many are available
if n_mazes < 4:
    maze_indices = list(np.arange(n_mazes))
    if n_goals < 4:
        goal_indices = list(np.arange(n_goals))
    else:
        goal_indices = [0, 1, 2, 3]
else:
    maze_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    if n_goals < 2:
        goal_indices = [0]
    else:
        goal_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

validation_set = PolicyValidationSet(
    data=data, dim=repr_dim, maze_sps=maze_sps, maze_indices=maze_indices, goal_indices=goal_indices, subsample=args.subsample,
    encoding_func=encoding_func, device=device
)

# Reset seeds here after generating data
torch.manual_seed(args.seed)
np.random.seed(args.seed)


# input is maze, loc, goal ssps, output is 2D direction to move
if 'learned' in args.spatial_encoding:
    model = LearnedEncoding(
        input_size=repr_dim,
        encoding_size=args.dim,
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

model.to(device)

rmses = validation_set.get_rmse(model)

if '.npz' in args.out_file:
    np.savez(args.outfile, rmses=rmses)
elif '.csv' in args.out_file:
    df = pd.DataFrame(
        data=rmses,
        columns='RMSE',
    )
    df['Dimensionality'] = args.dim
    df['Hidden Layer Size'] = args.hidden_size
    df['Hidden Layers'] = args.n_hidden_layers
    df['Encoding'] = args.spatial_encoding
    df['Seed'] = args.seed
    df['Maze ID Type'] = args.maze_id_type
    df.to_csv(args.out_file)

