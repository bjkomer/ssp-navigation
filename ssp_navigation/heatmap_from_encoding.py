import torch
import numpy as np
import argparse
import os
from ssp_navigation.utils.models import MLP, LearnedEncoding
from utils.encodings import get_encoding_function, encoding_func_from_model
import nengo.spa as spa
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    'View the similarity heatmap of a point encoded at a particular location'
)

parser.add_argument('--spatial-encoding', type=str, default='ssp',
                    choices=[
                        'ssp', 'random', '2d', '2d-normalized', 'one-hot', 'hex-trig',
                        'trig', 'random-trig', 'random-proj', 'learned', 'frozen-learned',
                        'pc-gauss', 'pc-dog', 'tile-coding'
                    ],
                    help='coordinate encoding for agent location and goal')

parser.add_argument('--hex-freq-coef', type=float, default=2.5, help='constant to scale frequencies by for hex-trig')
parser.add_argument('--pc-gauss-sigma', type=float, default=0.25, help='sigma for the gaussians')
parser.add_argument('--pc-diff-sigma', type=float, default=0.5, help='sigma for subtracted gaussian in DoG')
parser.add_argument('--n-tiles', type=int, default=8, help='number of layers for tile coding')
parser.add_argument('--n-bins', type=int, default=8, help='number of bins for tile coding')
parser.add_argument('--ssp-scaling', type=float, default=1.0)
parser.add_argument('--res', type=int, default=64, help='resolution of the heatmap')
parser.add_argument('--maze-id-type', type=str, choices=['ssp', 'one-hot', 'random-sp'], default='one-hot',
                    help='ssp: region corresponding to maze layout.'
                         'one-hot: each maze given a one-hot vector.'
                         'random-sp: each maze given a unique random SP as an ID')
parser.add_argument('--maze-id-dim', default=256, help='Dimensionality for the Maze ID')
parser.add_argument('--seed', type=int, default=13, help='Seed for training and generating axis SSPs')
parser.add_argument('--dim', type=int, default=512, help='Dimensionality of the SSPs')
parser.add_argument('--hidden-size', type=int, default=512, help='Size of the hidden layer in the model')
parser.add_argument('--n-hidden-layers', type=int, default=1, help='Number of hidden layers in the model')
parser.add_argument('--model', type=str, default='', help='Saved model to use')

parser.add_argument('--limit-low', type=float, default=0.0, help='lower limit for heatmap')
parser.add_argument('--limit-high', type=float, default=13.0, help='upper limit for heatmap')
parser.add_argument('--x-pos', type=float, default=0.0, help='x-position of the test point')
parser.add_argument('--y-pos', type=float, default=0.0, help='y-position of the test point')

parser.add_argument('--n-mazes', type=int, default=10)

args = parser.parse_args()

xs = np.linspace(args.limit_low, args.limit_high, args.res)
ys = np.linspace(args.limit_low, args.limit_high, args.res)
if '2d' in args.spatial_encoding:
    activations = np.zeros((args.res, args.res, 2))
    normalized_activations = np.zeros((args.res, args.res, 2))
else:
    activations = np.zeros((args.res, args.res, args.dim))
    normalized_activations = np.zeros((args.res, args.res, args.dim))
heatmap = np.zeros((args.res, args.res))

rng = np.random.RandomState(seed=args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

if args.maze_id_type == 'ssp':
    id_size = args.dim
elif args.maze_id_type == 'one-hot':
    id_size = args.n_mazes
    # overwrite data
    maze_sps = np.eye(args.n_mazes)
elif args.maze_id_type == 'random-sp':
    id_size = args.dim
    maze_sps = np.zeros((args.n_mazes, args.dim))
    # overwrite data
    for mi in range(args.n_mazes):
        maze_sps[mi, :] = spa.SemanticPointer(args.dim).v
else:
    raise NotImplementedError


# limit_low = 0
# limit_high = 13
limit_low = args.limit_low
limit_high = args.limit_high

encoding_func, repr_dim = get_encoding_function(args, limit_low=limit_low, limit_high=limit_high)


# Reset seeds here after generating data
torch.manual_seed(args.seed)
np.random.seed(args.seed)


# input is maze, loc, goal ssps, output is 2D direction to move
if 'learned' in args.spatial_encoding:
    enc_func = encoding_func_from_model(args.model, args.dim)
    def encoding_func(x, y):
        return enc_func(np.array([x, y]))
else:
    pass

for ix, x in enumerate(xs):
    for iy, y in enumerate(ys):
        activations[ix, iy, :] = encoding_func(x, y)
        normalized_activations[ix, iy, :] = activations[ix, iy, :] / np.linalg.norm(activations[ix, iy, :])

encoded_point = encoding_func(args.x_pos, args.y_pos)

heatmap = np.tensordot(encoded_point, activations, axes=([0], [2]))
normalized_heatmap = np.tensordot(encoded_point, normalized_activations, axes=([0], [2]))

plt.figure()
plt.imshow(heatmap)
plt.title('heatmap - dot product')
plt.figure()
plt.imshow(normalized_heatmap)
plt.title('normalized heatmap - cosine similarity')
plt.show()
