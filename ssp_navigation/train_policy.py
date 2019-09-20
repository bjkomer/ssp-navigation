from ssp_navigation.utils.training import PolicyValidationSet, create_policy_dataloader
import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import json
from tensorboardX import SummaryWriter
from datetime import datetime
from ssp_navigation.utils.models import FeedForward, MLP, LearnedEncoding
from utils.encodings import get_ssp_encode_func, encode_trig, encode_hex_trig, encode_random_trig, \
    encode_projection, encode_one_hot, get_one_hot_encode_func
from spatial_semantic_pointers.utils import encode_random
from functools import partial
import nengo.spa as spa

parser = argparse.ArgumentParser(
    'Train a function that given a maze and a goal location, computes the direction to move to get to that goal'
)

parser.add_argument('--epochs', type=int, default=250, help='Number of epochs to train for')
parser.add_argument('--epoch-offset', type=int, default=0,
                    help='Optional offset to start epochs counting from. To be used when continuing training')
parser.add_argument('--viz-period', type=int, default=50, help='number of epochs before a viz set run')
parser.add_argument('--val-period', type=int, default=25, help='number of epochs before a test/validation set run')
parser.add_argument('--spatial-encoding', type=str, default='ssp',
                    choices=[
                        'ssp', 'random', '2d', '2d-normalized', 'one-hot', 'hex-trig',
                        'trig', 'random-trig', 'random-proj', 'learned', 'frozen-learned',
                    ],
                    help='coordinate encoding for agent location and goal')
parser.add_argument('--hex-freq-coef', type=float, default=2.5, help='constant to scale frequencies by for hex-trig')
parser.add_argument('--subsample', type=int, default=1, help='amount to subsample for the visualization validation')
parser.add_argument('--maze-id-type', type=str, choices=['ssp', 'one-hot', 'random-sp'], default='one-hot',
                    help='ssp: region corresponding to maze layout.'
                         'one-hot: each maze given a one-hot vector.'
                         'random-sp: each maze given a unique random SP as an ID')
parser.add_argument('--seed', type=int, default=13, help='Seed for training and generating axis SSPs')
parser.add_argument('--dim', type=int, default=512, help='Dimensionality of the SSPs')
parser.add_argument('--n-train-samples', type=int, default=50000, help='Number of training samples')
parser.add_argument('--n-test-samples', type=int, default=5000, help='Number of testing samples')
parser.add_argument('--hidden-size', type=int, default=512, help='Size of the hidden layer in the model')
parser.add_argument('--n-hidden-layers', type=int, default=1, help='Number of hidden layers in the model')
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-histogram', action='store_true', help='Save histogram of the weights')
parser.add_argument('--res', type=int, default=64, help='resolution of the fine maze')
parser.add_argument('--dataset-dir', type=str, default='datasets/mixed_style_20mazes_50goals_64res_13size_13seed')
parser.add_argument('--no-wall-overlay', action='store_true', help='Do not use rainbow colours and wall overlay in validation images')
parser.add_argument('--variant-subfolder', type=str, default='',
                    help='Optional custom subfolder')
parser.add_argument('--logdir', type=str, default='policy',
                    help='Directory for saved model and tensorboard log, within dataset-dir')
parser.add_argument('--load-saved-model', type=str, default='', help='Saved model to load from')

args = parser.parse_args()

dataset_file = os.path.join(args.dataset_dir, 'maze_dataset.npz')

variant_folder = '{}_{}train_{}_id_{}layer_{}units'.format(
    args.spatial_encoding, args.n_train_samples, args.maze_id_type, args.n_hidden_layers, args.hidden_size
)

if args.variant_subfolder != '':
    variant_folder = os.path.join(variant_folder, args.variant_subfolder)

logdir = os.path.join(args.dataset_dir, 'policy', variant_folder)

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

# n_mazes by n_goals by dim
# goal_sps = data['goal_sps']

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

# Generate the encoding function
if args.spatial_encoding == '2d' or args.spatial_encoding == 'learned' or args.spatial_encoding == 'frozen-learned':
    # no special encoding required for these cases
    def encoding_func(x, y):
        return np.array([x, y])
elif args.spatial_encoding == '2d-normalized':
    def encoding_func(x, y):
        return ((np.array([x, y]) - limit_low) * 2 / (limit_high - limit_low)) - 1
elif args.spatial_encoding == 'ssp':
    encoding_func = get_ssp_encode_func(args.dim, args.seed)
elif args.spatial_encoding == 'one-hot':
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
    maze_indices = [0, 1, 2, 3]
    if n_goals < 2:
        goal_indices = [0]
    else:
        goal_indices = [0, 1]


validation_set = PolicyValidationSet(
    data=data, dim=args.dim, maze_sps=maze_sps, maze_indices=maze_indices, goal_indices=goal_indices, subsample=args.subsample,
    # spatial_encoding=args.spatial_encoding,
    encoding_func=encoding_func,
)


# TODO: ensure train and test are different
trainloader = create_policy_dataloader(
    data=data, n_samples=args.n_train_samples, maze_sps=maze_sps, args=args, encoding_func=encoding_func
)

testloader = create_policy_dataloader(
    data=data, n_samples=args.n_test_samples, maze_sps=maze_sps, args=args, encoding_func=encoding_func
)

# Reset seeds here after generating data
torch.manual_seed(args.seed)
np.random.seed(args.seed)

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
elif args.spatial_encoding == 'hex-trig':
    repr_dim = args.dim
elif args.spatial_encoding == 'random-trig':
    repr_dim = args.dim
elif args.spatial_encoding == 'random-proj':
    repr_dim = args.dim

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
# if args.n_hidden_layers > 1:
#     model = MLP(input_size=id_size + repr_dim * 2, hidden_size=args.hidden_size, output_size=2, n_layers=args.n_hidden_layers)
# else:
#     if 'learned' in args.spatial_encoding:
#         model = LearnedEncoding(input_size=repr_dim, maze_id_size=id_size, hidden_size=args.hidden_size, output_size=2)
#     else:
#         model = FeedForward(input_size=id_size + repr_dim * 2, hidden_size=args.hidden_size, output_size=2)

if args.load_saved_model:
    if args.spatial_encoding == 'frozen-learned':
        # TODO: make sure this is working correctly
        print("Loading learned first layer parameters from pretrained model")
        state_dict = torch.load(args.load_saved_model)

        for name, param in state_dict.items():
            if name in ['encoding_layer.weight', 'encoding_layer.bias']:
                model.state_dict()[name].copy_(param)

        print("Freezing first layer parameters for training")
        for name, param in model.named_parameters():
            if name in ['encoding_layer.weight', 'encoding_layer.bias']:
                param.requires_grad = False
            if param.requires_grad:
                print(name)
    else:
        model.load_state_dict(torch.load(args.load_saved_model), strict=False)
elif args.spatial_encoding == 'frozen-learned':
    raise NotImplementedError("Must select a model to load from when using frozen-learned spatial encoding")

# Open a tensorboard writer if a logging directory is given
if args.logdir != '':
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    save_dir = os.path.join(logdir, current_time)
    writer = SummaryWriter(log_dir=save_dir)
    if args.weight_histogram:
        # Log the initial parameters
        for name, param in model.named_parameters():
            writer.add_histogram('parameters/' + name, param.clone().cpu().data.numpy(), 0)


validation_set.run_ground_truth(writer=writer)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

for e in range(args.epoch_offset, args.epochs + args.epoch_offset):
    print('Epoch: {0}'.format(e + 1))

    if e % args.viz_period == 0:
        print("Running Viz Set")
        # do a validation run and save images
        validation_set.run_validation(model, writer, e, use_wall_overlay=not args.no_wall_overlay)

        if e > 0:
            # Save a copy of the model at this stage
            torch.save(model.state_dict(), os.path.join(save_dir, 'model_epoch_{}.pt'.format(e)))

    # Run the test set for validation
    if e % args.val_period == 0:
        print("Running Val Set")
        with torch.no_grad():
            # Everything is in one batch, so this loop will only happen once
            for i, data in enumerate(testloader):
                maze_loc_goal_ssps, directions, locs, goals = data

                outputs = model(maze_loc_goal_ssps)

                loss = criterion(outputs, directions)

            writer.add_scalar('test_loss', loss.data.item(), e)

    avg_loss = 0
    n_batches = 0
    for i, data in enumerate(trainloader):
        maze_loc_goal_ssps, directions, locs, goals = data

        if maze_loc_goal_ssps.size()[0] != args.batch_size:
            continue  # Drop data, not enough for a batch
        optimizer.zero_grad()

        outputs = model(maze_loc_goal_ssps)

        loss = criterion(outputs, directions)
        # print(loss.data.item())
        avg_loss += loss.data.item()
        n_batches += 1

        loss.backward()

        optimizer.step()

    if args.logdir != '':
        if n_batches > 0:
            avg_loss /= n_batches
            print(avg_loss)
            writer.add_scalar('avg_loss', avg_loss, e + 1)

        if args.weight_histogram and (e + 1) % 10 == 0:
            for name, param in model.named_parameters():
                writer.add_histogram('parameters/' + name, param.clone().cpu().data.numpy(), e + 1)

print("Testing")
with torch.no_grad():
    # Everything is in one batch, so this loop will only happen once
    for i, data in enumerate(testloader):
        maze_loc_goal_ssps, directions, locs, goals = data

        outputs = model(maze_loc_goal_ssps)

        loss = criterion(outputs, directions)

        # print(loss.data.item())

    if args.logdir != '':
        writer.add_scalar('final_test_loss', loss.data.item())

print("Visualization")
validation_set.run_validation(model, writer, args.epochs + args.epoch_offset, use_wall_overlay=not args.no_wall_overlay)


# Close tensorboard writer
if args.logdir != '':
    writer.close()

    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))

    params = vars(args)

    # if random-sp is used as maze-id-type, then save the sps used as well
    if args.maze_id_type == 'random-sp':
        params['maze_sps'] = [list(maze_sps[mi, :]) for mi in range(n_mazes)]

    with open(os.path.join(save_dir, "params.json"), "w") as f:
        json.dump(params, f)
