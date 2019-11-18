from ssp_navigation.utils.training import OpenEnvPolicyValidationSet, create_generalizing_train_test_dataloaders
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
    encode_projection, encode_one_hot, get_one_hot_encode_func, get_pc_gauss_encoding_func, get_tile_encoding_func
from spatial_semantic_pointers.utils import encode_random
from functools import partial
import nengo.spa as spa

parser = argparse.ArgumentParser(
    'Train a function that given a maze and a goal location, computes the direction to move to get to that goal'
)

parser.add_argument('--generalization-type', type=str, default='interpolate',
                    choices=['interpolate', 'extrapolate', 'patches'],
                    help='Which part of the space should be left out during training')
parser.add_argument('--loss-function', type=str, default='mse', choices=['cosine', 'mse'])
parser.add_argument('--epochs', type=int, default=250, help='Number of epochs to train for')
parser.add_argument('--epoch-offset', type=int, default=0,
                    help='Optional offset to start epochs counting from. To be used when continuing training')
parser.add_argument('--viz-period', type=int, default=50, help='number of epochs before a viz set run')
parser.add_argument('--val-period', type=int, default=25, help='number of epochs before a test/validation set run')
parser.add_argument('--spatial-encoding', type=str, default='ssp',
                    choices=[
                        'ssp', 'random', '2d', '2d-normalized', 'one-hot', 'hex-trig',
                        'trig', 'random-trig', 'random-proj', 'learned', 'frozen-learned',
                        'pc-gauss', 'tile-coding'
                    ],
                    help='coordinate encoding for agent location and goal')
parser.add_argument('--hex-freq-coef', type=float, default=2.5, help='constant to scale frequencies by for hex-trig')
parser.add_argument('--pc-gauss-sigma', type=float, default=0.25, help='sigma for the gaussians')
parser.add_argument('--n-tiles', type=int, default=8, help='number of layers for tile coding')
parser.add_argument('--n-bins', type=int, default=0, help='number of bins for tile coding')
parser.add_argument('--subsample', type=int, default=1, help='amount to subsample for the visualization validation')
parser.add_argument('--seed', type=int, default=13, help='Seed for training and generating axis SSPs')
parser.add_argument('--dim', type=int, default=512, help='Dimensionality of the encoding')
parser.add_argument('--n-train-samples', type=int, default=10000, help='Number of training samples')
parser.add_argument('--n-test-samples', type=int, default=10000, help='Number of testing samples')
parser.add_argument('--hidden-size', type=int, default=512, help='Size of the hidden layer in the model')
parser.add_argument('--n-hidden-layers', type=int, default=1, help='Number of hidden layers in the model')
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-histogram', action='store_true', help='Save histogram of the weights')
parser.add_argument('--res', type=int, default=64, help='resolution of the fine maze')
parser.add_argument('--no-wall-overlay', action='store_true', help='Do not use rainbow colours and wall overlay in validation images')
parser.add_argument('--variant-subfolder', type=str, default='',
                    help='Optional custom subfolder')
parser.add_argument('--load-saved-model', type=str, default='', help='Saved model to load from')

parser.add_argument('--gpu', type=int, default=-1,
                    help="set to an integer corresponding to the gpu to use. Set to -1 to use the CPU")

parser.add_argument('--limit-low', type=float, default=-5.0)
parser.add_argument('--limit-high', type=float, default=5.0)

args = parser.parse_args()

variant_folder = '{}_{}train_{}_id_{}layer_{}units'.format(
    args.spatial_encoding, args.n_train_samples, args.generalization_type, args.n_hidden_layers, args.hidden_size
)

if args.variant_subfolder != '':
    variant_folder = os.path.join(args.variant_subfolder, variant_folder)

logdir = os.path.join('generalization_exps', args.generalization_type, variant_folder)

rng = np.random.RandomState(seed=args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)


if args.gpu == -1:
    device = torch.device('cpu:0')
    pin_memory = False
else:
    device = torch.device('cuda:{}'.format(int(args.gpu)))
    pin_memory = True

limit_low = args.limit_low
limit_high = args.limit_high

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
elif args.spatial_encoding == 'pc-gauss':
    rng = np.random.RandomState(seed=args.seed)
    encoding_func = get_pc_gauss_encoding_func(
        limit_low=limit_low, limit_high=limit_high, dim=args.dim, sigma=args.pc_gauss_sigma, use_softmax=False, rng=rng
    )
elif args.spatial_encoding == 'tile-coding':
    rng = np.random.RandomState(seed=args.seed)
    encoding_func = get_tile_encoding_func(
        limit_low=limit_low, limit_high=limit_high, dim=args.dim, n_tiles=args.n_tiles, n_bins=args.n_bins, rng=rng
    )
elif args.spatial_encoding == 'random-proj':
    encoding_func = partial(encode_projection, dim=args.dim, seed=args.seed)
elif args.spatial_encoding == 'random':
    encoding_func = partial(encode_random, dim=args.dim)
else:
    raise NotImplementedError

# Create a validation/visualization set to run periodically while training and at the end

validation_set = OpenEnvPolicyValidationSet(
    dim=args.dim,
    n_goals=5,
    res=64,
    limit_low=args.limit_low,
    limit_high=args.limit_high,
    encoding_func=encoding_func,
    device=device
)


trainloader, testloader = create_generalizing_train_test_dataloaders(
    dim=args.dim,
    generalization_type=args.generalization_type,
    n_train_samples=args.n_train_samples,
    n_test_samples=args.n_test_samples,
    limit_low=args.limit_low,
    limit_high=args.limit_high,
    encoding_func=encoding_func,
    batch_size=args.batch_size,
    pin_memory=pin_memory,
)

# Reset seeds here after generating data
torch.manual_seed(args.seed)
np.random.seed(args.seed)


# input is maze, loc, goal ssps, output is 2D direction to move
if 'learned' in args.spatial_encoding:
    model = LearnedEncoding(
        input_size=repr_dim,
        encoding_size=args.dim,
        maze_id_size=0,  # TODO: make sure this works correctly
        hidden_size=args.hidden_size,
        output_size=2,
        n_layers=args.n_hidden_layers
    )
else:
    model = MLP(
        input_size=0 + repr_dim * 2,
        hidden_size=args.hidden_size,
        output_size=2,
        n_layers=args.n_hidden_layers
    )

model.to(device)


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
current_time = datetime.now().strftime('%b%d_%H-%M-%S')
save_dir = os.path.join(logdir, current_time)
writer = SummaryWriter(log_dir=save_dir)
if args.weight_histogram:
    # Log the initial parameters
    for name, param in model.named_parameters():
        writer.add_histogram('parameters/' + name, param.clone().cpu().data.numpy(), 0)


validation_set.run_ground_truth(writer=writer)

# criterion = nn.MSELoss()
cosine_criterion = nn.CosineEmbeddingLoss()
mse_criterion = nn.MSELoss()
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
        avg_test_mse_loss = 0
        avg_test_cosine_loss = 0
        n_test_batches = 0
        with torch.no_grad():
            # Everything is in one batch, so this loop will only happen once
            for i, data in enumerate(testloader):
                maze_loc_goal_ssps, directions, locs, goals = data

                outputs = model(maze_loc_goal_ssps.to(device))

                mse_loss = mse_criterion(outputs, directions.to(device))
                cosine_loss = cosine_criterion(
                    outputs,
                    directions.to(device),
                    torch.ones(maze_loc_goal_ssps.size()[0]).to(device)
                )

                avg_test_mse_loss += mse_loss.data.item()
                avg_test_cosine_loss += cosine_loss.data.item()
                n_test_batches += 1

        if n_test_batches > 0:
            avg_test_mse_loss /= n_test_batches
            avg_test_cosine_loss /= n_test_batches
            print(avg_test_mse_loss, avg_test_cosine_loss)
            writer.add_scalar('test_mse_loss', avg_test_mse_loss, e)
            writer.add_scalar('test_cosine_loss', avg_test_cosine_loss, e)

    avg_mse_loss = 0
    avg_cosine_loss = 0
    n_batches = 0
    for i, data in enumerate(trainloader):
        maze_loc_goal_ssps, directions, locs, goals = data

        if maze_loc_goal_ssps.size()[0] != args.batch_size:
            continue  # Drop data, not enough for a batch
        optimizer.zero_grad()

        outputs = model(maze_loc_goal_ssps.to(device))

        mse_loss = mse_criterion(outputs, directions.to(device))
        cosine_loss = cosine_criterion(
            outputs,
            directions.to(device),
            torch.ones(args.batch_size).to(device)
        )
        # print(loss.data.item())
        avg_mse_loss += mse_loss.data.item()
        avg_cosine_loss += cosine_loss.data.item()
        n_batches += 1

        if args.loss_function == 'mse':
            mse_loss.backward()
        elif args.loss_function == 'cosine':
            cosine_loss.backward()

        optimizer.step()

    if n_batches > 0:
        avg_mse_loss /= n_batches
        avg_cosine_loss /= n_batches
        print(avg_mse_loss, avg_cosine_loss)
        writer.add_scalar('avg_mse_loss', avg_mse_loss, e + 1)
        writer.add_scalar('avg_cosine_loss', avg_cosine_loss, e + 1)

    if args.weight_histogram and (e + 1) % 10 == 0:
        for name, param in model.named_parameters():
            writer.add_histogram('parameters/' + name, param.clone().cpu().data.numpy(), e + 1)


print("Testing")
avg_test_mse_loss = 0
avg_test_cosine_loss = 0
n_test_batches = 0
with torch.no_grad():
    # Everything is in one batch, so this loop will only happen once
    for i, data in enumerate(testloader):
        maze_loc_goal_ssps, directions, locs, goals = data

        outputs = model(maze_loc_goal_ssps.to(device))

        mse_loss = mse_criterion(outputs, directions.to(device))
        cosine_loss = cosine_criterion(
            outputs,
            directions.to(device),
            torch.ones(maze_loc_goal_ssps.size()[0]).to(device)
        )

        avg_test_mse_loss += mse_loss.data.item()
        avg_test_cosine_loss += cosine_loss.data.item()
        n_test_batches += 1

if n_test_batches > 0:
    avg_test_mse_loss /= n_test_batches
    avg_test_cosine_loss /= n_test_batches
    print(avg_test_mse_loss, avg_test_cosine_loss)
    writer.add_scalar('test_mse_loss', avg_test_mse_loss, args.epochs + args.epoch_offset)
    writer.add_scalar('test_cosine_loss', avg_test_cosine_loss, args.epochs + args.epoch_offset)


print("Visualization")
validation_set.run_validation(model, writer, args.epochs + args.epoch_offset, use_wall_overlay=not args.no_wall_overlay)


# Close tensorboard writer
writer.close()

torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))

params = vars(args)

with open(os.path.join(save_dir, "params.json"), "w") as f:
    json.dump(params, f)
