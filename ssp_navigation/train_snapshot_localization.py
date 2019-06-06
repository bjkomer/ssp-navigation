import matplotlib
import os
# allow code to work on machines without a display or in a screen session
display = os.environ.get('DISPLAY')
if display is None or 'localhost' in display:
    matplotlib.use('agg')

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from tensorboardX import SummaryWriter
import json
from spatial_semantic_pointers.utils import get_heatmap_vectors, ssp_to_loc, ssp_to_loc_v
from ssp_navigation.utils.training import SnapshotValidationSet, snapshot_localization_train_test_loaders, FeedForward


parser = argparse.ArgumentParser('Run 2D supervised localization experiment with snapshots using pytorch')

parser.add_argument('--seed', type=int, default=13)
parser.add_argument('--n-epochs', type=int, default=20)
parser.add_argument('--n-samples', type=int, default=1000)
parser.add_argument('--encoding', type=str, default='ssp', choices=['ssp', '2d', 'pc'])
parser.add_argument('--eval-period', type=int, default=50)
parser.add_argument('--load-saved-model', type=str, default='', help='Saved model to load from')
parser.add_argument('--loss-function', type=str, default='cosine', choices=['cosine', 'mse'])
parser.add_argument('--n-mazes', type=int, default=0, help='number of mazes to use. Set to 0 to use all in the dataset')
parser.add_argument('--batch-size', type=int, default=10)
parser.add_argument('--learning-rate', type=float, default=1e-5, help='Step size multiplier in the RMSProp algorithm')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum parameter of the RMSProp algorithm')

parser.add_argument('--dataset-dir', type=str,
                    default='datasets/mixed_style_20mazes_50goals_64res_13size_13seed/36sensors_360fov')
parser.add_argument('--variant-subfolder', type=str, default='',
                    help='Optional custom subfolder')

args = parser.parse_args()

dataset_file = os.path.join(args.dataset_dir, 'maze_dataset.npz')

variant_folder = '{}_rec{}_lin{}_trajlen{}'.format(
    args.encoding, args.lstm_hidden_size, args.hidden_size, args.trajectory_length
)

if args.variant_subfolder != '':
    variant_folder = os.path.join(variant_folder, args.variant_subfolder)

logdir = os.path.join(args.dataset_dir, 'snapshot_network', variant_folder)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

use_cosine_loss = False
if args.loss_function == 'cosine':
    use_cosine_loss = True

if use_cosine_loss:
    print("Using Cosine Loss")
else:
    print("Using MSE Loss")

current_time = datetime.now().strftime('%b%d_%H-%M-%S')
save_dir = os.path.join(logdir, current_time)
writer = SummaryWriter(log_dir=save_dir)

data = np.load(args.dataset)

x_axis_vec = data['x_axis_sp']
y_axis_vec = data['y_axis_sp']

xs = data['xs']
ys = data['ys']

# dist sensors is (n_mazes, res, res, n_sensors)
n_sensors = data['dist_sensors'].shape[3]
n_mazes = data['coarse_mazes'].shape[0]

if args.encoding == 'ssp':
    dim = 512
elif args.encoding == '2d':
    dim = 2
    # ssp_scaling = 1  # no scaling used for 2D coordinates directly
elif args.encoding == 'pc':
    dim = args.n_place_cells
    # ssp_scaling = 1
else:
    raise NotImplementedError

# Used for visualization of test set performance using pos = ssp_to_loc(sp, heatmap_vectors, xs, ys)
heatmap_vectors = get_heatmap_vectors(xs, ys, x_axis_vec, y_axis_vec)

# n_samples = 5000
n_samples = args.n_samples#1000
batch_size = args.minibatch_size#10
n_epochs = args.n_epochs#20
# n_epochs = 5

# Input is the distance sensor measurements
model = FeedForward(
    input_size=n_sensors + n_mazes,
    hidden_size=512,
    output_size=dim,
)

if args.load_saved_model:
    model.load_state_dict(torch.load(args.load_saved_model), strict=False)

cosine_criterion = nn.CosineEmbeddingLoss()
mse_criterion = nn.MSELoss()

optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

trainloader, testloader = snapshot_localization_train_test_loaders(
    data,
    n_train_samples=n_samples,
    n_test_samples=n_samples,
    batch_size=batch_size,
    encoding=args.encoding,
    n_mazes_to_use=args.n_mazes,
)

validation_set = SnapshotValidationSet(
    dataloader=testloader,
    heatmap_vectors=heatmap_vectors,
    xs=xs,
    ys=ys,
    spatial_encoding=args.encoding,
)

params = vars(args)
with open(os.path.join(save_dir, "params.json"), "w") as f:
    json.dump(params, f)

print("Training")
for epoch in range(n_epochs):
    print("Epoch {} of {}".format(epoch + 1, n_epochs))

    # Every 'eval_period' epochs, create a test loss and image
    if epoch % args.eval_period == 0:

        print("Evaluating")
        validation_set.run_eval(
            model=model,
            writer=writer,
            epoch=epoch,
        )

    avg_mse_loss = 0
    avg_cosine_loss = 0
    n_batches = 0
    for i, data in enumerate(trainloader):
        # sensor_inputs, maze_ids, ssp_outputs = data
        # sensor and map IDs combined
        combined_inputs, ssp_outputs = data

        if combined_inputs.size()[0] != batch_size:
            continue  # Drop data, not enough for a batch
        optimizer.zero_grad()
        # model.zero_grad()

        # ssp_pred = model(sensor_inputs, maze_ids)
        ssp_pred = model(combined_inputs)

        cosine_loss = cosine_criterion(ssp_pred, ssp_outputs, torch.ones(ssp_pred.shape[0]))
        mse_loss = mse_criterion(ssp_pred, ssp_outputs)

        if use_cosine_loss:
            cosine_loss.backward()
        else:
            mse_loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_thresh)

        optimizer.step()

        avg_mse_loss += mse_loss.data.item()
        avg_cosine_loss += cosine_loss.data.item()
        n_batches += 1

    avg_mse_loss /= n_batches
    avg_cosine_loss /= n_batches
    print("mse loss:", avg_mse_loss)
    print("cosine loss:", avg_cosine_loss)
    writer.add_scalar('avg_mse_loss', avg_mse_loss, epoch + 1)
    writer.add_scalar('avg_cosine_loss', avg_cosine_loss, epoch + 1)


print("Testing")
validation_set.run_eval(
    model=model,
    writer=writer,
    epoch=n_epochs,
)

if args.encoding == 'ssp':
    torch.save(model.state_dict(), os.path.join(save_dir, 'ssp_snapshot_localization_model.pt'))
elif args.encoding == '2d':
    torch.save(model.state_dict(), os.path.join(save_dir, '2d_snapshot_localization_model.pt'))
