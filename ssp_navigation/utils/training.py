import nengo.spa as spa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from spatial_semantic_pointers.utils import encode_point, encode_random, ssp_to_loc, ssp_to_loc_v, get_heatmap_vectors
from spatial_semantic_pointers.plots import plot_predictions, plot_predictions_v
from ssp_navigation.utils.datasets import MazeDataset
from ssp_navigation.utils.encodings import encode_one_hot, encode_projection, encode_trig, encode_random_trig, encode_hex_trig
from ssp_navigation.utils.path import plot_path_predictions, plot_path_predictions_image
import matplotlib.pyplot as plt


class PolicyValidationSet(object):

    def __init__(self, data, dim, maze_sps, maze_indices, goal_indices, subsample=2,
                 encoding_func=None, device=None,
                 # spatial_encoding='ssp',
                 ):
        # x_axis_sp = spa.SemanticPointer(data=data['x_axis_sp'])
        # y_axis_sp = spa.SemanticPointer(data=data['y_axis_sp'])

        # Either cpu or cuda
        self.device = device

        # n_mazes by res by res
        fine_mazes = data['fine_mazes']

        # n_mazes by n_goals by res by res by 2
        solved_mazes = data['solved_mazes']

        # NOTE: this can be modified from the original dataset, so it is explicitly passed in
        # n_mazes by dim
        # maze_sps = data['maze_sps']

        # n_mazes by n_goals by 2
        goals = data['goals']

        n_mazes = data['goal_sps'].shape[0]
        n_goals = data['goal_sps'].shape[1]
        # dim = data['goal_sps'].shape[2]

        # NOTE: this code is assuming xs as ys are the same
        assert(np.all(data['xs'] == data['ys']))
        limit_low = data['xs'][0]
        limit_high = data['xs'][1]

        # # NOTE: only used for one-hot encoded location representation case
        # xso = np.linspace(limit_low, limit_high, int(np.sqrt(dim)))
        # yso = np.linspace(limit_low, limit_high, int(np.sqrt(dim)))

        goal_sps = np.zeros((n_mazes, n_goals, dim))
        for ni in range(goal_sps.shape[0]):
            for gi in range(goal_sps.shape[1]):
                goal_sps[ni, gi, :] = encoding_func(x=goals[ni, gi, 0], y=goals[ni, gi, 1])

        # # n_mazes by n_goals by dim
        # if spatial_encoding == 'ssp':
        #     goal_sps = data['goal_sps']
        # elif spatial_encoding == 'random':
        #     goal_sps = np.zeros_like(data['goal_sps'])
        #     for ni in range(goal_sps.shape[0]):
        #         for gi in range(goal_sps.shape[1]):
        #             goal_sps[ni, gi, :] = encode_random(x=goals[ni, gi, 0], y=goals[ni, gi, 1], dim=dim)
        # elif spatial_encoding == '2d' or spatial_encoding == 'learned' or spatial_encoding == 'frozen-learned':
        #     goal_sps = goals.copy()
        # elif spatial_encoding == '2d-normalized':
        #     goal_sps = goals.copy()
        #     goal_sps = ((goal_sps - xso[0]) * 2 / (xso[-1] - xso[0])) - 1
        # elif spatial_encoding == 'one-hot':
        #     goal_sps = np.zeros((n_mazes, n_goals, len(xso) * len(yso)))
        #     for ni in range(goal_sps.shape[0]):
        #         for gi in range(goal_sps.shape[1]):
        #             goal_sps[ni, gi, :] = encode_one_hot(x=goals[ni, gi, 0], y=goals[ni, gi, 1], xs=xso, ys=yso)
        # elif spatial_encoding == 'trig':
        #     goal_sps = np.zeros((n_mazes, n_goals, dim))
        #     for ni in range(goal_sps.shape[0]):
        #         for gi in range(goal_sps.shape[1]):
        #             goal_sps[ni, gi, :] = encode_trig(x=goals[ni, gi, 0], y=goals[ni, gi, 1], dim=dim)
        # elif spatial_encoding == 'random-trig':
        #     goal_sps = np.zeros((n_mazes, n_goals, dim))
        #     for ni in range(goal_sps.shape[0]):
        #         for gi in range(goal_sps.shape[1]):
        #             goal_sps[ni, gi, :] = encode_random_trig(x=goals[ni, gi, 0], y=goals[ni, gi, 1], dim=dim)
        # elif spatial_encoding == 'hex-trig':
        #     goal_sps = np.zeros((n_mazes, n_goals, dim))
        #     for ni in range(goal_sps.shape[0]):
        #         for gi in range(goal_sps.shape[1]):
        #             goal_sps[ni, gi, :] = encode_hex_trig(x=goals[ni, gi, 0], y=goals[ni, gi, 1], dim=dim) #TODO: add frequency option
        # elif spatial_encoding == 'random-proj':
        #     goal_sps = np.zeros((n_mazes, n_goals, dim))
        #     for ni in range(goal_sps.shape[0]):
        #         for gi in range(goal_sps.shape[1]):
        #             goal_sps[ni, gi, :] = encode_projection(x=goals[ni, gi, 0], y=goals[ni, gi, 1], dim=dim)
        # else:
        #     raise NotImplementedError

        self.xs = data['xs']
        self.ys = data['ys']

        self.maze_indices = maze_indices
        self.goal_indices = goal_indices
        self.n_mazes = len(maze_indices)
        self.n_goals = len(goal_indices)

        res = fine_mazes.shape[1]
        dim = goal_sps.shape[2]
        n_samples = int(res/subsample) * int(res/subsample) * self.n_mazes * self.n_goals

        # Visualization
        viz_locs = np.zeros((n_samples, 2))
        viz_goals = np.zeros((n_samples, 2))
        viz_loc_sps = np.zeros((n_samples, goal_sps.shape[2]))
        viz_goal_sps = np.zeros((n_samples, goal_sps.shape[2]))
        viz_output_dirs = np.zeros((n_samples, 2))
        viz_maze_sps = np.zeros((n_samples, maze_sps.shape[1]))

        # Generate data so each batch contains a single maze and goal
        si = 0  # sample index, increments each time
        for mi in maze_indices:
            for gi in goal_indices:
                for xi in range(0, res, subsample):
                    for yi in range(0, res, subsample):
                        loc_x = self.xs[xi]
                        loc_y = self.ys[yi]

                        viz_locs[si, 0] = loc_x
                        viz_locs[si, 1] = loc_y
                        viz_goals[si, :] = goals[mi, gi, :]

                        viz_loc_sps[si, :] = encoding_func(x=loc_x, y=loc_y)

                        # if spatial_encoding == 'ssp':
                        #     viz_loc_sps[si, :] = encode_point(loc_x, loc_y, x_axis_sp, y_axis_sp).v
                        # elif spatial_encoding == 'random':
                        #     viz_loc_sps[si, :] = encode_random(loc_x, loc_y, dim)
                        # elif spatial_encoding == '2d' or spatial_encoding == 'learned' or spatial_encoding == 'frozen-learned':
                        #     viz_loc_sps[si, :] = np.array([loc_x, loc_y])
                        # elif spatial_encoding == '2d-normalized':
                        #     viz_loc_sps[si, :] = ((np.array([loc_x, loc_y]) - limit_low)*2 / (limit_high - limit_low)) - 1
                        # elif spatial_encoding == 'one-hot':
                        #     viz_loc_sps[si, :] = encode_one_hot(x=loc_x, y=loc_y, xs=xso, ys=yso)
                        # elif spatial_encoding == 'trig':
                        #     viz_loc_sps[si, :] = encode_trig(x=loc_x, y=loc_y, dim=dim)
                        # elif spatial_encoding == 'random-trig':
                        #     viz_loc_sps[si, :] = encode_random_trig(x=loc_x, y=loc_y, dim=dim)
                        # elif spatial_encoding == 'hex-trig':
                        #     viz_loc_sps[si, :] = encode_hex_trig(x=loc_x, y=loc_y, dim=dim)
                        # elif spatial_encoding == 'random-proj':
                        #     viz_loc_sps[si, :] = encode_projection(x=loc_x, y=loc_y, dim=dim)

                        viz_goal_sps[si, :] = goal_sps[mi, gi, :]

                        viz_output_dirs[si, :] = solved_mazes[mi, gi, xi, yi, :]

                        viz_maze_sps[si, :] = maze_sps[mi]

                        si += 1

        self.batch_size = int(si / (self.n_mazes * self.n_goals))

        print("Visualization Data Generated")
        print("Total Samples: {}".format(si))
        print("Mazes: {}".format(self.n_mazes))
        print("Goals: {}".format(self.n_goals))
        print("Batch Size: {}".format(self.batch_size))
        print("Batches: {}".format(self.n_mazes * self.n_goals))

        dataset_viz = MazeDataset(
            maze_ssp=viz_maze_sps,
            loc_ssps=viz_loc_sps,
            goal_ssps=viz_goal_sps,
            locs=viz_locs,
            goals=viz_goals,
            direction_outputs=viz_output_dirs,
        )

        # Each batch will contain the samples for one maze. Must not be shuffled
        self.vizloader = torch.utils.data.DataLoader(
            dataset_viz, batch_size=self.batch_size, shuffle=False, num_workers=0,
        )

    def run_ground_truth(self, writer):

        with torch.no_grad():
            # Each maze is in one batch
            for i, data in enumerate(self.vizloader):
                print("Ground Truth Viz batch {} of {}".format(i + 1, self.n_mazes * self.n_goals))
                maze_loc_goal_ssps, directions, locs, goals = data

                wall_overlay = (directions.detach().numpy()[:, 0] == 0) & (directions.detach().numpy()[:, 1] == 0)

                fig_truth, rmse = plot_path_predictions_image(
                    directions=directions.detach().numpy(), coords=locs.detach().numpy(), wall_overlay=wall_overlay
                )

                # fig_truth_quiver = plot_path_predictions(
                #     directions=directions.detach().numpy(), coords=locs.detach().numpy(), dcell=self.xs[1] - self.xs[0]
                # )

                # Record figures to tensorboard
                writer.add_figure('v{}/ground truth'.format(i), fig_truth)
                # writer.add_figure('v{}/ground truth quiver'.format(i), fig_truth_quiver)

    # Note that this must be a separate function because the previous cannot contain yields
    def run_ground_truth_generator(self):

        with torch.no_grad():
            # Each maze is in one batch
            for i, data in enumerate(self.vizloader):
                print("Ground Truth Viz batch {} of {}".format(i + 1, self.n_mazes * self.n_goals))
                maze_loc_goal_ssps, directions, locs, goals = data

                wall_overlay = (directions.detach().numpy()[:, 0] == 0) & (directions.detach().numpy()[:, 1] == 0)

                fig_truth, rmse = plot_path_predictions_image(
                    directions=directions.detach().numpy(), coords=locs.detach().numpy(), wall_overlay=wall_overlay
                )

                fig_truth_quiver = plot_path_predictions(
                    directions=directions.detach().numpy(), coords=locs.detach().numpy(), dcell=self.xs[1] - self.xs[0]
                )

                yield fig_truth, fig_truth_quiver

    def run_validation(self, model, writer, epoch, use_wall_overlay=True):
        criterion = nn.MSELoss()

        with torch.no_grad():
            # Each maze is in one batch
            for i, data in enumerate(self.vizloader):
                print("Viz batch {} of {}".format(i + 1, self.n_mazes * self.n_goals))
                maze_loc_goal_ssps, directions, locs, goals = data

                outputs = model(maze_loc_goal_ssps.to(self.device))

                loss = criterion(outputs, directions.to(self.device))

                if use_wall_overlay:

                    wall_overlay = (directions.detach().numpy()[:, 0] == 0) & (directions.detach().numpy()[:, 1] == 0)

                    fig_pred, rmse = plot_path_predictions_image(
                        directions=outputs.detach().cpu().numpy(), coords=locs.detach().cpu().numpy(), wall_overlay=wall_overlay
                    )

                    # fig_pred_quiver = plot_path_predictions(
                    #     directions=outputs.detach().numpy(), coords=locs.detach().numpy(), dcell=self.xs[1] - self.xs[0], wall_overlay=wall_overlay
                    # )

                    writer.add_scalar(tag='viz_rmse/{}'.format(i), scalar_value=rmse, global_step=epoch)
                else:

                    fig_pred = plot_path_predictions(
                        directions=outputs.detach().cpu().numpy(), coords=locs.detach().cpu().numpy(), type='colour'
                    )

                    # fig_pred_quiver = plot_path_predictions(
                    #     directions=outputs.detach().numpy(), coords=locs.detach().numpy(), dcell=self.xs[1] - self.xs[0]
                    # )

                # Record figures to tensorboard
                writer.add_figure('v{}/viz set predictions'.format(i), fig_pred, epoch)
                # writer.add_figure('v{}/viz set predictions quiver'.format(i), fig_pred_quiver, epoch)
                writer.add_scalar(tag='viz_loss/{}'.format(i), scalar_value=loss.data.item(), global_step=epoch)

    # Note that this must be a separate function because the previous cannot contain yields
    def run_validation_generator(self, model, epoch, use_wall_overlay=True):
        criterion = nn.MSELoss()

        with torch.no_grad():
            # Each maze is in one batch
            for i, data in enumerate(self.vizloader):
                print("Viz batch {} of {}".format(i + 1, self.n_mazes * self.n_goals))
                maze_loc_goal_ssps, directions, locs, goals = data

                outputs = model(maze_loc_goal_ssps.to(self.device))

                loss = criterion(outputs, directions)

                if use_wall_overlay:

                    wall_overlay = (directions.detach().numpy()[:, 0] == 0) & (directions.detach().numpy()[:, 1] == 0)

                    fig_pred, rmse = plot_path_predictions_image(
                        directions=outputs.detach().numpy(), coords=locs.detach().numpy(), wall_overlay=wall_overlay
                    )

                    fig_pred_quiver = plot_path_predictions(
                        directions=outputs.detach().numpy(), coords=locs.detach().numpy(), dcell=self.xs[1] - self.xs[0], wall_overlay=wall_overlay
                    )
                else:

                    fig_pred = plot_path_predictions(
                        directions=outputs.detach().numpy(), coords=locs.detach().numpy(), type='colour'
                    )

                    fig_pred_quiver = plot_path_predictions(
                        directions=outputs.detach().numpy(), coords=locs.detach().numpy(), dcell=self.xs[1] - self.xs[0]
                    )

                yield fig_pred, fig_pred_quiver

    def get_rmse(self, model):

        ret = np.zeros((self.n_mazes * self.n_goals,))

        with torch.no_grad():
            # Each maze is in one batch
            for i, data in enumerate(self.vizloader):
                print("Viz batch {} of {}".format(i + 1, self.n_mazes * self.n_goals))
                maze_loc_goal_ssps, directions, locs, goals = data

                outputs = model(maze_loc_goal_ssps.to(self.device))

                wall_overlay = (directions.detach().numpy()[:, 0] == 0) & (directions.detach().numpy()[:, 1] == 0)

                fig_pred, rmse = plot_path_predictions_image(
                    directions=outputs.detach().cpu().numpy(), coords=locs.detach().cpu().numpy(), wall_overlay=wall_overlay
                )

                ret[i] = rmse

        return ret

def create_policy_dataloader(data, n_samples, maze_sps, args, encoding_func, pin_memory=False):
    # x_axis_sp = spa.SemanticPointer(data=data['x_axis_sp'])
    # y_axis_sp = spa.SemanticPointer(data=data['y_axis_sp'])

    # n_mazes by size by size
    # coarse_mazes = data['coarse_mazes']

    # n_mazes by res by res
    fine_mazes = data['fine_mazes']

    # n_mazes by res by res by 2
    solved_mazes = data['solved_mazes']

    # NOTE: this can be modified from the original dataset, so it is explicitly passed in
    # n_mazes by dim
    # maze_sps = data['maze_sps']

    # n_mazes by n_goals by 2
    goals = data['goals']

    n_goals = goals.shape[1]
    n_mazes = fine_mazes.shape[0]

    # NOTE: only used for one-hot encoded location representation case
    xs = data['xs']
    ys = data['ys']
    xso = np.linspace(xs[0], xs[-1], int(np.sqrt(args.dim)))
    yso = np.linspace(ys[0], ys[-1], int(np.sqrt(args.dim)))

    # n_mazes by n_goals by dim
    if args.spatial_encoding == '2d' or args.spatial_encoding == 'learned' or args.spatial_encoding == 'frozen-learned':
        goal_sps = goals.copy()
    elif args.spatial_encoding == '2d-normalized':
        goal_sps = goals.copy()
        goal_sps = ((goal_sps - xso[0]) * 2 / (xso[-1] - xso[0])) - 1
    else:
        goal_sps = np.zeros((n_mazes, n_goals, args.dim))
        for ni in range(n_mazes):
            for gi in range(n_goals):
                goal_sps[ni, gi, :] = encoding_func(x=goals[ni, gi, 0], y=goals[ni, gi, 1])

    if 'xs' in data.keys():
        xs = data['xs']
        ys = data['ys']
    else:
        # backwards compatibility
        xs = np.linspace(args.limit_low, args.limit_high, args.res)
        ys = np.linspace(args.limit_low, args.limit_high, args.res)

    free_spaces = np.argwhere(fine_mazes == 0)
    n_free_spaces = free_spaces.shape[0]

    # Training
    train_locs = np.zeros((n_samples, 2))
    train_goals = np.zeros((n_samples, 2))
    train_loc_sps = np.zeros((n_samples, goal_sps.shape[2]))
    train_goal_sps = np.zeros((n_samples, goal_sps.shape[2]))
    train_output_dirs = np.zeros((n_samples, 2))
    train_maze_sps = np.zeros((n_samples, maze_sps.shape[1]))

    train_indices = np.random.randint(low=0, high=n_free_spaces, size=n_samples)

    for n in range(n_samples):
        # print("Sample {} of {}".format(n + 1, n_samples))

        # n_mazes by res by res
        indices = free_spaces[train_indices[n], :]
        maze_index = indices[0]
        x_index = indices[1]
        y_index = indices[2]
        goal_index = np.random.randint(low=0, high=n_goals)

        # 2D coordinate of the agent's current location
        loc_x = xs[x_index]
        loc_y = ys[y_index]

        train_locs[n, 0] = loc_x
        train_locs[n, 1] = loc_y
        train_goals[n, :] = goals[maze_index, goal_index, :]

        train_loc_sps[n, :] = encoding_func(x=loc_x, y=loc_y)

        # if args.spatial_encoding == 'ssp':
        #     train_loc_sps[n, :] = encode_point(loc_x, loc_y, x_axis_sp, y_axis_sp).v
        # elif args.spatial_encoding == 'random':
        #     train_loc_sps[n, :] = encode_random(loc_x, loc_y, args.dim)
        # elif args.spatial_encoding == '2d' or args.spatial_encoding == 'learned' or args.spatial_encoding == 'frozen-learned':
        #     train_loc_sps[n, :] = np.array([loc_x, loc_y])
        # elif args.spatial_encoding == '2d-normalized':
        #     train_loc_sps[n, :] = ((np.array([loc_x, loc_y]) - xso[0]) * 2 / (xso[-1] - xso[0])) - 1
        # elif args.spatial_encoding == 'one-hot':
        #     train_loc_sps[n, :] = encode_one_hot(x=loc_x, y=loc_y, xs=xso, ys=yso)
        # elif args.spatial_encoding == 'trig':
        #     train_loc_sps[n, :] = encode_trig(x=loc_x, y=loc_y, dim=args.dim)
        # elif args.spatial_encoding == 'random-trig':
        #     train_loc_sps[n, :] = encode_random_trig(x=loc_x, y=loc_y, dim=args.dim)
        # elif args.spatial_encoding == 'hex-trig':
        #     train_loc_sps[n, :] = encode_hex_trig(x=loc_x, y=loc_y, dim=args.dim)
        # elif args.spatial_encoding == 'random-proj':
        #     train_loc_sps[n, :] = encode_projection(x=loc_x, y=loc_y, dim=args.dim)
        train_goal_sps[n, :] = goal_sps[maze_index, goal_index, :]

        train_output_dirs[n, :] = solved_mazes[maze_index, goal_index, x_index, y_index, :]

        train_maze_sps[n, :] = maze_sps[maze_index]

    dataset_train = MazeDataset(
        maze_ssp=train_maze_sps,
        loc_ssps=train_loc_sps,
        goal_ssps=train_goal_sps,
        locs=train_locs,
        goals=train_goals,
        direction_outputs=train_output_dirs,
    )

    trainloader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory
    )

    return trainloader


class TrajectoryValidationSet(object):

    def __init__(self, dataloader, heatmap_vectors, xs, ys, ssp_scaling=1, spatial_encoding='ssp'):

        self.dataloader = dataloader
        self.heatmap_vectors = heatmap_vectors
        self.xs = xs
        self.ys = ys
        self.ssp_scaling = ssp_scaling
        self.spatial_encoding = spatial_encoding
        self.cosine_criterion = nn.CosineEmbeddingLoss()
        self.mse_criterion = nn.MSELoss()

    def run_eval(self, model, writer, epoch):

        with torch.no_grad():
            # Everything is in one batch, so this loop will only happen once
            for i, data in enumerate(self.dataloader):
                combined_inputs, ssp_inputs, ssp_outputs = data

                ssp_pred = model(combined_inputs, ssp_inputs)

                # NOTE: need to permute axes of the targets here because the output is
                #       (sequence length, batch, units) instead of (batch, sequence_length, units)
                #       could also permute the outputs instead
                # NOTE: for cosine loss the input needs to be flattened first
                cosine_loss = self.cosine_criterion(
                    ssp_pred.reshape(ssp_pred.shape[0] * ssp_pred.shape[1], ssp_pred.shape[2]),
                    ssp_outputs.permute(1, 0, 2).reshape(ssp_pred.shape[0] * ssp_pred.shape[1], ssp_pred.shape[2]),
                    torch.ones(ssp_pred.shape[0] * ssp_pred.shape[1])
                )
                mse_loss = self.mse_criterion(ssp_pred, ssp_outputs.permute(1, 0, 2))

                print("test mse loss", mse_loss.data.item())
                print("test cosine loss", mse_loss.data.item())

            writer.add_scalar('test_mse_loss', mse_loss.data.item(), epoch)
            writer.add_scalar('test_cosine_loss', cosine_loss.data.item(), epoch)

            # Just use start and end location to save on memory and computation
            predictions_start = np.zeros((ssp_pred.shape[1], 2))
            coords_start = np.zeros((ssp_pred.shape[1], 2))

            predictions_end = np.zeros((ssp_pred.shape[1], 2))
            coords_end = np.zeros((ssp_pred.shape[1], 2))

            if self.spatial_encoding == 'ssp':
                print("computing prediction locations")
                predictions_start[:, :] = ssp_to_loc_v(
                    ssp_pred.detach().numpy()[0, :, :],
                    self.heatmap_vectors, self.xs, self.ys
                )
                predictions_end[:, :] = ssp_to_loc_v(
                    ssp_pred.detach().numpy()[-1, :, :],
                    self.heatmap_vectors, self.xs, self.ys
                )
                print("computing ground truth locations")
                coords_start[:, :] = ssp_to_loc_v(
                    ssp_outputs.detach().numpy()[:, 0, :],
                    self.heatmap_vectors, self.xs, self.ys
                )
                coords_end[:, :] = ssp_to_loc_v(
                    ssp_outputs.detach().numpy()[:, -1, :],
                    self.heatmap_vectors, self.xs, self.ys
                )
            elif self.spatial_encoding == '2d':
                print("copying prediction locations")
                predictions_start[:, :] = ssp_pred.detach().numpy()[0, :, :]
                predictions_end[:, :] = ssp_pred.detach().numpy()[-1, :, :]
                print("copying ground truth locations")
                coords_start[:, :] = ssp_outputs.detach().numpy()[:, 0, :]
                coords_end[:, :] = ssp_outputs.detach().numpy()[:, -1, :]

            fig_pred_start, ax_pred_start = plt.subplots()
            fig_truth_start, ax_truth_start = plt.subplots()
            fig_pred_end, ax_pred_end = plt.subplots()
            fig_truth_end, ax_truth_end = plt.subplots()

            print("plotting predicted locations")
            plot_predictions_v(
                predictions_start / self.ssp_scaling,
                coords_start / self.ssp_scaling,
                ax_pred_start,
                min_val=self.xs[0],
                max_val=self.xs[-1],
            )
            plot_predictions_v(
                predictions_end / self.ssp_scaling,
                coords_end / self.ssp_scaling,
                ax_pred_end,
                min_val=self.xs[0],
                max_val=self.xs[-1],
            )

            writer.add_figure("predictions start", fig_pred_start, epoch)
            writer.add_figure("predictions end", fig_pred_end, epoch)

            # Only plotting ground truth if the epoch is 0
            if epoch == 0:

                print("plotting ground truth locations")
                plot_predictions_v(
                    coords_start / self.ssp_scaling,
                    coords_start / self.ssp_scaling,
                    ax_truth_start,
                    min_val=self.xs[0],
                    max_val=self.xs[-1],
                )
                plot_predictions_v(
                    coords_end / self.ssp_scaling,
                    coords_end / self.ssp_scaling,
                    ax_truth_end,
                    min_val=self.xs[0],
                    max_val=self.xs[-1],
                )

                writer.add_figure("ground truth start", fig_truth_start, epoch)
                writer.add_figure("ground truth end", fig_truth_end, epoch)


class SnapshotValidationSet(object):

    def __init__(self, dataloader, heatmap_vectors, xs, ys, spatial_encoding='ssp'):

        self.dataloader = dataloader
        self.heatmap_vectors = heatmap_vectors
        self.xs = xs
        self.ys = ys
        self.spatial_encoding = spatial_encoding
        self.cosine_criterion = nn.CosineEmbeddingLoss()
        self.mse_criterion = nn.MSELoss()

    def run_eval(self, model, writer, epoch):

        with torch.no_grad():
            # Everything is in one batch, so this loop will only happen once
            for i, data in enumerate(self.dataloader):
                # sensor_inputs, map_ids, ssp_outputs = data
                # sensors and map ID combined
                combined_inputs, ssp_outputs = data

                # ssp_pred = model(sensor_inputs, map_ids)
                ssp_pred = model(combined_inputs)

                cosine_loss = self.cosine_criterion(ssp_pred, ssp_outputs, torch.ones(ssp_pred.shape[0]))
                mse_loss = self.mse_criterion(ssp_pred, ssp_outputs)

                print("test mse loss", mse_loss.data.item())
                print("test cosine loss", mse_loss.data.item())

            writer.add_scalar('test_mse_loss', mse_loss.data.item(), epoch)
            writer.add_scalar('test_cosine_loss', cosine_loss.data.item(), epoch)

            # One prediction and ground truth coord for every element in the batch
            # NOTE: this is assuming the eval set only has one giant batch
            predictions = np.zeros((ssp_pred.shape[0], 2))
            coords = np.zeros((ssp_pred.shape[0], 2))

            if self.spatial_encoding == 'ssp':
                print("computing prediction locations")
                predictions[:, :] = ssp_to_loc_v(
                    ssp_pred.detach().numpy()[:, :],
                    self.heatmap_vectors, self.xs, self.ys
                )

                print("computing ground truth locations")
                coords[:, :] = ssp_to_loc_v(
                    ssp_outputs.detach().numpy()[:, :],
                    self.heatmap_vectors, self.xs, self.ys
                )

            elif self.spatial_encoding == '2d':
                print("copying prediction locations")
                predictions[:, :] = ssp_pred.detach().numpy()[:, :]
                print("copying ground truth locations")
                coords[:, :] = ssp_outputs.detach().numpy()[:, :]

            fig_pred, ax_pred = plt.subplots()
            fig_truth, ax_truth = plt.subplots()

            print("plotting predicted locations")
            plot_predictions_v(
                # predictions / self.ssp_scaling,
                # coords / self.ssp_scaling,
                predictions,
                coords,
                ax_pred,
                # min_val=0,
                # max_val=2.2
                min_val=self.xs[0],
                max_val=self.xs[-1],
            )

            writer.add_figure("predictions", fig_pred, epoch)

            # Only plot ground truth if epoch is 0
            if epoch == 0:
                print("plotting ground truth locations")
                plot_predictions_v(
                    # coords / self.ssp_scaling,
                    # coords / self.ssp_scaling,
                    coords,
                    coords,
                    ax_truth,
                    # min_val=0,
                    # max_val=2.2
                    min_val=self.xs[0],
                    max_val=self.xs[-1],
                )

                writer.add_figure("ground truth", fig_truth, epoch)


class LocalizationTrajectoryDataset(data.Dataset):

    def __init__(self, velocity_inputs, sensor_inputs, maze_ids, ssp_inputs, ssp_outputs, return_velocity_list=True):

        self.velocity_inputs = velocity_inputs.astype(np.float32)
        self.sensor_inputs = sensor_inputs.astype(np.float32)
        self.maze_ids = maze_ids.astype(np.float32)

        # self.combined_inputs = np.hstack([self.velocity_inputs, self.sensor_inputs, self.maze_ids])
        self.combined_inputs = np.concatenate([self.velocity_inputs, self.sensor_inputs, self.maze_ids], axis=2)
        assert (self.velocity_inputs.shape[0] == self.combined_inputs.shape[0])
        assert (self.velocity_inputs.shape[1] == self.combined_inputs.shape[1])
        assert (self.combined_inputs.shape[2] == self.velocity_inputs.shape[2] + self.sensor_inputs.shape[2] + self.maze_ids.shape[2])
        self.ssp_inputs = ssp_inputs.astype(np.float32)
        self.ssp_outputs = ssp_outputs.astype(np.float32)

        # flag for whether the velocities returned are a single tensor or a list of tensors
        self.return_velocity_list = return_velocity_list

    def __getitem__(self, index):

        if self.return_velocity_list:
            return [self.combined_inputs[index, i] for i in range(self.combined_inputs.shape[1])], \
                   self.ssp_inputs[index], self.ssp_outputs[index],
        else:
            return self.combined_inputs[index], self.ssp_inputs[index], self.ssp_outputs[index]

    def __len__(self):
        return self.combined_inputs.shape[0]


class LocalizationSnapshotDataset(data.Dataset):

    def __init__(self, sensor_inputs, maze_ids, ssp_outputs):

        self.sensor_inputs = sensor_inputs.astype(np.float32)
        self.maze_ids = maze_ids.astype(np.float32)
        self.ssp_outputs = ssp_outputs.astype(np.float32)
        self.combined_inputs = np.hstack([self.sensor_inputs, self.maze_ids])
        assert(self.sensor_inputs.shape[0] == self.combined_inputs.shape[0])

    def __getitem__(self, index):

        # return self.sensor_inputs[index], self.maze_ids[index], self.ssp_outputs[index]
        return self.combined_inputs[index], self.ssp_outputs[index]

    def __len__(self):
        return self.sensor_inputs.shape[0]


def localization_train_test_loaders(
        data, n_train_samples=1000, n_test_samples=1000,
        rollout_length=100, batch_size=10, encoding='ssp', n_mazes_to_use=0
):

    # Option to use SSPs or the 2D location directly
    assert encoding in ['ssp', '2d', 'pc']

    positions = data['positions']

    dist_sensors = data['dist_sensors']
    n_sensors = dist_sensors.shape[3]

    cartesian_vels = data['cartesian_vels']
    ssps = data['ssps']
    # n_place_cells = data['pc_centers'].shape[0]
    #
    # pc_activations = data['pc_activations']

    # coarse_maps = data['coarse_maps']
    # n_mazes = coarse_maps.shape[0]

    n_mazes = positions.shape[0]
    n_trajectories = positions.shape[1]
    trajectory_length = positions.shape[2]
    dim = ssps.shape[3]

    # Split the trajectories into train and test sets
    train_test_traj_split = n_trajectories // 2

    for test_set, n_samples in enumerate([n_train_samples, n_test_samples]):

        velocity_inputs = np.zeros((n_samples, rollout_length, 2))

        sensor_inputs = np.zeros((n_samples, rollout_length, n_sensors))

        # these include outputs for every time-step
        ssp_outputs = np.zeros((n_samples, rollout_length, dim))

        ssp_inputs = np.zeros((n_samples, dim))

        # for the 2D encoding method
        pos_outputs = np.zeros((n_samples, rollout_length, 2))

        pos_inputs = np.zeros((n_samples, 2))

        # # for the place cell encoding method
        # pc_outputs = np.zeros((n_samples, rollout_length, n_place_cells))
        #
        # pc_inputs = np.zeros((n_samples, n_place_cells))

        maze_ids = np.zeros((n_samples, rollout_length, n_mazes))

        for i in range(n_samples):
            # choose random map
            if n_mazes_to_use <= 0:
                # use all available mazes
                maze_ind = np.random.randint(low=0, high=n_mazes)
            else:
                # use only some mazes
                maze_ind = np.random.randint(low=0, high=n_mazes_to_use)
            # choose random trajectory
            if test_set == 0:
                traj_ind = np.random.randint(low=0, high=train_test_traj_split)
            elif test_set == 1:
                traj_ind = np.random.randint(low=train_test_traj_split, high=n_trajectories)
            # choose random segment of trajectory
            step_ind = np.random.randint(low=0, high=trajectory_length - rollout_length - 1)

            # index of final step of the trajectory
            step_ind_final = step_ind + rollout_length

            velocity_inputs[i, :, :] = cartesian_vels[maze_ind, traj_ind, step_ind:step_ind_final, :]

            sensor_inputs[i, :, :] = dist_sensors[maze_ind, traj_ind, step_ind:step_ind_final, :]

            # ssp output is shifted by one timestep (since it is a prediction of the future by one step)
            ssp_outputs[i, :, :] = ssps[maze_ind, traj_ind, step_ind + 1:step_ind_final + 1, :]
            # initial state of the LSTM is a linear transform of the ground truth ssp
            ssp_inputs[i, :] = ssps[maze_ind, traj_ind, step_ind]

            # for the 2D encoding method
            pos_outputs[i, :, :] = positions[maze_ind, traj_ind, step_ind + 1:step_ind_final + 1, :]
            pos_inputs[i, :] = positions[maze_ind, traj_ind, step_ind]

            # # for the place cell encoding method
            # pc_outputs[i, :, :] = pc_activations[traj_ind, step_ind + 1:step_ind_final + 1, :]
            # pc_inputs[i, :] = pc_activations[traj_ind, step_ind]

            # one-hot maze ID
            maze_ids[i, :, maze_ind] = 1

        if encoding == 'ssp':
            dataset = LocalizationTrajectoryDataset(
                velocity_inputs=velocity_inputs,
                sensor_inputs=sensor_inputs,
                maze_ids=maze_ids,
                ssp_inputs=ssp_inputs,
                ssp_outputs=ssp_outputs,
            )
        elif encoding == '2d':
            dataset = LocalizationTrajectoryDataset(
                velocity_inputs=velocity_inputs,
                sensor_inputs=sensor_inputs,
                maze_ids=maze_ids,
                ssp_inputs=pos_inputs,
                ssp_outputs=pos_outputs,
            )
        # elif encoding == 'pc':
        #     dataset = LocalizationTrajectoryDataset(
        #         velocity_inputs=velocity_inputs,
        #         sensor_inputs=sensor_inputs,
        #         ssp_inputs=pc_inputs,
        #         ssp_outputs=pc_outputs,
        #     )

        if test_set == 0:
            trainloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True, num_workers=0,
            )
        elif test_set == 1:
            testloader = torch.utils.data.DataLoader(
                dataset, batch_size=n_samples, shuffle=True, num_workers=0,
            )

    return trainloader, testloader


# TODO: need to handle multiple mazes still
def snapshot_localization_train_test_loaders(
        data, n_train_samples=1000, n_test_samples=1000, batch_size=10, encoding='ssp', n_mazes_to_use=0
):

    # Option to use SSPs or the 2D location directly
    assert encoding in ['ssp', '2d']

    xs = data['xs']
    ys = data['ys']
    x_axis_vec = data['x_axis_sp']
    y_axis_vec = data['y_axis_sp']
    heatmap_vectors = get_heatmap_vectors(xs, ys, x_axis_vec, y_axis_vec)

    # positions = data['positions']

    # shape is (n_mazes, res, res, n_sensors)
    dist_sensors = data['dist_sensors']

    fine_mazes = data['fine_mazes']

    n_sensors = dist_sensors.shape[3]

    # ssps = data['ssps']

    n_mazes = data['coarse_mazes'].shape[0]
    dim = x_axis_vec.shape[0]

    for test_set, n_samples in enumerate([n_train_samples, n_test_samples]):

        sensor_inputs = np.zeros((n_samples, n_sensors))

        # these include outputs for every time-step
        ssp_outputs = np.zeros((n_samples, dim))

        # for the 2D encoding method
        pos_outputs = np.zeros((n_samples, 2))

        maze_ids = np.zeros((n_samples, n_mazes))

        for i in range(n_samples):
            # choose random maze and position in maze
            if n_mazes_to_use <= 0:
                # use all available mazes
                maze_ind = np.random.randint(low=0, high=n_mazes)
            else:
                # use only some mazes
                maze_ind = np.random.randint(low=0, high=n_mazes_to_use)
            xi = np.random.randint(low=0, high=len(xs))
            yi = np.random.randint(low=0, high=len(ys))
            # Keep choosing position until it is not inside a wall
            while fine_mazes[maze_ind, xi, yi] == 1:
                xi = np.random.randint(low=0, high=len(xs))
                yi = np.random.randint(low=0, high=len(ys))

            sensor_inputs[i, :] = dist_sensors[maze_ind, xi, yi, :]

            ssp_outputs[i, :] = heatmap_vectors[xi, yi, :]

            # one-hot maze ID
            maze_ids[i, maze_ind] = 1

            # for the 2D encoding method
            pos_outputs[i, :] = np.array([xs[xi], ys[yi]])

        if encoding == 'ssp':
            dataset = LocalizationSnapshotDataset(
                sensor_inputs=sensor_inputs,
                maze_ids=maze_ids,
                ssp_outputs=ssp_outputs,
            )
        elif encoding == '2d':
            dataset = LocalizationSnapshotDataset(
                sensor_inputs=sensor_inputs,
                maze_ids=maze_ids,
                ssp_outputs=pos_outputs,
            )

        if test_set == 0:
            trainloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True, num_workers=0,
            )
        elif test_set == 1:
            testloader = torch.utils.data.DataLoader(
                dataset, batch_size=n_samples, shuffle=True, num_workers=0,
            )

    return trainloader, testloader


class LocalizationModel(nn.Module):

    def __init__(self, input_size, lstm_hidden_size=128, linear_hidden_size=512,
                 unroll_length=100, sp_dim=512, dropout_p=0.5):

        super(LocalizationModel, self).__init__()

        self.input_size = input_size  # velocity and sensor measurements
        self.lstm_hidden_size = lstm_hidden_size
        self.linear_hidden_size = linear_hidden_size
        self.unroll_length = unroll_length

        self.sp_dim = sp_dim

        # Full LSTM that can be given the full sequence and produce the full output in one step
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=1
        )

        self.linear = nn.Linear(
            in_features=self.lstm_hidden_size,
            out_features=self.linear_hidden_size,
        )

        self.dropout = nn.Dropout(p=dropout_p)

        self.ssp_output = nn.Linear(
            in_features=self.linear_hidden_size,
            out_features=self.sp_dim
        )

        # Linear transforms for ground truth ssp into initial hidden and cell state of lstm
        self.w_c = nn.Linear(
            in_features=self.sp_dim,
            out_features=self.lstm_hidden_size,
            bias=False,
        )

        self.w_h = nn.Linear(
            in_features=self.sp_dim,
            out_features=self.lstm_hidden_size,
            bias=False,
        )

    def forward(self, inputs, initial_ssp):
        """
        :param inputs: contains both velocity and distance sensor measurments (and potentially map_id)
        :param initial_ssp: SSP ground truth for the start location of the trajectory
        :return: predicted SSP for agent location at the end of the trajectory
        """

        ssp_pred, output = self.forward_activations(inputs, initial_ssp)
        return ssp_pred

    def forward_activations(self, inputs, initial_ssp):
        """Returns the hidden layer activations as well as the prediction"""

        batch_size = inputs[0].shape[0]

        # Compute initial hidden state
        cell_state = self.w_c(initial_ssp)
        hidden_state = self.w_h(initial_ssp)

        vel_sense_inputs = torch.cat(inputs).view(len(inputs), batch_size, -1)

        output, (_, _) = self.lstm(
            vel_sense_inputs,
            (
                hidden_state.view(1, batch_size, self.lstm_hidden_size),
                cell_state.view(1, batch_size, self.lstm_hidden_size)
            )
        )

        features = self.dropout(self.linear(output))

        # TODO: should normalization be used here?
        ssp_pred = self.ssp_output(features)

        return ssp_pred, output


def pc_to_loc_v(pc_activations, centers, jitter=0.01):
    """
    Approximate decoding of place cell activations.
    Rounding to the nearest place cell center. Just to get a sense of whether the output is in the right ballpark
    :param pc_activations: activations of each place cell, of shape (n_samples, n_place_cells)
    :param centers: centers of each place cell, of shape (n_place_cells, 2)
    :param jitter: noise to add to the output, so locations on top of each other can be seen
    :return: array of the 2D coordinates that the place cell activation most closely represents
    """

    n_samples = pc_activations.shape[0]

    indices = np.argmax(pc_activations, axis=1)

    return centers[indices] + np.random.normal(loc=0, scale=jitter, size=(n_samples, 2))
