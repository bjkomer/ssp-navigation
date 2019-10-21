import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np


class FeedForward(nn.Module):
    """
    Single hidden layer feed-forward model
    """

    def __init__(self, input_size=512, hidden_size=512, output_size=512):
        super(FeedForward, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # For compatibility with DeepRL code
        self.feature_dim = hidden_size #output_size #hidden_size #FIXME, this should be hidden size, the downstream code needs modification

        self.input_layer = nn.Linear(self.input_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inputs):

        features = F.relu(self.input_layer(inputs))
        prediction = self.output_layer(features)

        return prediction

    def forward_activations(self, inputs):
        """Returns the hidden layer activations as well as the prediction"""

        features = F.relu(self.input_layer(inputs))
        prediction = self.output_layer(features)

        return prediction, features


class MLP(nn.Module):
    """
    Multi-layer feed-forward model
    """

    def __init__(self, input_size=512, hidden_size=512, output_size=512, n_layers=2):
        super(MLP, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        # Need to use ModuleList rather than a regular Python list so that the module correctly keeps track
        # of the parameters, and allows network.to(device) to work correctly
        self.inner_layers = nn.ModuleList()

        self.input_layer = nn.Linear(self.input_size, self.hidden_size)
        for i in range(self.n_layers - 1):
            self.inner_layers.append(nn.Linear(self.hidden_size, self.hidden_size))
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inputs):

        features = F.relu(self.input_layer(inputs))
        for i in range(self.n_layers - 1):
            features = F.relu(self.inner_layers[i](features))
        prediction = self.output_layer(features)

        return prediction


class LearnedEncoding(nn.Module):

    def __init__(self, input_size=2, encoding_size=512, maze_id_size=512, hidden_size=512, output_size=2, n_layers=1):
        super(LearnedEncoding, self).__init__()

        self.input_size = input_size
        self.encoding_size = encoding_size
        self.maze_id_size = maze_id_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.inner_layers = []

        self.encoding_layer = nn.Linear(self.input_size, self.encoding_size)

        self.input_layer = nn.Linear(self.encoding_size*2 + self.maze_id_size, self.hidden_size)
        for i in range(self.n_layers - 1):
            self.inner_layers.append(nn.Linear(self.hidden_size, self.hidden_size))
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)

    def forward_activations_encoding(self, inputs):
        """Returns the hidden layer activations as well as the prediction"""

        maze_id = inputs[:, :self.maze_id_size]
        loc_pos = inputs[:, self.maze_id_size:self.maze_id_size + self.input_size]
        goal_pos = inputs[:, self.maze_id_size + self.input_size:self.maze_id_size + self.input_size*2]

        loc_encoding = F.relu(self.encoding_layer(loc_pos))
        goal_encoding = F.relu(self.encoding_layer(goal_pos))
        features = F.relu(self.input_layer(torch.cat([maze_id, loc_encoding, goal_encoding], dim=1)))

        for i in range(self.n_layers - 1):
            features = F.relu(self.inner_layers[i](features))
        prediction = self.output_layer(features)

        return prediction, features, loc_encoding, goal_encoding

    def forward(self, inputs):

        return self.forward_activations_encoding(inputs)[0]


class EncodingLayer(nn.Module):

    def __init__(self, input_size=2, encoding_size=512):
        super(EncodingLayer, self).__init__()

        self.input_size = input_size
        self.encoding_size = encoding_size

        self.encoding_layer = nn.Linear(self.input_size, self.encoding_size)

    def forward(self, inputs):

        encoding = F.relu(self.encoding_layer(inputs))

        return encoding


# class LearnedEncoding(nn.Module):
#
#     def __init__(self, input_size=2, encoding_size=512, maze_id_size=512, hidden_size=512, output_size=2):
#         super(LearnedEncoding, self).__init__()
#
#         self.input_size = input_size
#         self.encoding_size = encoding_size
#         self.maze_id_size = maze_id_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#
#         self.encoding_layer = nn.Linear(self.input_size, self.encoding_size)
#         # self.input_layer = nn.Linear(self.encoding_size, self.hidden_size)
#         self.input_layer = nn.Linear(self.encoding_size*2 + self.maze_id_size, self.hidden_size)
#         self.output_layer = nn.Linear(self.hidden_size, self.output_size)
#
#     def forward_activations_encoding(self, inputs):
#         """Returns the hidden layer activations as well as the prediction"""
#
#         maze_id = inputs[:, :self.maze_id_size]
#         loc_pos = inputs[:, self.maze_id_size:self.maze_id_size + self.input_size]
#         goal_pos = inputs[:, self.maze_id_size + self.input_size:self.maze_id_size + self.input_size*2]
#
#         loc_encoding = F.relu(self.encoding_layer(loc_pos))
#         goal_encoding = F.relu(self.encoding_layer(goal_pos))
#         features = F.relu(self.input_layer(torch.cat([maze_id, loc_encoding, goal_encoding], dim=1)))
#         prediction = self.output_layer(features)
#
#         return prediction, features, loc_encoding, goal_encoding
#
#     def forward(self, inputs):
#
#         return self.forward_activations_encoding(inputs)[0]


class TwoLayer(nn.Module):

    def __init__(self, input_size=2, encoding_size=512, hidden_size=512, output_size=2):
        super(TwoLayer, self).__init__()

        self.input_size = input_size
        self.encoding_size = encoding_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.encoding_layer = nn.Linear(self.input_size, self.encoding_size)
        self.input_layer = nn.Linear(self.encoding_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inputs):
        encoding = F.relu(self.encoding_layer(inputs))
        features = F.relu(self.input_layer(encoding))
        prediction = self.output_layer(features)

        return prediction

    def forward_activations(self, inputs):
        """Returns the hidden layer activations as well as the prediction"""

        encoding = F.relu(self.encoding_layer(inputs))
        features = F.relu(self.input_layer(encoding))
        prediction = self.output_layer(features)

        return prediction, features

    def forward_activations_encoding(self, inputs):
        """Returns the hidden layer activations and encoding layer activations, as well as the prediction"""

        encoding = F.relu(self.encoding_layer(inputs))
        features = F.relu(self.input_layer(encoding))
        prediction = self.output_layer(features)

        return prediction, features, encoding


def load_model(model_path, params_path, n_mazes):

    with open(params_path, "r") as f:
        params = json.load(f)

    if params['maze_id_type'] == 'ssp':
        id_size = params['dim']
    elif params['maze_id_type'] == 'one-hot':
        id_size = n_mazes
    else:
        raise NotImplementedError

    # Dimension of location representation is dependent on the encoding used
    if params['spatial_encoding'] == 'ssp':
        repr_dim = params['dim']
    elif params['spatial_encoding'] == 'random':
        repr_dim = params['dim']
    elif params['spatial_encoding'] == '2d':
        repr_dim = 2
    elif params['spatial_encoding'] == 'learned':
        repr_dim = 2
    elif params['spatial_encoding'] == '2d-normalized':
        repr_dim = 2
    elif params['spatial_encoding'] == 'one-hot':
        repr_dim = int(np.sqrt(params['dim'])) ** 2
    elif params['spatial_encoding'] == 'trig':
        repr_dim = params['dim']
    elif params['spatial_encoding'] == 'random-trig':
        repr_dim = params['dim']
    elif params['spatial_encoding'] == 'random-proj':
        repr_dim = params['dim']
    else:
        raise NotImplementedError

    if params['spatial_encoding'] == 'learned':
        # input is maze, loc, goal ssps, output is 2D direction to move
        model = LearnedEncoding(input_size=repr_dim, maze_id_size=id_size, hidden_size=512, output_size=2)
    else:
        # input is maze, loc, goal ssps, output is 2D direction to move
        model = FeedForward(input_size=id_size + repr_dim * 2, output_size=2)

    model.load_state_dict(torch.load(model_path), strict=False)

    return model, params['maze_id_type']
