import numpy as np
from spatial_semantic_pointers.utils import encode_point, make_good_unitary, encode_random
from functools import partial
from ssp_navigation.utils.models import EncodingLayer
import torch


def encode_projection(x, y, dim, seed=13):

    # Use the same rstate every time for a consistent transform
    # NOTE: would be more efficient to save the transform rather than regenerating,
    #       but then it would have to get passed everywhere
    rstate = np.random.RandomState(seed=seed)

    proj = rstate.uniform(low=-1, high=1, size=(2, dim))

    # return np.array([x, y]).reshape(1, 2) @ proj
    return np.dot(np.array([x, y]).reshape(1, 2), proj).reshape(dim)


def encode_trig(x, y, dim):
    # sin and cos with difference spatial frequencies and offsets
    ret = []

    # denominator for normalization
    denom = np.sqrt(dim // 8)

    for i in range(dim // 16):
        for m in [0, .5, 1, 1.5]:
            ret += [np.cos((dim // 8) * (m * np.pi + x) / (i + 1.)) / denom,
                    np.sin((dim // 8) * (m * np.pi + x) / (i + 1.)) / denom,
                    np.cos((dim // 8) * (m * np.pi + y) / (i + 1.)) / denom,
                    np.sin((dim // 8) * (m * np.pi + y) / (i + 1.)) / denom]

    return np.array(ret)


def encode_random_trig(x, y, dim, seed=13):

    rstate = np.random.RandomState(seed=seed)

    freq = rstate.uniform(low=-10, high=10, size=dim)
    phase = rstate.uniform(low=-np.pi, high=np.pi, size=dim)

    ret = np.zeros((dim,))

    for i in range(dim):
        if i % 2 == 0:
            ret[i] = np.sin(x*freq[i] + phase[i])
        else:
            ret[i] = np.sin(y*freq[i] + phase[i])

    # normalize
    ret = ret / np.linalg.norm(ret)

    return ret


# Defining axes of 2D representation with respect to the 3D hexagonal one
x_axis = np.array([1, -1, 0])
y_axis = np.array([-1, -1, 2])
x_axis = x_axis / np.linalg.norm(x_axis)
y_axis = y_axis / np.linalg.norm(y_axis)


# Converts a 2D coordinate into the corresponding
# 3D coordinate in the hexagonal representation
def to_xyz(coord):
    return x_axis*coord[1]+y_axis*coord[0]


def encode_hex_trig(x, y, dim, frequencies=(1, 1.4, 1.4*1.4), seed=13):

    rstate = np.random.RandomState(seed=seed)

    # choose which of the 3 hex axes to use
    axis = rstate.randint(low=0, high=3, size=dim)
    # choose which scaling/frequency to use
    freq = rstate.randint(low=0, high=len(frequencies), size=dim)
    # choose phase offset
    phase = rstate.uniform(low=-np.pi, high=np.pi, size=dim)

    ret = np.zeros((dim,))

    # convert to hexagonal coordinates
    hx, hy, hz = to_xyz((x, y))
    # print(hx, hy, hz)

    for i in range(dim):
        if axis[i] == 0:
            ret[i] = np.sin(hx*frequencies[freq[i]] + phase[i])
        elif axis[i] == 1:
            ret[i] = np.sin(hy*frequencies[freq[i]] + phase[i])
        elif axis[i] == 2:
            ret[i] = np.sin(hz*frequencies[freq[i]] + phase[i])
        else:
            assert False  # this should never happen

    return ret

# # FIXME: simplified for debugging
# def encode_hex_trig(x, y, dim=3, frequencies=(1, 1.4, 1.4*1.4), seed=13):
#
#     ret = np.zeros((3,))
#
#     # convert to hexagonal coordinates
#     hx, hy, hz = to_xyz((x, y))
#     # print(hx, hy, hz)
#
#     ret[0] = np.sin(hx*1 + 0)
#     ret[1] = np.sin(hy*1 + 0)
#     ret[2] = np.sin(hz*1 + 0)
#
#     return ret


# https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def get_pc_gauss_encoding_func(limit_low=0, limit_high=1, dim=512, sigma=0.25, use_softmax=False, rng=np.random):
    # generate PC centers
    pc_centers = rng.uniform(low=limit_low, high=limit_high, size=(dim, 2))

    # TODO: make this more efficient
    def encoding_func(x, y):
        activations = np.zeros((dim,))
        for i in range(dim):
            activations[i] = np.exp(-((pc_centers[i, 0] - x) ** 2 + (pc_centers[i, 1] - y) ** 2) / sigma / sigma)
        if use_softmax:
            return softmax(activations)
        else:
            return activations

    return encoding_func


def get_tile_encoding_func(limit_low=0, limit_high=1, dim=512, n_tiles=8, n_bins=8, rng=np.random):

    assert dim == n_tiles * n_bins * n_bins

    # Make the random offsets relative to the bin_size
    bin_size = (limit_high - limit_low) / (n_bins)

    # A series of shiften linspaces
    xss = np.zeros((n_tiles, n_bins))
    yss = np.zeros((n_tiles, n_bins))

    # The x and y offsets for each tile. Max offset is half the bin_size
    offsets = rng.uniform(-bin_size/2, bin_size/2, size=(n_tiles, 2))

    for i in range(n_tiles):
        xss[i, :] = np.linspace(limit_low + offsets[i, 0], limit_high + offsets[i, 0], n_bins)
        yss[i, :] = np.linspace(limit_low + offsets[i, 1], limit_high + offsets[i, 1], n_bins)

    def encoding_func(x, y):
        arr = np.zeros((n_tiles, n_bins, n_bins))
        for i in range(n_tiles):
            indx = (np.abs(xss[i, :] - x)).argmin()
            indy = (np.abs(yss[i, :] - y)).argmin()
            arr[i, indx, indy] = 1
        return arr.flatten()

    return encoding_func


def encode_one_hot(x, y, xs, ys):
    arr = np.zeros((len(xs), len(ys)))
    indx = (np.abs(xs - x)).argmin()
    indy = (np.abs(ys - y)).argmin()
    arr[indx, indy] = 1

    return arr.flatten()


def get_ssp_encode_func(dim, seed, scaling=1.0):
    """
    Generates an encoding function for SSPs that only takes (x,y) as input
    :param dim: dimension of the SSP
    :param seed: seed for randomly choosing axis vectors
    :param scaling: scaling the resolution of the space
    :return:
    """
    rng = np.random.RandomState(seed=seed)
    x_axis_sp = make_good_unitary(dim=dim, rng=rng)
    y_axis_sp = make_good_unitary(dim=dim, rng=rng)

    def encode_ssp(x, y):
        return encode_point(x * scaling, y * scaling, x_axis_sp, y_axis_sp).v

    return encode_ssp


def get_one_hot_encode_func(dim=512, limit_low=0, limit_high=13):

    optimal_side = int(np.floor(np.sqrt(dim)))

    if optimal_side != np.sqrt(dim):
        print("Warning, could not evenly square {}D for one hot encoding, total dimension is now {}D".format(
            dim, optimal_side*optimal_side)
        )

    xs = np.linspace(limit_low, limit_high, optimal_side)
    ys = np.linspace(limit_low, limit_high, optimal_side)

    def encoding_func(x, y):
        arr = np.zeros((len(xs), len(ys)))
        indx = (np.abs(xs - x)).argmin()
        indy = (np.abs(ys - y)).argmin()
        arr[indx, indy] = 1

        return arr.flatten()

    return encoding_func


def encoding_func_from_model(path, dim=512):

    encoding_layer = EncodingLayer(input_size=2, encoding_size=dim)

    # TODO: modify this to have a network that just does the encoding
    # TODO: make sure this is working correctly
    print("Loading learned first layer parameters from pretrained model")
    state_dict = torch.load(path)

    for name, param in state_dict.items():
        if name in ['encoding_layer.weight', 'encoding_layer.bias']:
            encoding_layer.state_dict()[name].copy_(param)

    print("Freezing first layer parameters for training")
    for name, param in encoding_layer.named_parameters():
        if name in ['encoding_layer.weight', 'encoding_layer.bias']:
            param.requires_grad = False
        if param.requires_grad:
            print(name)

    # def encoding_func(x, y):
    def encoding_func(positions):
        return encoding_layer(torch.Tensor(positions)).detach().numpy()

    return encoding_func


def get_encoding_function(args, limit_low=0, limit_high=13):
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
        repr_dim = args.dim
        encoding_func = get_ssp_encode_func(args.dim, args.seed, scaling=args.ssp_scaling)
    elif args.spatial_encoding == 'one-hot':
        repr_dim = int(np.sqrt(args.dim)) ** 2
        encoding_func = get_one_hot_encode_func(dim=args.dim, limit_low=limit_low, limit_high=limit_high)
    elif args.spatial_encoding == 'trig':
        repr_dim = args.dim
        encoding_func = partial(encode_trig, dim=args.dim)
    elif args.spatial_encoding == 'random-trig':
        repr_dim = args.dim
        encoding_func = partial(encode_random_trig, dim=args.dim, seed=args.seed)
    elif args.spatial_encoding == 'hex-trig':
        repr_dim = args.dim
        encoding_func = partial(
            encode_hex_trig,
            dim=args.dim, seed=args.seed,
            frequencies=(args.hex_freq_coef, args.hex_freq_coef * 1.4, args.hex_freq_coef * 1.4 * 1.4)
        )
    elif args.spatial_encoding == 'pc-gauss':
        repr_dim = args.dim
        rng = np.random.RandomState(seed=args.seed)
        encoding_func = get_pc_gauss_encoding_func(
            limit_low=limit_low, limit_high=limit_high, dim=args.dim, sigma=args.pc_gauss_sigma,
            use_softmax=False, rng=rng
        )
    elif args.spatial_encoding == 'tile-coding':
        repr_dim = args.dim
        rng = np.random.RandomState(seed=args.seed)
        encoding_func = get_tile_encoding_func(
            limit_low=limit_low, limit_high=limit_high,
            dim=args.dim, n_tiles=args.n_tiles, n_bins=args.n_bins, rng=rng
        )
    elif args.spatial_encoding == 'random-proj':
        repr_dim = args.dim
        encoding_func = partial(encode_projection, dim=args.dim, seed=args.seed)
    elif args.spatial_encoding == 'random':
        repr_dim = args.dim
        encoding_func = partial(encode_random, dim=args.dim)
    else:
        raise NotImplementedError

    return encoding_func, repr_dim
