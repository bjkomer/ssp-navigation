import numpy as np


def encode_projection(x, y, dim, seed=13):

    # Use the same rstate every time for a consistent transform
    # NOTE: would be more efficient to save the transform rather than regenerating,
    #       but then it would have to get passed everywhere
    rstate = np.random.RandomState(seed=seed)

    proj = rstate.uniform(low=-1, high=1, size=(2, dim))

    # return np.array([x, y]).reshape(1, 2) @ proj
    return np.dot(np.array([x, y]).reshape(1, 2), proj)


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


def encode_one_hot(x, y, xs, ys):
    arr = np.zeros((len(xs), len(ys)))
    indx = (np.abs(xs - x)).argmin()
    indy = (np.abs(ys - y)).argmin()
    arr[indx, indy] = 1

    return arr.flatten()
