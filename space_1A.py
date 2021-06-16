from .generate_convs import ConvSettings

from ..conv_branch import ConvBranch
from ..pool_branch import PoolBranch

n_branches = 4


def set_func(layer, in_planes, out_planes):

    layer.branch_0 = ConvBranch(in_planes,
                                out_planes,
                                kernel_size=3,
                                padding=1)
    layer.branch_1 = ConvBranch(in_planes,
                                out_planes,
                                kernel_size=5,
                                padding=2)
    layer.branch_2 = PoolBranch(in_planes, out_planes, 'avg')
    layer.branch_3 = PoolBranch(in_planes, out_planes, 'max')

    layer.conv_settings = ConvSettings(in_planes, out_planes, 3, 1, 1,
                                       1).generate_type_eq_settings(8, 8)

    global n_branches
    n_branches = 4 + len(layer.conv_settings)

    return n_branches


def pick_func(layer, layer_type, x):
    if layer_type == 0:
        out = layer.branch_0(x)
    elif layer_type == 1:
        out = layer.branch_1(x)
    elif layer_type == 2:
        out = layer.branch_2(x)
    elif layer_type == 3:
        out = layer.branch_3(x)
    elif 4 <= layer_type < n_branches:
        out = ConvBranch()

    return out


functions = (set_func, pick_func)
