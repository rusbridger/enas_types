from ..conv_branch import ConvBranch
from ..pool_branch import PoolBranch

n_branches = 4


def set_func(layer, in_planes, out_planes):
    layer.branch_0 = ConvBranch(in_planes, out_planes, kernel_size=3, padding=1)
    layer.branch_1 = ConvBranch(in_planes, out_planes, kernel_size=5, padding=2)
    layer.branch_2 = PoolBranch(in_planes, out_planes, 'avg')
    layer.branch_3 = PoolBranch(in_planes, out_planes, 'max')

    return n_branches


def pick_func(layer, layer_type, x):
    if not (0 <= layer_type < n_branches):
        exit(1)
    return getattr(layer, "branch_{}".format(layer_type.cpu().item()))(x)


functions = (set_func, pick_func)
