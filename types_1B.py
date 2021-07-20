from .generate_convs import ConvSettings

from lib.model.conv_branch import ConvBranch
from lib.model.pool_branch import PoolBranch

settings = ConvSettings(0, 0, 3, 1, 1, 1).generate_type_eq_settings(8, 8)
n_branches = 6 + len(settings)


def set_func(layer, in_planes, out_planes):

    layer.branch_0 = ConvBranch(in_planes,
                                out_planes,
                                kernel_size=3,
                                padding=1)
    layer.branch_1 = ConvBranch(in_planes,
                                out_planes,
                                kernel_size=3,
                                padding=1,
                                separable=True)
    layer.branch_2 = ConvBranch(in_planes,
                                out_planes,
                                kernel_size=5,
                                padding=2)
    layer.branch_3 = ConvBranch(in_planes,
                                out_planes,
                                kernel_size=5,
                                padding=2,
                                separable=True)
    layer.branch_4 = PoolBranch(in_planes, out_planes, 'avg')
    layer.branch_5 = PoolBranch(in_planes, out_planes, 'max')

    for i in range(len(settings)):
        setattr(
            layer, "branch_{}".format(6 + i),
            ConvBranch(in_planes, out_planes, settings[i].kernel_size,
                       settings[i].padding, settings[i].dilation,
                       settings[i].stride))

    return n_branches


def pick_func(layer, layer_type, x):
    if not (0 <= layer_type < n_branches):
        exit(1)
    return getattr(layer, "branch_{}".format(layer_type.cpu().item()))(x)
