from pointnet2_ops import pointnet2_utils


def fps(data, number):
    indices = pointnet2_utils.furthest_point_sample(data, number)
    return (
        pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), indices)
        .transpose(1, 2)
        .contiguous()
    )


def fps_gs(data, number, attribute=("xyz",), return_idx=False):
    channels = []
    for name, indices in (
        ("xyz", (0, 1, 2)),
        ("opacity", (3,)),
        ("scale", (4, 5, 6)),
        ("rotation", (7, 8, 9, 10)),
        ("sh", (11, 12, 13)),
    ):
        if name in attribute:
            channels.extend(indices)
    indices = pointnet2_utils.furthest_point_sample(data[..., channels].contiguous(), number)
    if return_idx:
        return indices
    return (
        pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), indices)
        .transpose(1, 2)
        .contiguous()
    )
