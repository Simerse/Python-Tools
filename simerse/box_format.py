
from enum import Enum


class BoxFormat(Enum):
    min_max = 0
    min_extents = 1
    center_extents = 2
    center_half = 3


def make_polygon_min_max(box_data):
    return [
        box_data[0], box_data[1],
        box_data[0], box_data[3],
        box_data[2], box_data[3],
        box_data[2], box_data[1]
    ]


def make_polygon_min_extents(box_data):
    return [
        box_data[0], box_data[1],
        box_data[0], box_data[1] + box_data[3],
        box_data[0] + box_data[2], box_data[1] + box_data[3],
        box_data[0] + box_data[2], box_data[1]
    ]


def make_polygon_center_extents(box_data):
    return [
        box_data[0] - .5 * box_data[2], box_data[1] - .5 * box_data[3],
        box_data[0] - .5 * box_data[2], box_data[1] + .5 * box_data[3],
        box_data[0] + .5 * box_data[2], box_data[1] + .5 * box_data[3],
        box_data[0] + .5 * box_data[2], box_data[1] - .5 * box_data[3]
    ]


def make_polygon_center_half(box_data):
    return [
        box_data[0] - box_data[2], box_data[1] - box_data[3],
        box_data[0] - box_data[2], box_data[1] + box_data[3],
        box_data[0] + box_data[2], box_data[1] + box_data[3],
        box_data[0] + box_data[2], box_data[1] - box_data[3]
    ]


polygon_makers = {
    BoxFormat.min_max: make_polygon_min_max,
    BoxFormat.min_extents: make_polygon_min_extents,
    BoxFormat.center_extents: make_polygon_center_extents,
    BoxFormat.center_half: make_polygon_center_half,
}


def get_polygon_maker(box_format):
    return polygon_makers[box_format]


def box_hull_min_max(box_data_packed):
    return [
        min(box_data_packed[0::4]), min(box_data_packed[1::4]),
        max(box_data_packed[2::4]), max(box_data_packed[3::4])
    ]


def box_hull_min_extents(box_data_packed):
    max_x = max(box_data_packed[i + 2] + box_data_packed[i] for i in range(0, len(box_data_packed), 4))
    max_y = max(box_data_packed[i + 3] + box_data_packed[i + 1] for i in range(0, len(box_data_packed), 4))
    min_x, min_y = min(box_data_packed[0::4]), min(box_data_packed[1::4])
    return [
        min_x, min_y,
        max_x - min_x, max_y - min_y
    ]


def box_hull_center_extents(box_data_packed):
    min_x = min(box_data_packed[i] - .5 * box_data_packed[i + 2] for i in range(0, len(box_data_packed), 4))
    min_y = min(box_data_packed[i + 1] - .5 * box_data_packed[i + 3] for i in range(0, len(box_data_packed), 4))
    max_x = max(box_data_packed[i] + .5 * box_data_packed[i + 2] for i in range(0, len(box_data_packed), 4))
    max_y = max(box_data_packed[i + 1] + .5 * box_data_packed[i + 3] for i in range(0, len(box_data_packed), 4))
    return [
        (min_x + max_x) * .5, (min_y + max_y) * .5,
        (max_x - min_x), (max_y - min_y)
    ]


def box_hull_center_half(box_data_packed):
    min_x = min(box_data_packed[i] - box_data_packed[i + 2] for i in range(0, len(box_data_packed), 4))
    min_y = min(box_data_packed[i + 1] - box_data_packed[i + 3] for i in range(0, len(box_data_packed), 4))
    max_x = max(box_data_packed[i] + box_data_packed[i + 2] for i in range(0, len(box_data_packed), 4))
    max_y = max(box_data_packed[i + 1] + box_data_packed[i + 3] for i in range(0, len(box_data_packed), 4))
    return [
        (min_x + max_x) * .5, (min_y + max_y) * .5,
        (max_x - min_x) * .5, (max_y - min_y) * .5
    ]


box_hullers = {
    BoxFormat.min_max: box_hull_min_max,
    BoxFormat.min_extents: box_hull_min_extents,
    BoxFormat.center_extents: box_hull_center_extents,
    BoxFormat.center_half: box_hull_center_half
}


def get_box_huller(box_format):
    return box_hullers[box_format]


to_min_max_converters_2d = {
    BoxFormat.min_max: lambda box: box,
    BoxFormat.min_extents: lambda box: (box[0], box[1], box[0] + box[2], box[1] + box[3]),
    BoxFormat.center_extents: lambda box: (box[0] - .5 * box[2], box[1] - .5 * box[3],
                                           box[0] + .5 * box[2], box[1] + .5 * box[3]),
    BoxFormat.center_half: lambda box: (box[0] - box[2], box[1] - box[3], box[0] + box[2], box[1] + box[3])
}

from_min_max_converters_2d = {
    BoxFormat.min_max: lambda box: box,
    BoxFormat.min_extents: lambda box: (box[0], box[1], box[2] - box[0], box[3] - box[1]),
    BoxFormat.center_extents: lambda box: (.5 * (box[0] + box[2]), .5 * (box[1] + box[3]),
                                           box[2] - box[0], box[3] - box[1]),
    BoxFormat.center_half: lambda box: (.5 * (box[0] + box[2]), .5 * (box[1] + box[3]),
                                        .5 * (box[2] - box[0]), .5 * (box[3] - box[1]))
}

to_min_max_converters_3d = {
    BoxFormat.min_max: lambda box: box,
    BoxFormat.min_extents: lambda box: (box[0], box[1], box[2],
                                        box[0] + box[3], box[1] + box[4], box[2] + box[5]),
    BoxFormat.center_extents: lambda box: (box[0] - .5 * box[3], box[1] - .5 * box[4], box[2] - .5 * box[5],
                                           box[0] + .5 * box[3], box[1] + .5 * box[4], box[2] + .5 * box[5]),
    BoxFormat.center_half: lambda box: (box[0] - box[3], box[1] - box[4], box[2] - box[5],
                                        box[0] + box[3], box[1] + box[4], box[2] + box[5])
}

from_min_max_converters_3d = {
    BoxFormat.min_max: lambda box: box,
    BoxFormat.min_extents: lambda box: (box[0], box[1], box[2],
                                        box[3] - box[0], box[4] - box[1], box[5] - box[2]),
    BoxFormat.center_extents: lambda box: (.5 * (box[0] + box[3]), .5 * (box[1] + box[4]), .5 * (box[2] + box[5]),
                                           box[3] - box[0], box[4] - box[1], box[5] - box[2]),
    BoxFormat.center_half: lambda box: (.5 * (box[0] + box[3]), .5 * (box[1] + box[4]), .5 * (box[2] + box[5]),
                                        .5 * (box[3] - box[0]), .5 * (box[4] - box[1]), .5 * (box[5] - box[2]))
}


def convert_single_2d(box, src_format, dst_format):
    return from_min_max_converters_2d[dst_format](to_min_max_converters_2d[dst_format](box)) if src_format != dst_format \
        else box


def convert_single_3d(box, src_format, dst_format):
    return from_min_max_converters_3d[dst_format](to_min_max_converters_3d[dst_format](box)) if src_format != dst_format \
        else box
