
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
