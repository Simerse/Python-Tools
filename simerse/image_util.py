
import numpy as np

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None
    ImageDraw = None

try:
    import torch
except ImportError:
    torch = None


def to_float(im):
    if not hasattr(im, "astype"):
        im = np.array(im)
    if not np.issubdtype(im.dtype, np.floating):
        return im.astype(np.float32) / 255
    return im


def to_int(im):
    if not hasattr(im, "astype"):
        im = np.array(im)
    if np.issubdtype(im.dtype, np.floating):
        return (im * 255).astype(np.uint8)
    elif np.issubdtype(im.dtype, np.integer):
        return im
    else:
        return im.astype(np.uint8)


if torch is not None:
    # noinspection PyUnresolvedReferences, PyTypeChecker
    def to_torch(im, device=None):
        if Image is not None and isinstance(im, Image.Image):
            return to_torch(to_float(im), device)
        elif isinstance(im, (list, tuple, set)):
            return torch.cat([to_torch(i, device) for i in im])
        elif isinstance(im, np.ndarray):
            a = to_float(im)
            if len(a.shape) == 2:
                a = np.stack([np.stack([a])])
            elif len(a.shape) == 3:
                a = np.stack([a]).transpose((0, 3, 1, 2))
            else:
                a = a.transpose(0, 3, 1, 2)
            return torch.tensor(a) if device is None else torch.tensor(a, device=device)
        elif isinstance(im, torch.Tensor):
            if device is None or im.device == device:
                return im
            else:
                return im.to(device)
else:
    # noinspection PyUnusedLocal
    def to_torch(im, device=None):
        raise ImportError("Could not import PyTorch.")


# noinspection PyUnresolvedReferences, PyTypeChecker
def to_numpy(im):
    if torch is not None and isinstance(im, torch.Tensor):
        if len(im.shape) == 4:
            return im.detach().cpu().numpy().transpose(0, 2, 3, 1).squeeze()
        elif len(im.shape) == 3:
            return im.detach().cpu().numpy().transpose(1, 2, 0).squeeze()
        elif len(im.shape) == 2:
            return im.detach().cpu().numpy()
        else:
            raise ValueError('Invalid tensor provided to to_numpy.')
    elif isinstance(im, Image.Image):
        a = np.array(im)
        return a
    elif isinstance(im, (list, tuple, set)):
        return np.stack([to_numpy(i) for i in im])
    else:
        return to_float(im)


if Image is not None:
    # noinspection PyUnresolvedReferences
    def to_pil(im):
        if torch is not None and isinstance(im, torch.Tensor):
            return to_pil(to_numpy(im))
        elif isinstance(im, np.ndarray):
            if len(im.shape) == 4:
                return to_pil(list(im))
            else:
                return Image.fromarray(to_int(im))
        elif isinstance(im, (list, tuple, set)):
            return [to_pil(i) for i in im]
        else:
            return to_pil(to_float(im))
else:
    # noinspection PyUnusedLocal
    def to_pil(im):
        raise ImportError("Could not import PIL.")


def insert_dim(shape, i, value=1):
    return tuple(shape[:i]) + (value,) + tuple(shape[i:])


def remove_dim(shape, i):
    return tuple(shape[:i]) + tuple(shape[i + 1:])


def decompress_rle(rle, uids, shape):
    """
    Combines and decompresses a set of binary RLE images.
    :param rle: Iterable of binary RLE images.
    :param uids: Parallel iterable with rle containing the values to substitute for 1 in the binary RLE images.
    :param shape: The output image array shape, i.e., a tuple (height, width).
    :return: decompressed numpy image
    """
    all_rle = list(map(lambda x: list(map(int, x)), rle))
    all_uid = list(map(int, uids))

    if len(all_rle) != len(all_uid):
        raise ValueError('Inconsistent number of uids and rle images provided')

    start_inside = [all_rle[i][0] == all_uid[i] for i in range(len(all_rle))]
    for i in range(len(all_rle)):
        all_rle[i][0] = 0

    switch_indices = [np.stack([np.cumsum(r), np.repeat(uid, len(r))], axis=1) for r, uid in zip(all_rle, all_uid)]
    enter_indices = [switches[0::2] if start_inside[i] else switches[1::2] for i, switches in enumerate(switch_indices)]
    enter_indices = np.concatenate(enter_indices, axis=0)
    enter_order = np.argsort(enter_indices[:, 0])
    enter_indices = enter_indices[enter_order]

    out_image_flat = np.zeros(shape[0] * shape[1], dtype=np.int32)

    for i in range(len(enter_indices) - 1):
        out_image_flat[enter_indices[i][0]:enter_indices[i + 1][0]] = enter_indices[i][1]

    return out_image_flat.reshape(shape)


def draw_polygons(polygon_lists, uids, shape, fill=True, line_thickness=1, joint=None):
    if Image is None:
        raise ValueError("Please install PIL to use polygon drawing functions.")

    im = Image.fromarray(np.zeros(shape, dtype=np.int32))
    draw = ImageDraw.Draw(im)
    if fill:
        for polygon_list, uid in zip(polygon_lists, uids):
            for polygon in polygon_list:
                polygon = tuple(map(lambda x: int(round(float(x))), polygon))
                draw.polygon(polygon, fill=int(uid))
    else:
        for polygon_list, uid in zip(polygon_lists, uids):
            for polygon in polygon_list:
                polygon = list(polygon)
                polygon.extend(polygon[0:2])
                polygon = tuple((int(round(float(polygon[i]))), int(round(float(polygon[i + 1]))))
                                for i in range(0, len(polygon), 2))
                draw.line(polygon, fill=int(uid), width=line_thickness, joint=joint)
    return np.array(im)


def draw_texts_and_polygons(
        im, texts, text_positions, polygon_lists, colors, font, line_thickness=1, joint=None,
        show_text=True
):
    if Image is None:
        raise ValueError("Please install PIL to use polygon drawing functions.")

    im = Image.fromarray(im)
    draw = ImageDraw.Draw(im)
    if show_text:
        for text, text_position_list, color, polygon_list in zip(texts, text_positions, colors, polygon_lists):
            for polygon, text_position in zip(polygon_list, text_position_list):
                polygon = list(polygon)
                polygon.extend(polygon[0:2])
                polygon = tuple((int(polygon[i]), int(polygon[i + 1])) for i in range(0, len(polygon), 2))
                draw.line(polygon, fill=tuple(color), width=line_thickness, joint=joint)
                draw.text(tuple(map(int, text_position)), text, fill=tuple(color), font=font)
    else:
        for polygon_list, color in zip(polygon_lists, colors):
            for polygon in polygon_list:
                polygon = list(polygon)
                polygon.extend(polygon[0:2])
                polygon = tuple((int(polygon[i]), int(polygon[i + 1])) for i in range(0, len(polygon), 2))
                draw.line(polygon, fill=tuple(color), width=line_thickness, joint=joint)

    return np.array(im)


def draw_points(points, im, color, size):
    if Image is None:
        raise ValueError("Please install PIL to use polygon drawing functions.")

    im = Image.fromarray(im)
    draw = ImageDraw.Draw(im)

    for point in points:
        draw.ellipse((
            float(point[0]) - size * .5, float(point[1]) - size * .5,
            float(point[0]) + size * .5, float(point[1]) + size * .5
        ), fill=color)

    return np.array(im)
