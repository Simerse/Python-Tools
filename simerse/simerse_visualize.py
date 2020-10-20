from simerse import image_util
import numpy as np
import matplotlib.pyplot as plt
import imageio

try:
    from PIL import ImageFont
except ImportError:
    ImageFont = None

from simerse.simerse_keys import BuiltinDimension, Visualize
from simerse.box_format import get_polygon_maker, get_box_huller


def ecdf(x):
    x = np.sort(x)
    n = len(x)

    def _ecdf(v):
        return np.searchsorted(x, v, side='right') / n

    return _ecdf


class VisualLDRVisualizer:
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def visualize(self, observation_uid, save_on_finish=None):
        if BuiltinDimension.visual_ldr in self.data_loader.dimensions:
            im = image_util.to_numpy(self.data_loader.load(observation_uid, BuiltinDimension.visual_ldr))
            plt.imshow(im)
            plt.show()
            if save_on_finish:
                imageio.imwrite(save_on_finish, im)
        else:
            raise ValueError(f'Dataset {self.data_loader.name} does not contain visual ldr in any form. '
                             f'Dimensions are {self.data_loader.dimensions}')


class VisualHDRVisualizer:
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def visualize(self, observation_uid, mapping=None, save_on_finish=None):
        if BuiltinDimension.visual_hdr in self.data_loader.dimensions:
            im = image_util.to_numpy(self.data_loader.load(observation_uid, BuiltinDimension.visual_hdr))
            if mapping is not None:
                im = mapping(im)
            plt.imshow(im)
            plt.show()
            if save_on_finish is not None:
                imageio.imwrite(save_on_finish, im)
        else:
            raise ValueError(f'Dataset {self.data_loader.name} does not contain visual hdr in any form. '
                             f'Dimensions are {self.data_loader.dimensions}')


def make_object_uid_filter(object_name_filter, object_uid_filter, uid_to_name):
    if object_name_filter is None and object_uid_filter is None:
        def object_filter(__):
            return True
    elif object_name_filter is None:
        def object_filter(x):
            return object_uid_filter(x)
    elif object_uid_filter is None:
        def object_filter(x):
            return object_name_filter(uid_to_name[int(x)])
    else:
        def object_filter(x):
            object_uid_filter(x) and object_name_filter(uid_to_name[int(x)])

    return object_filter


class SegmentationVisualizer:
    def __init__(self, data_loader):
        self.data_loader = data_loader

    @property
    def color_mapping(self):
        if 'visualize_cached_color_mapping' not in type(self.data_loader).cache:
            max_uid = max(self.data_loader.object_uid_name_mapping.uid_to_name.keys())
            type(self.data_loader).cache['visualize_cached_color_mapping'] = \
                np.random.random_integers(50, 256, (max_uid + 1, 3)).astype(np.uint8)
            type(self.data_loader).cache['visualize_cached_color_mapping'][0] = (0, 0, 0)
        return type(self.data_loader).cache['visualize_cached_color_mapping']

    def visualize(self, observation_uid,
                  mode='overlay',
                  object_name_filter=None, object_uid_filter=None,
                  overlay_alpha=.7,
                  outline=False, line_thickness=4, joint=None,
                  save_on_finish=None):
        object_filter = make_object_uid_filter(object_name_filter, object_uid_filter,
                                               self.data_loader.object_uid_name_mapping.uid_to_name)
        color_mapping = self.color_mapping

        if mode == 'overlay' and BuiltinDimension.visual_ldr not in self.data_loader.dimensions:
            raise ValueError(f'Cannot use \'overlay\' mode for segmentation visualization for dataset '
                             f'{self.data_loader.name} because the dataset does not have a Visual LDR dimension.'
                             f' Dimensions are {self.data_loader.dimensions}')

        if not outline:
            if BuiltinDimension.segmentation in self.data_loader.dimensions:
                capture = image_util.to_numpy(self.data_loader.load(observation_uid, BuiltinDimension.segmentation))
                if object_name_filter is not None or object_uid_filter is not None:
                    uids = list(filter(object_filter, self.data_loader.object_uid_name_mapping.uid_to_name.keys()))
                    color_mapping = np.zeros_like(color_mapping)
                    color_mapping[uids] = self.color_mapping[uids]
            elif BuiltinDimension.segmentation_rle in self.data_loader.dimensions:
                rle_uid = zip(*self.data_loader.load(observation_uid, (
                    BuiltinDimension.segmentation_rle, BuiltinDimension.object_uid
                )))
                final_rle, final_uid = tuple(zip(*filter(
                    lambda x: x[0] != self.data_loader.na_value and object_filter(x[1]), rle_uid
                )))
                capture = image_util.decompress_rle(
                    final_rle, final_uid,
                    (self.data_loader.capture_resolution.height, self.data_loader.capture_resolution.width)
                )
            elif BuiltinDimension.segmentation_polygon in self.data_loader.dimensions:
                ply_uid = zip(*self.data_loader.load(observation_uid, (
                    BuiltinDimension.segmentation_polygon, BuiltinDimension.object_uid
                )))
                final_polygon, final_uid = tuple(zip(*filter(
                    lambda x: x[0] != self.data_loader.na_value and object_filter(x[1]), ply_uid
                )))
                capture = image_util.draw_polygons(
                    final_polygon, final_uid,
                    (self.data_loader.capture_resolution.height, self.data_loader.capture_resolution.width)
                )
            else:
                raise TypeError(f'Dataset {self.data_loader.name} does not contain segmentation in any form.'
                                f' Dimensions are {self.data_loader.dimensions}')

        else:
            if BuiltinDimension.segmentation_outline in self.data_loader.dimensions:
                capture = self.data_loader.load(observation_uid, BuiltinDimension.segmentation_outline)
                if object_name_filter is not None or object_uid_filter is not None:
                    uids = list(filter(object_filter, self.data_loader.object_uid_name_mapping.uid_to_name.keys()))
                    color_mapping = np.zeros_like(color_mapping)
                    color_mapping[uids] = self.color_mapping[uids]
            elif BuiltinDimension.segmentation_polygon in self.data_loader.dimensions:
                ply_uid = zip(*self.data_loader.load(observation_uid, (
                    BuiltinDimension.segmentation_polygon, BuiltinDimension.object_uid
                )))
                final_polygon, final_uid = tuple(zip(*filter(
                    lambda x: x[0] != self.data_loader.na_value and object_filter(x[1]), ply_uid
                )))
                print(final_uid)
                capture = image_util.draw_polygons(
                    final_polygon, final_uid,
                    (self.data_loader.capture_resolution.height, self.data_loader.capture_resolution.width), fill=False,
                    line_thickness=line_thickness, joint=joint
                )
            else:
                raise TypeError(f'Dataset {self.data_loader.name} does not contain segmentation outline in any form.'
                                f' Dimensions are {self.data_loader.dimensions}')

        if mode == 'overlay':
            visual = image_util.to_int(
                image_util.to_numpy(self.data_loader.load(observation_uid, BuiltinDimension.visual_ldr))[:, :, :3]
            )
            overlay = color_mapping[capture]
            zero_mask = (overlay == np.zeros((1, 1, 3), dtype=np.uint8)).astype(np.uint8)
            overlay_im = visual * zero_mask + (1 - zero_mask) * (
                    visual.astype(np.float32) * (1 - overlay_alpha) + overlay.astype(np.float32) * overlay_alpha
            ).astype(np.uint8)
            plt.imshow(overlay_im)
            plt.show()
            if save_on_finish is not None:
                imageio.imwrite(save_on_finish, overlay_im)
        elif mode == 'raw':
            final_image = color_mapping[capture]
            plt.imshow(final_image)
            plt.show()
            if save_on_finish is not None:
                imageio.imwrite(save_on_finish, final_image)
        else:
            raise ValueError(f'Given segmentation visualization mode {mode} is not supported;'
                             f' choose one of \'overlay\' or \'raw\'.')


class BoundingBox2DVisualizer:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.polygon_maker = get_polygon_maker(self.data_loader.box_format2d)

    @staticmethod
    def get_text_locations(polygons, text_size, line_thickness):
        location_lists = []
        for polygon_list in polygons:
            locations = []
            for p in polygon_list:
                locations.append((p[0], p[1] - text_size - line_thickness - 2))
            location_lists.append(locations)
        return location_lists

    def get_total_boxes(self, contiguous_lists):
        box_huller = get_box_huller(self.data_loader.box_format2d)
        return [box_huller(boxes) for boxes in contiguous_lists]

    def make_polygons(self, box_data):
        polygon_lists = []
        origin = self.data_loader.image_coordinate_origin
        for box_list in box_data:
            polygons = []
            for i in range(len(box_list) // 4):
                polygon = list(self.polygon_maker(box_list[4 * i: 4 * (i + 1)]))
                for p in range(0, 4, 2):
                    polygon[p] -= origin.x
                for p in range(1, 4, 2):
                    polygon[p] -= origin.y
                polygons.append(polygon)
            polygon_lists.append(polygons)
        return polygon_lists

    @property
    def color_mapping(self):
        if 'visualize_cached_color_mapping' not in type(self.data_loader).cache:
            max_uid = max(self.data_loader.object_uid_name_mapping.uid_to_name.keys())
            type(self.data_loader).cache['visualize_cached_color_mapping'] = \
                np.random.random_integers(50, 256, (max_uid + 1, 3)).astype(np.uint8)
            type(self.data_loader).cache['visualize_cached_color_mapping'][0] = (0, 0, 0)
        return type(self.data_loader).cache['visualize_cached_color_mapping']

    def visualize(self, observation_uid,
                  object_name_filter=None, object_uid_filter=None,
                  kind='total',
                  line_thickness=1, joint=None,
                  show_names=True, name_font_size=12,
                  save_on_finish=None):
        if BuiltinDimension.visual_ldr not in self.data_loader.dimensions:
            raise TypeError(f'Cannot visualize 2d bounding boxes for dataset {self.data_loader.name} because'
                            f' the dataset does not have a Visual LDR dimension. Dimensions are '
                            f'{self.data_loader.dimensions}')

        object_filter = make_object_uid_filter(object_name_filter, object_uid_filter,
                                               self.data_loader.object_uid_name_mapping.uid_to_name)

        font = None
        if show_names:
            if ImageFont is None:
                raise ValueError('Please install PIL to use image drawing functions.')
            font = ImageFont.truetype('arial.ttf', size=name_font_size)

        im = image_util.to_int(
            image_util.to_numpy(self.data_loader.load(observation_uid, BuiltinDimension.visual_ldr))[:, :, :3]
        )

        if kind == 'total':
            if BuiltinDimension.bounding_box_2d_total in self.data_loader.dimensions:
                boxes_uids = zip(*self.data_loader.load(observation_uid, (
                    BuiltinDimension.bounding_box_2d_total, BuiltinDimension.object_uid
                )))
                boxes, uids = tuple(zip(*filter(lambda x: x[0] != self.data_loader.na_value and object_filter(x[1]),
                                                boxes_uids)))
                polygons = self.make_polygons(boxes)
                colors = self.color_mapping[list(uids)]
                if show_names:
                    texts = (self.data_loader.object_uid_name_mapping.uid_to_name[int(uid)] for uid in uids)
                    text_locations = BoundingBox2DVisualizer.get_text_locations(polygons, name_font_size,
                                                                                line_thickness)
                    vis = image_util.draw_texts_and_polygons(
                        im, texts, text_locations, polygons, colors,
                        font=font, line_thickness=line_thickness, joint=joint
                    )
                else:
                    vis = image_util.draw_texts_and_polygons(
                        im, None, None, polygons, colors, font=None, line_thickness=line_thickness, joint=joint,
                        show_text=False
                    )
            elif BuiltinDimension.bounding_box_2d_contiguous in self.data_loader.dimensions:
                boxes_uids = zip(*self.data_loader.load(observation_uid, (
                    BuiltinDimension.bounding_box_2d_contiguous, BuiltinDimension.object_uid
                )))
                boxes, uids = tuple(zip(*filter(lambda x: x[0] != self.data_loader.na_value and object_filter(x[1]),
                                                boxes_uids)))
                boxes = self.get_total_boxes(boxes)
                polygons = self.make_polygons(boxes)
                colors = self.color_mapping[list(uids)]
                if show_names:
                    texts = (self.data_loader.object_uid_name_mapping.uid_to_name[int(uid)] for uid in uids)
                    text_locations = BoundingBox2DVisualizer.get_text_locations(polygons, name_font_size,
                                                                                line_thickness)
                    vis = image_util.draw_texts_and_polygons(
                        im, texts, text_locations, polygons, colors,
                        font=font, line_thickness=line_thickness, joint=joint
                    )
                else:
                    vis = image_util.draw_texts_and_polygons(
                        im, None, None, polygons, colors, font=None, line_thickness=line_thickness, joint=joint,
                        show_text=False
                    )
            else:
                raise TypeError(f'Dataset {self.data_loader.name} does not contain 2d total bounding '
                                f'boxes in any form. Dimensions are {self.data_loader.dimensions}')
        elif kind == 'contiguous':
            if BuiltinDimension.bounding_box_2d_contiguous in self.data_loader.dimensions:
                boxes_uids = zip(*self.data_loader.load(observation_uid, (
                    BuiltinDimension.bounding_box_2d_contiguous, BuiltinDimension.object_uid
                )))
                boxes, uids = tuple(zip(*filter(lambda x: x[0] != self.data_loader.na_value and object_filter(x[1]),
                                                boxes_uids)))
                polygons = self.make_polygons(boxes)
                colors = self.color_mapping[list(uids)]
                if show_names:
                    texts = (self.data_loader.object_uid_name_mapping.uid_to_name[int(uid)] for uid in uids)
                    text_locations = BoundingBox2DVisualizer.get_text_locations(polygons, name_font_size,
                                                                                line_thickness)
                    vis = image_util.draw_texts_and_polygons(
                        im, texts, text_locations, polygons, colors,
                        font=font, line_thickness=line_thickness, joint=joint
                    )
                else:
                    vis = image_util.draw_texts_and_polygons(
                        im, None, None, polygons, colors, font=None, line_thickness=line_thickness, joint=joint,
                        show_text=False
                    )
            else:
                raise TypeError(f'Dataset {self.data_loader.name} does not contain 2d contiguous bounding '
                                f'boxes in any form. Dimensions are {self.data_loader.dimensions}')
        else:
            raise ValueError(f'Given bounding box kind {kind} is not a valid kind of bounding box. Choose one of'
                             f' \'total\' or \'contiguous\'.')

        plt.imshow(vis)
        plt.show()
        if save_on_finish is not None:
            imageio.imwrite(save_on_finish, vis)


class DepthVisualizer:
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def visualize(self, observation_uid, mapping=None, save_on_finish=None, background_cutoff=50000):
        if BuiltinDimension.depth in self.data_loader.dimensions:
            depth = image_util.to_numpy(self.data_loader.load(observation_uid, BuiltinDimension.depth))
            depth = np.minimum(depth, background_cutoff)
            if mapping == 'ecdf':
                dist = ecdf(depth.flatten())
                depth = np.stack([(dist(depth) * 255).astype(np.uint8)] * 3, axis=2)
            elif mapping is not None:
                depth = np.stack([(mapping(depth) * 255).astype(np.uint8)] * 3, axis=2)
            plt.imshow(depth)
            plt.show()
            if save_on_finish is not None:
                if mapping is None:
                    raise ValueError('Must supply a depth mapping in order to save a depth visualization.')
                imageio.imwrite(save_on_finish, depth)
        else:
            raise ValueError(f'Dataset {self.data_loader.name} does not contain depth in any form. '
                             f'Dimensions are {self.data_loader.dimensions}')


class NormalVisualizer:
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def visualize(self, observation_uid, separate_channels=False, save_on_finish=None):
        normal = None
        if BuiltinDimension.world_normal in self.data_loader.dimensions:
            normal = image_util.to_numpy(
                self.data_loader.load(observation_uid, BuiltinDimension.world_normal)
            )[:, :, :3]
        elif BuiltinDimension.world_tangent in self.data_loader.dimensions and \
                BuiltinDimension.world_bitangent in self.data_loader.dimensions:
            tangent = image_util.to_numpy(
                self.data_loader.load(observation_uid, BuiltinDimension.world_tangent)
            )[:, :, :3]
            bitangent = image_util.to_numpy(
                self.data_loader.load(observation_uid, BuiltinDimension.world_bitangent)
            )[:, :, :3]
            normal = np.cross(tangent, bitangent, axis=2)
        if normal is not None:
            if separate_channels:
                fig = plt.figure(figsize=(8, 8))
                fig.add_subplot(3, 1, 1)
                plt.imshow(normal[:, :, 0])
                fig.add_subplot(3, 1, 2)
                plt.imshow(normal[:, :, 1])
                fig.add_subplot(3, 1, 3)
                plt.imshow(normal[:, :, 2])
                plt.show()
                if save_on_finish is not None:
                    import os
                    name, ext = os.path.splitext(save_on_finish)
                    normal = normal * .5 + .5
                    imageio.imwrite(name + '_x' + ext, (normal[:, :, 0] * 255).astype(np.uint8))
                    imageio.imwrite(name + '_y' + ext, (normal[:, :, 1] * 255).astype(np.uint8))
                    imageio.imwrite(name + '_z' + ext, (normal[:, :, 2] * 255).astype(np.uint8))
            else:
                plt.imshow(normal * .5 + .5)
                plt.show()
                if save_on_finish is not None:
                    imageio.imwrite(save_on_finish, image_util.to_int(normal * .5 + .5))
        else:
            raise ValueError(f'Dataset {self.data_loader.name} does not contain world normal in any form. '
                             f'Dimensions are {self.data_loader.dimensions}')


class TangentVisualizer:
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def visualize(self, observation_uid, separate_channels=False, save_on_finish=None):
        tangent = None
        if BuiltinDimension.world_tangent in self.data_loader.dimensions:
            tangent = image_util.to_numpy(
                self.data_loader.load(observation_uid, BuiltinDimension.world_tangent)
            )[:, :, :3]
        elif BuiltinDimension.world_normal in self.data_loader.dimensions and \
                BuiltinDimension.world_bitangent in self.data_loader.dimensions:
            normal = image_util.to_numpy(
                self.data_loader.load(observation_uid, BuiltinDimension.world_normal)
            )[:, :, :3]
            bitangent = image_util.to_numpy(
                self.data_loader.load(observation_uid, BuiltinDimension.world_bitangent)
            )[:, :, :3]
            tangent = np.cross(bitangent, normal, axis=2)
        if tangent is not None:
            if separate_channels:
                fig = plt.figure(figsize=(8, 8))
                fig.add_subplot(3, 1, 1)
                plt.imshow(tangent[:, :, 0])
                fig.add_subplot(3, 1, 2)
                plt.imshow(tangent[:, :, 1])
                fig.add_subplot(3, 1, 3)
                plt.imshow(tangent[:, :, 2])
                plt.show()
                if save_on_finish is not None:
                    import os
                    tangent = tangent * .5 + .5
                    name, ext = os.path.splitext(save_on_finish)
                    imageio.imwrite(name + '_x' + ext, (tangent[:, :, 0] * 255).astype(np.uint8))
                    imageio.imwrite(name + '_y' + ext, (tangent[:, :, 1] * 255).astype(np.uint8))
                    imageio.imwrite(name + '_z' + ext, (tangent[:, :, 2] * 255).astype(np.uint8))
            else:
                plt.imshow(tangent * .5 + .5)
                plt.show()
                if save_on_finish is not None:
                    imageio.imwrite(save_on_finish, image_util.to_int(tangent * .5 + .5))
        else:
            raise ValueError(f'Dataset {self.data_loader.name} does not contain world tangent in any form. '
                             f'Dimensions are {self.data_loader.dimensions}')


class BitangentVisualizer:
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def visualize(self, observation_uid, separate_channels=False, save_on_finish=None):
        bitangent = None
        if BuiltinDimension.world_bitangent in self.data_loader.dimensions:
            bitangent = image_util.to_numpy(
                self.data_loader.load(observation_uid, BuiltinDimension.world_bitangent)
            )[:, :, :3]
        elif BuiltinDimension.world_normal in self.data_loader.dimensions and \
                BuiltinDimension.world_bitangent in self.data_loader.dimensions:
            normal = image_util.to_numpy(
                self.data_loader.load(observation_uid, BuiltinDimension.world_normal)
            )[:, :, :3]
            tangent = image_util.to_numpy(
                self.data_loader.load(observation_uid, BuiltinDimension.world_tangent)
            )[:, :, :3]
            bitangent = np.cross(normal, tangent, axis=2)
        if bitangent is not None:
            if separate_channels:
                fig = plt.figure(figsize=(8, 8))
                fig.add_subplot(3, 1, 1)
                plt.imshow(bitangent[:, :, 0])
                fig.add_subplot(3, 1, 2)
                plt.imshow(bitangent[:, :, 1])
                fig.add_subplot(3, 1, 3)
                plt.imshow(bitangent[:, :, 2])
                plt.show()
                if save_on_finish is not None:
                    import os
                    bitangent = bitangent * .5 + .5
                    name, ext = os.path.splitext(save_on_finish)
                    imageio.imwrite(name + '_x' + ext, (bitangent[:, :, 0] * 255).astype(np.uint8))
                    imageio.imwrite(name + '_y' + ext, (bitangent[:, :, 1] * 255).astype(np.uint8))
                    imageio.imwrite(name + '_z' + ext, (bitangent[:, :, 2] * 255).astype(np.uint8))
            else:
                plt.imshow(bitangent * .5 + .5)
                plt.show()
                if save_on_finish is not None:
                    imageio.imwrite(save_on_finish, image_util.to_int(bitangent * .5 + .5))
        else:
            raise ValueError(f'Dataset {self.data_loader.name} does not contain world bitangent in any form. '
                             f'Dimensions are {self.data_loader.dimensions}')


class UVVisualizer:
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def visualize(self, observation_uid, save_on_finish=None):
        if BuiltinDimension.uv in self.data_loader.dimensions:
            uv = image_util.to_int(image_util.to_numpy(self.data_loader.load(observation_uid, BuiltinDimension.uv)))
            plt.imshow(uv)
            plt.show()
            if save_on_finish is not None:
                imageio.imwrite(save_on_finish, uv)
        else:
            raise ValueError(f'Dataset {self.data_loader.name} does not contain uv in any form. '
                             f'Dimensions are {self.data_loader.dimensions}')


class KeypointsVisualizer:
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def visualize(self, observation_uid, color=(0, 255, 0, 255), point_size=4,
                  object_name_filter=None, object_uid_filter=None,
                  save_on_finish=None):
        object_filter = make_object_uid_filter(object_name_filter, object_uid_filter,
                                               self.data_loader.object_uid_name_mapping.uid_to_name)

        if BuiltinDimension.keypoints in self.data_loader.dimensions:
            uids, keypoints = self.data_loader.load(observation_uid,
                                                    (BuiltinDimension.object_uid, BuiltinDimension.keypoints))
            uids, keypoints = tuple(zip(*filter(lambda x: object_filter(x[0]), zip(uids, keypoints))))
            all_keypoints = []
            for keypoint_list in keypoints:
                all_keypoints.extend(keypoint_list)
            keypoints = [(all_keypoints[i], all_keypoints[i + 1]) for i in range(0, len(all_keypoints), 2)
                         if all_keypoints[i] != self.data_loader.na_value]
            im = image_util.to_int(
                image_util.to_numpy(self.data_loader.load(observation_uid, BuiltinDimension.visual_ldr))
            )
            vis = image_util.draw_points(keypoints, im, color, point_size)
            plt.imshow(vis)
            plt.show()
            if save_on_finish is not None:
                imageio.imwrite(save_on_finish, vis)
        else:
            raise ValueError(f'Dataset {self.data_loader.name} does not contain keypoints in any form. '
                             f'Dimensions are {self.data_loader.dimensions}')


class PositionVisualizer:
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def visualize(self, observation_uid, save_on_finish=None):
        if BuiltinDimension.world_position in self.data_loader.dimensions:
            position = image_util.to_numpy(
                self.data_loader.load(observation_uid, BuiltinDimension.world_position)
            )[:, :, :3]

            fig = plt.figure(figsize=(8, 8))
            fig.add_subplot(3, 1, 1)
            plt.imshow(position[:, :, 0])
            fig.add_subplot(3, 1, 2)
            plt.imshow(position[:, :, 1])
            fig.add_subplot(3, 1, 3)
            plt.imshow(position[:, :, 2])
            plt.show()

            if save_on_finish is not None:
                import os
                name, ext = os.path.splitext(save_on_finish)
                min_values = position.min(axis=0).min(axis=0).reshape((1, 1, 3))
                max_values = position.max(axis=0).max(axis=0).reshape((1, 1, 3))
                position = (position - min_values) / (max_values - min_values)
                imageio.imwrite(name + '_x' + ext, (position[:, :, 0] * 255).astype(np.uint8))
                imageio.imwrite(name + '_y' + ext, (position[:, :, 1] * 255).astype(np.uint8))
                imageio.imwrite(name + '_z' + ext, (position[:, :, 2] * 255).astype(np.uint8))
        else:
            raise ValueError(f'Dataset {self.data_loader.name} does not contain world position in any form. '
                             f'Dimensions are {self.data_loader.dimensions}')


visualizers = {
    Visualize.visual_ldr: VisualLDRVisualizer,
    Visualize.visual_hdr: VisualHDRVisualizer,
    Visualize.bounding_box_2d: BoundingBox2DVisualizer,
    Visualize.depth: DepthVisualizer,
    Visualize.segmentation: SegmentationVisualizer,
    Visualize.normal: NormalVisualizer,
    Visualize.tangent: TangentVisualizer,
    Visualize.bitangent: BitangentVisualizer,
    Visualize.uv: UVVisualizer,
    Visualize.keypoints: KeypointsVisualizer,
    Visualize.position: PositionVisualizer,
}

visualizer_dimensions = {
    Visualize.visual_ldr: {
        BuiltinDimension.visual_ldr
    },
    Visualize.visual_hdr: {
        BuiltinDimension.visual_hdr
    },
    Visualize.bounding_box_2d: {
        BuiltinDimension.bounding_box_2d_contiguous, BuiltinDimension.bounding_box_2d_total
    },
    Visualize.bounding_box_3d: {
        BuiltinDimension.bounding_box_3d_custom, BuiltinDimension.bounding_box_3d_global,
        BuiltinDimension.bounding_box_3d_local
    },
    Visualize.depth: {
        BuiltinDimension.depth
    },
    Visualize.segmentation: {
        BuiltinDimension.segmentation, BuiltinDimension.segmentation_rle, BuiltinDimension.segmentation_outline,
        BuiltinDimension.segmentation_polygon
    },
    Visualize.normal: {
        BuiltinDimension.world_normal, (BuiltinDimension.world_tangent, BuiltinDimension.world_bitangent)
    },
    Visualize.tangent: {
        BuiltinDimension.world_tangent, (BuiltinDimension.world_normal, BuiltinDimension.world_bitangent)
    },
    Visualize.bitangent: {
        BuiltinDimension.world_bitangent, (BuiltinDimension.world_normal, BuiltinDimension.world_tangent)
    },
    Visualize.uv: {
        BuiltinDimension.uv
    },
    Visualize.keypoints: {
        BuiltinDimension.keypoints
    },
    Visualize.position: {
        BuiltinDimension.world_position
    }
}


def get_visualizers(data_loader):
    ret = {}
    for vis, req in visualizer_dimensions.items():
        single_req = {x for x in req if not isinstance(x, tuple)}
        multi_req = [set(x) for x in req if isinstance(x, tuple)]
        if len(single_req.intersection(data_loader.dimensions)) > 0 or \
                any(multi.intersection(data_loader.dimensions) == len(multi) for multi in multi_req):
            ret[vis] = visualizers[vis](data_loader)
    return ret


class SimerseVisualizer:
    def __init__(self, data_loader):
        global visualizers
        self.data_loader = data_loader
        self.visualizers = get_visualizers(data_loader)

    def visualize(self, observation_uid, dimension, **kwargs):
        if dimension not in visualizers:
            raise ValueError(f'Given dimension {dimension} is not yet supported for visualization.')

        if dimension not in self.visualizers:
            raise ValueError(f'Dataset {self.data_loader.name} does not support visualization of {dimension}.'
                             f' Must be one of {list(map(str, self.visualizers.keys()))}')

        self.visualizers[dimension].visualize(observation_uid, **kwargs)

    @property
    def color_mapping(self):
        if 'visualize_cached_color_mapping' not in type(self.data_loader).cache:
            max_uid = max(self.data_loader.object_uid_name_mapping.uid_to_name.keys())
            type(self.data_loader).cache['visualize_cached_color_mapping'] = \
                np.random.random_integers(50, 256, (max_uid + 1, 3)).astype(np.uint8)
            type(self.data_loader).cache['visualize_cached_color_mapping'][0] = (0, 0, 0)
        return type(self.data_loader).cache['visualize_cached_color_mapping']

    def set_color_for_uid(self, uid, rgb_color):
        self.color_mapping[uid] = rgb_color

    def set_color_for_name(self, name, rgb_color):
        self.color_mapping[self.data_loader.object_uid_name_mapping.name_to_uid[name]] = rgb_color

    def set_color(self, rgb_color, object_name_filter=None, object_uid_filter=None):
        if object_name_filter is not None or object_uid_filter is not None:
            object_filter = make_object_uid_filter(object_name_filter, object_uid_filter,
                                                   self.data_loader.object_uid_name_mapping.uid_to_name)
            uids = list(filter(object_filter, self.data_loader.object_uid_name_mapping.uid_to_name.keys()))
            self.color_mapping[uids] = rgb_color
        else:
            self.color_mapping[:] = rgb_color

    def get_color_for_uid(self, uid):
        return self.color_mapping[uid]

    def get_color_for_name(self, name):
        return self.color_mapping[self.data_loader.object_uid_name_mapping.name_to_uid[name]]
