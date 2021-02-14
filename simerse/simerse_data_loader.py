
from collections import namedtuple, deque
import os

from simerse import image_util, data_loader, logtools, iotools
from simerse.logtools import LogVerbosity

try:
    import torch as default_array_provider
    default_array_maker = default_array_provider.tensor
    default_image_maker = image_util.to_torch
except ImportError:
    import numpy as default_array_provider
    default_array_maker = default_array_provider.array
    default_image_maker = image_util.to_numpy

from simerse.box_format import BoxFormat, convert_single_2d, convert_single_3d

from simerse.simerse_keys import BuiltinDimension, MetaKey, BatchKey


# ==== Configuration stuff ====

supported_output_file_formats = [
    'JSON',
    'XML',
    'CSV',
]

file_format_extensions = {
    'JSON': 'json',
    'XML': 'xml',
    'CSV': 'csv',
}


def get_batch_file_path(root, batch_uid):
    return os.path.join(root, 'Batches', f'Batch_{batch_uid}')


ImageResolution = namedtuple('ImageResolution', ('width', 'height'))
ImageCoordinatesPoint = namedtuple('ImageCoordinatesPoint', ('x', 'y'))
SimerseLoaderCache = namedtuple('SimerseLoaderCache', ('batches', 'batch_sizes', 'batch_queue', 'total_cache_size'))
BoundingBoxPair = namedtuple('BoundingBoxPair', ('actor', 'object'))

box_format_mapping = {
    'MIN_MAX': BoxFormat.min_max,
    'MIN_EXTENTS': BoxFormat.min_extents,
    'CENTER_EXTENTS': BoxFormat.center_extents,
    'CENTER_HALF_EXTENTS': BoxFormat.center_half
}


class CacheLimitExceededWarning(Warning):
    pass


class LoaderIgnoredWarning(Warning):
    pass


class LoaderNotFoundWarning(Warning):
    pass


# ==== End configuration stuff ====


def get_box_format(meta_format):
    global box_format_mapping

    return box_format_mapping[meta_format]


def process_depth_capture(raw):
    return [raw[:, :, 0]]


def process_position_capture(raw, origin):
    import numpy as np

    origin = np.array(origin)
    return raw[:, :, :3] + origin.reshape((1, 1, 3))


def process_integer_capture(raw):
    import numpy as np

    raw = raw.astype(np.uint32)
    return (raw[:, :, 0] | (raw[:, :, 1] << 8) | (raw[:, :, 2] << 16)).astype(np.int32)


def process_norm_vector_capture(raw):
    return raw[:, :, :3] * 2 - 1


def load_meta(meta, explicit_folder, logger):
    if isinstance(meta, dict):

        if explicit_folder is None:
            logger('', ValueError('When using an explicit dataset meta dict the "explicit_folder" argument must'
                                  ' be a path to the dataset folder'))

        for meta_key, value in MetaKey.defaults.items():
            meta.setdefault(meta_key, value)
        summary = meta.setdefault(MetaKey.summary, {})

        for meta_key, value in MetaKey.summary_defaults.items():
            summary.setdefault(meta_key, value)

        return meta, explicit_folder

    else:
        import json

        if os.path.isdir(meta):
            meta_file = os.path.join(meta, 'meta.txt')
        else:
            meta_file = meta

        try:
            with open(meta_file, 'r') as m:
                meta = json.loads('{' + m.read().replace('\n', '') + '}')
        except IOError as e:
            logger.log(f'Failed to read meta file {meta_file}', e)

        logger.log(f'Read meta file {meta_file}')

        return meta, os.path.dirname(meta_file)


def attach_load_batch_function(meta_dict, clazz, root, logger, na_value):
    if meta_dict[MetaKey.batch_file_format] == 'JSON':
        attach_batch_file_loader_json(clazz, root, logger, na_value)
    elif meta_dict[MetaKey.batch_file_format] == 'XML':
        attach_batch_file_loader_xml(clazz, root, logger, na_value)
    elif meta_dict[MetaKey.batch_file_format] == 'CSV':
        attach_batch_file_loader_csv(clazz, root, logger, na_value)
    else:
        logger.log('', ValueError(
            f'meta_dict contains an invalid batch file format: {meta_dict["Batch File Format"]}; batch'
            f' file format must be one of {supported_output_file_formats}'
        ))


def attach_load_object_uid_function(cls):

    @data_loader.loader
    def load_object_uid(self, points):
        observer_map = ObserverMap()
        observer_map[None] = []
        processed_points = 0
        for point in points:
            observation = self.load_observation(point)

            observer_map[None].append(self.array_maker(tuple(
                object_value[BatchKey.object_uid] for object_value in
                observation[BatchKey.per_observation_values][BatchKey.object_values]
            )))

            for observer, value in observation[BatchKey.per_observer_values].items():
                observer_uids = observer_map.setdefault(observer, [])
                observer_uids.append(self.array_maker(tuple(
                    object_value[BatchKey.object_uid] for object_value in
                    value[BatchKey.object_values]
                )))

            processed_points += 1

            for uid_list in observer_map.values():
                if len(uid_list) != processed_points:
                    uid_list.append(self.array_maker(tuple()))

        return observer_map if len(observer_map) > 1 else tuple(observer_map.values())[0] if len(observer_map) == 1 \
            else ()

    load_object_uid.__set_name__(cls, BuiltinDimension.object_uid)


def is_image_ref(data):
    return len(data) == 2 and BatchKey.uri in data and BatchKey.resolution in data


_default_image_loaders = {}


def get_default_image_loaders():
    global _default_image_loaders

    if len(_default_image_loaders) == 0:
        import imageio
        _default_image_loaders = {
            BuiltinDimension.world_tangent: lambda root, image_ref:
                process_norm_vector_capture(imageio.imread(os.path.join(root, image_ref[BatchKey.uri]))),
            BuiltinDimension.world_bitangent: lambda root, image_ref:
                process_norm_vector_capture(imageio.imread(os.path.join(root, image_ref[BatchKey.uri]))),
            BuiltinDimension.world_normal: lambda root, image_ref:
                process_norm_vector_capture(imageio.imread(os.path.join(root, image_ref[BatchKey.uri]))),

            BuiltinDimension.visual_ldr: lambda root, image_ref:
                imageio.imread(os.path.join(root, image_ref[BatchKey.uri])),
            BuiltinDimension.visual_hdr: lambda root, image_ref:
                imageio.imread(os.path.join(root, image_ref[BatchKey.uri])),

            BuiltinDimension.segmentation: lambda root, image_ref:
                process_integer_capture(imageio.imread(os.path.join(root, image_ref[BatchKey.uri]))),
            BuiltinDimension.segmentation_outline: lambda root, image_ref:
                process_integer_capture(imageio.imread(os.path.join(root, image_ref[BatchKey.uri]))),

            BuiltinDimension.keypoint: lambda root, image_ref:
                process_integer_capture(imageio.imread(os.path.join(root, image_ref[BatchKey.uri]))),

            BuiltinDimension.uv: lambda root, image_ref: imageio.imread(os.path.join(root, image_ref[BatchKey.uri])),

            BuiltinDimension.position: lambda root, image_ref:
                process_position_capture(
                    imageio.imread(os.path.join(root, image_ref[BatchKey.position_capture][BatchKey.uri])),
                    image_ref[BatchKey.position_coordinates_origin]
                ),

            BuiltinDimension.depth: lambda root, image_ref:
                process_depth_capture(imageio.imread(os.path.join(root, image_ref[BatchKey.uri])))

        }

    return _default_image_loaders


class ObserverMap(dict):
    def __getitem__(self, item):
        if isinstance(item, int):
            return {k: v[item] for k, v in self.items()}
        else:
            return dict.__getitem__(self, item)


def build_image_only_loader(dimension):
    image_loader = get_default_image_loaders()[dimension]

    def load(self, points):
        observer_map = ObserverMap()
        num_images = 0

        for point in points:
            num_images += 1
            observation = self.load_observation(point)

            for observer, observer_value in observation[BatchKey.per_observer_values].items():
                observer_map.setdefault(observer, [])
                self._logger(f'Loading Dimension {BuiltinDimension.get_standard_name(dimension)} Capture ID {point}'
                             f' observed by {observer}', LogVerbosity.EVERYTHING)
                observer_map[observer].append(image_loader(self.folder, observer_value[dimension]))

            for image_list in observer_map.values():
                if len(image_list) < num_images:
                    image_list.append(())

        for observer, images in observer_map.items():
            observer_map[observer] = self.image_maker(images)

        return observer_map if len(observer_map) > 1 else tuple(observer_map.values())[0] if len(observer_map) == 1 \
            else ()

    return load


def build_only_per_observation_array_loader(dimension, jagged_by_observation=False,
                                            process=lambda x: x, collection=None):

    def load(self, points):
        array_maker = collection if collection is not None else self.array_maker
        value = []
        for point in points:
            observation = self.load_observation(point)
            value.append(process(observation[BatchKey.per_observation_values][dimension]))

        return tuple(array_maker(array) for array in value) if jagged_by_observation else array_maker(value)

    return load


def build_per_observation_array_loader(dimension, jagged_by_observation=False, jagged_by_object=False,
                                       per_observation_process=None, process=lambda x: x, collection=None):
    def load(self, points):
        array_maker = collection if collection is not None else self.array_maker
        value = []
        for point in points:
            observation = self.load_observation(point)
            if per_observation_process is not None:
                per_observation_process(observation[BatchKey.per_observation_values][dimension])

            value.append(tuple(
                process(object_value[dimension])
                for object_value in observation[BatchKey.per_observation_values][BatchKey.object_values]
            ))

        if jagged_by_object:
            return tuple(tuple(array_maker(obj) for obj in array) for array in value)
        elif jagged_by_observation:
            return tuple(array_maker(array) for array in value)
        else:
            return array_maker(value)

    return load


def build_only_per_observer_array_loader(dimension, jagged_by_observation=False, per_observation_process=None,
                                         process=lambda x: x, collection=None):

    def load(self, points):
        array_maker = collection if collection is not None else self.array_maker
        observer_map = ObserverMap()
        num_points = 0

        for point in points:
            num_points += 1
            observation = self.load_observation(point)
            if per_observation_process is not None:
                per_observation_process(observation[BatchKey.per_observation_values][dimension])

            for observer, observer_value in observation[BatchKey.per_observer_values].items():
                observer_list = observer_map.setdefault(observer, [])
                observer_list.append(process(observer_value[dimension]))

            for observer_list in observer_map.values():
                if len(observer_list) < num_points:
                    observer_list.append(())

        for observer, value in observer_map.items():
            observer_map[observer] = tuple(array_maker(array) for array in value) if jagged_by_observation \
                else array_maker(value)

        return observer_map if len(observer_map) > 1 else tuple(observer_map.values())[0] if len(observer_map) == 1 \
            else ()

    return load


def build_per_observer_array_loader(dimension, jagged_by_observation=False, jagged_by_object=False,
                                    per_observation_process=None, per_observer_process=None, process=lambda x: x,
                                    collection=None):

    def load(self, points):
        array_maker = collection if collection is not None else self.array_maker
        observer_map = ObserverMap()
        num_points = 0

        for point in points:
            num_points += 1
            observation = self.load_observation(point)
            if per_observation_process is not None:
                per_observation_process(observation[BatchKey.per_observation_values][dimension])

            for observer, observer_value in observation[BatchKey.per_observer_values].items():
                if per_observer_process is not None:
                    per_observer_process(observer_value[dimension])

                observer_map.setdefault(observer, [])
                observer_map[observer].append(tuple(
                    process(object_value[dimension])
                    for object_value in observer_value[BatchKey.object_values]
                ))

            for observer_list in observer_map.values():
                if len(observer_list) < num_points:
                    observer_list.append(())

        for observer, value in observer_map.items():
            if jagged_by_object:
                observer_map[observer] = tuple(tuple(array_maker(obj) for obj in array) for array in value)
            elif jagged_by_observation:
                observer_map[observer] = tuple(array_maker(array) for array in value)
            else:
                observer_map[observer] = array_maker(value)

        return observer_map if len(observer_map) > 1 else tuple(observer_map.values())[0] if len(observer_map) == 1 \
            else ()

    return load


def get_object_bounding_box(box_data):
    return box_data[BatchKey.object_bounding_box] if isinstance(box_data, dict) else box_data


def get_actor_bounding_box(box_data):
    return box_data[BatchKey.actor_bounding_box] if isinstance(box_data, dict) else None


def build_local_bb3_loader():

    def load(self, points):
        object_boxes = []
        actor_boxes = []
        for point in points:
            observation = self.load_observation(point)

            object_boxes.append(self.array_maker(tuple(
                get_object_bounding_box(object_value[BuiltinDimension.local_bounding_box_3d])
                for object_value in observation[BatchKey.per_observation_values][BatchKey.object_values]
            )))

            for object_value in observation[BatchKey.per_observation_values][BatchKey.object_values]:
                if isinstance(object_value[BuiltinDimension.local_bounding_box_3d], dict):
                    actor_boxes.append(tuple(
                        get_actor_bounding_box(object_value[BuiltinDimension.local_bounding_box_3d])
                        for object_value in observation[BatchKey.per_observation_values][BatchKey.object_values]
                    ))
                    actor_boxes[-1] = tuple(
                        self.array_maker(box) if box is not None else box for box in actor_boxes[-1]
                    )
                    break

            if 0 < len(actor_boxes) < len(object_boxes):
                actor_boxes.append(())

        return BoundingBoxPair(actor_boxes, object_boxes) if len(actor_boxes) > 0 else object_boxes

    return load


def build_global_bb3_loader():

    def load(self, points):
        object_boxes = []
        actor_boxes = []
        for point in points:
            observation = self.load_observation(point)
            stored_format = box_format_mapping[
                observation[BatchKey.per_observation_values][BuiltinDimension.global_bounding_box_3d]
            ]

            object_boxes.append(self.array_maker(tuple(
                convert_single_3d(get_object_bounding_box(object_value[BuiltinDimension.global_bounding_box_3d]),
                                  stored_format, self.bounding_box_3d_format)
                for object_value in observation[BatchKey.per_observation_values][BatchKey.object_values]
            )))

            for object_value in observation[BatchKey.per_observation_values][BatchKey.object_values]:
                if isinstance(object_value[BuiltinDimension.global_bounding_box_3d], dict):
                    actor_boxes.append(tuple(
                        get_actor_bounding_box(object_value[BuiltinDimension.global_bounding_box_3d])
                        for object_value in observation[BatchKey.per_observation_values][BatchKey.object_values]
                    ))
                    actor_boxes[-1] = tuple(
                        self.array_maker(convert_single_3d(box, stored_format, self.bounding_box_3d_format))
                        if box is not None else box
                        for box in actor_boxes[-1]
                    )
                    break

            if 0 < len(actor_boxes) < len(object_boxes):
                actor_boxes.append(())

        return BoundingBoxPair(actor_boxes, object_boxes) if len(actor_boxes) > 0 else object_boxes

    return load


def build_custom_bb3_loader():

    def load(self, points):
        object_boxes = []
        actor_boxes = []
        for point in points:
            observation = self.load_observation(point)

            object_boxes.append(tuple(
                self.array_maker(get_object_bounding_box(object_value[BuiltinDimension.local_bounding_box_3d]))
                for object_value in observation[BatchKey.per_observation_values][BatchKey.object_values]
            ))

            for object_value in observation[BatchKey.per_observation_values][BatchKey.object_values]:
                if isinstance(object_value[BuiltinDimension.local_bounding_box_3d], dict):
                    actor_boxes.append(tuple(
                        get_actor_bounding_box(object_value[BuiltinDimension.local_bounding_box_3d])
                        for object_value in observation[BatchKey.per_observation_values][BatchKey.object_values]
                    ))
                    actor_boxes[-1] = tuple(
                        self.array_maker(box) if box is not None else box for box in actor_boxes[-1]
                    )
                    break

            if 0 < len(actor_boxes) < len(object_boxes):
                actor_boxes.append(())

        return BoundingBoxPair(actor_boxes, object_boxes) if len(actor_boxes) > 0 else object_boxes

    return load


class BoundingBox2DLoaderBuilder:
    def __init__(self, dimension, jagged_by_object):
        self.stored_format = None
        self.target_format = None
        self.dimension = dimension
        self.jagged_by_object = jagged_by_object

    def per_observation_process(self, per_observation_value):
        self.stored_format = box_format_mapping[per_observation_value]

    def process(self, box):
        return convert_single_2d(box, self.stored_format, self.cls.bounding_box_2d_format)

    def __call__(self, cls):
        self.cls = cls
        return build_per_observer_array_loader(
            self.dimension, jagged_by_observation=True, jagged_by_object=self.jagged_by_object,
            per_observation_process=self.per_observation_process, process=self.process
        )


def get_camera_transform_processor(cls):
    def process_camera_transform(camera_view_value):
        camera_view_value[BatchKey.camera_transform_matrix] = \
            cls.array_maker(camera_view_value[BatchKey.camera_transform_matrix])
        return camera_view_value

    return process_camera_transform


def get_object_transform_processor(cls):
    def process_object_transform(object_transform):
        if isinstance(object_transform, dict):
            object_transform[BatchKey.spline_object_transform] = \
                cls.array_maker(object_transform[BatchKey.spline_object_transform])
            return object_transform
        else:
            return cls.array_maker(object_transform)

    return process_object_transform


_default_loader_builders = {}


def get_default_loader_builders():
    global _default_loader_builders
    if len(_default_loader_builders) == 0:
        _default_loader_builders = {
            BuiltinDimension.world_tangent:
                lambda cls: build_image_only_loader(BuiltinDimension.world_tangent),
            BuiltinDimension.world_bitangent:
                lambda cls: build_image_only_loader(BuiltinDimension.world_bitangent),
            BuiltinDimension.world_normal:
                lambda cls: build_image_only_loader(BuiltinDimension.world_normal),

            BuiltinDimension.visual_ldr: lambda cls: build_image_only_loader(BuiltinDimension.visual_ldr),
            BuiltinDimension.visual_hdr: lambda cls: build_image_only_loader(BuiltinDimension.visual_hdr),

            BuiltinDimension.segmentation:
                lambda cls: build_image_only_loader(BuiltinDimension.segmentation),
            BuiltinDimension.segmentation_outline:
                lambda cls: build_image_only_loader(BuiltinDimension.segmentation_outline),

            # TODO: implement this whenever keypoints are implemented in the Unreal plugin
            BuiltinDimension.keypoint: None,

            BuiltinDimension.uv: lambda cls: build_image_only_loader(BuiltinDimension.uv),

            BuiltinDimension.position: lambda cls: build_image_only_loader(BuiltinDimension.position),

            BuiltinDimension.depth: lambda cls: build_image_only_loader(BuiltinDimension.depth),

            BuiltinDimension.local_bounding_box_3d: lambda cls: build_local_bb3_loader(),
            BuiltinDimension.global_bounding_box_3d: lambda cls: build_global_bb3_loader(),
            BuiltinDimension.custom_bounding_box_3d: lambda cls: build_custom_bb3_loader(),

            BuiltinDimension.total_bounding_box_2d: BoundingBox2DLoaderBuilder(
                BuiltinDimension.total_bounding_box_2d, False),
            BuiltinDimension.connected_bounding_box_2d: lambda cls: BoundingBox2DLoaderBuilder(
                BuiltinDimension.connected_bounding_box_2d, True),

            BuiltinDimension.time: lambda cls: build_only_per_observation_array_loader(cls, BuiltinDimension.time),
            BuiltinDimension.object_transform: lambda cls: build_per_observation_array_loader(
                BuiltinDimension.object_transform, collection=tuple, process=get_object_transform_processor(cls)),
            BuiltinDimension.mesh_name: lambda cls: build_per_observation_array_loader(
                BuiltinDimension.mesh_name, collection=tuple),
            BuiltinDimension.unreal_name: lambda cls: build_per_observation_array_loader(
                BuiltinDimension.unreal_name, collection=tuple),
            BuiltinDimension.camera_view: lambda cls: build_only_per_observer_array_loader(
                BuiltinDimension.camera_view, collection=tuple, process=get_camera_transform_processor(cls)),

        }

    return _default_loader_builders


mebibytes = 2 ** 20


# noinspection PyPep8Naming
def SimerseDataLoader(
    meta, explicit_folder=None,
    logger=logtools.default_logger,
    custom_array_maker=default_array_maker, custom_image_maker=default_image_maker,
    na_value='N/A',
    cache_limit=256 * mebibytes,
    bounding_box_2d_load_format=BoxFormat.min_max, bounding_box_3d_load_format=BoxFormat.min_max,
    **custom_loaders
):

    meta_dict, root = load_meta(meta, explicit_folder, logger)
    logger.log(f'Retrieved meta data:\n\t{meta_dict}', LogVerbosity.EVERYTHING)

    # noinspection PyMethodParameters
    class SimerseDataLoaderInstance(data_loader.DataLoader):
        _logger = logger

        folder = root

        @staticmethod
        def array_maker(data):
            return custom_array_maker(data)

        @staticmethod
        def image_maker(data):
            return custom_image_maker(data)

        bounding_box_2d_format = bounding_box_2d_load_format
        bounding_box_3d_format = bounding_box_3d_load_format

        batch_cache_limit = cache_limit

        name = meta_dict[MetaKey.dataset_name]
        description = meta_dict[MetaKey.description]

        num_observations = meta_dict[MetaKey.summary][MetaKey.observation_count]
        batch_size = meta_dict[MetaKey.summary][MetaKey.batch_size]

        license = meta_dict[MetaKey.license]

        def summary(self):
            dimensions_string = ''

            # noinspection PyTypeChecker
            for dimension in self.dimensions:
                dimensions_string += f'\t{dimension}\n'

            return f"""
====== Summary of dataset: {self.name} ======

{self.description}

Dimensions: 
{dimensions_string}

Size:
    {self.num_observations} observations

License:
{self.license}
"""

        simerse_cache = SimerseLoaderCache(batches={}, batch_queue=deque(), batch_sizes={}, total_cache_size=[0])

        @staticmethod
        def load_observation(uid):
            return SimerseDataLoaderInstance.load_batch(
                uid // SimerseDataLoaderInstance.batch_size
            )[BatchKey.observations][uid % SimerseDataLoaderInstance.batch_size]

        @staticmethod
        def is_na(value):
            return isinstance(value, type(na_value)) and value == na_value

        @property
        def na_value(self):
            return na_value

    # attach the appropriate batch loader for the dataset
    logger.log('Attaching load_batch function')
    attach_load_batch_function(meta_dict, SimerseDataLoaderInstance, root, logger, na_value)

    # attach the data loader for object uid dimension
    logger.log('Attaching data loader for dimension object_uid')
    attach_load_object_uid_function(SimerseDataLoaderInstance)

    # get the dimension names as a set because we will be checking for containment/removing elements
    dimensions_set = set(meta_dict[MetaKey.summary][MetaKey.dimensions] + [BuiltinDimension.object_uid])
    dimensions_set.remove(BuiltinDimension.object_uid)

    # attach any custom data loaders
    logger.log('Attaching custom data loaders')
    for dimension_name, custom_loader in custom_loaders.items():
        if dimension_name in dimensions_set:
            logger.log(f'Attaching custom loader for dimension {BuiltinDimension.get_standard_name(dimension_name)}')
            dimensions_set.remove(dimension_name)
            loader = data_loader.loader(custom_loader(SimerseDataLoaderInstance))
            loader.__set_name__(SimerseDataLoaderInstance, dimension_name)
        else:
            logger.log(f'Could not attach custom loader for dimension '
                       f'{BuiltinDimension.get_standard_name(dimension_name)}'
                       f' because the dataset does not have that dimension', LoaderIgnoredWarning)

    # try to attach default loaders for the remaining dimensions that don't have any
    logger.log('Attaching default data loaders')
    logger.log(f'Remaining dimensions that want a default loader: {dimensions_set}', LogVerbosity.EVERYTHING)

    default_loader_builders = get_default_loader_builders()

    for dimension_name in dimensions_set:
        if dimension_name in default_loader_builders:
            logger.log(f'Building default loader for dimension {dimension_name}')
            loader_builder = default_loader_builders[dimension_name]
            loader = data_loader.loader(loader_builder(SimerseDataLoaderInstance))
            loader.__set_name__(SimerseDataLoaderInstance, dimension_name)
        else:
            logger.log(f'Dimension {BuiltinDimension.get_standard_name(dimension_name)} was not given a custom loader '
                       f'and there is no default loader for it. You will not be able to load this dimension unless you'
                       f' attach a loader with the DataLoader.attach_loader function', LoaderNotFoundWarning)

    return SimerseDataLoaderInstance()


def safe_add_batch(batch_number, cache, data, cache_limit, logger):
    logger.log(f'Beginning analysis of loaded batch size')
    cache.batch_sizes[batch_number] = iotools.get_size_recursive(data)
    cache.total_cache_size[0] += cache.batch_sizes[batch_number]
    logger.log(f'Loaded batch {batch_number} into memory with a size of'
               f' {cache.batch_sizes[batch_number] // 2 ** 10} kiB', LogVerbosity.EVERYTHING)

    if cache.total_cache_size[0] >= cache_limit:
        if len(cache.batch_queue) == 1:
            logger.log(f'Batch cache limit of {cache_limit // 2 ** 20} MiB exceeded.',
                       ValueError(f'Batch cache limit {cache_limit // 2 ** 20} MiB is too small'
                                  f'to store even one batch. Please increase it and try again!'))
        else:
            logger.log(f'Batch cache limit of {cache_limit // 2 ** 20} MiB exceeded. '
                       f'Attempting to remove oldest batch data to make space', CacheLimitExceededWarning)

        while len(cache.batch_queue) > 1 and cache.total_cache_size[0] >= cache_limit:
            last_batch = cache.batch_queue.popleft()
            cache.batches.pop(last_batch)
            cached_size = cache.batch_sizes.pop(last_batch)
            cache.total_cache_size[0] -= cached_size
            logger.log(f'Cached batch {last_batch} removed, freeing {cached_size} bytes')

        if cache.total_cache_size[0] >= cache_limit:
            logger.log(f'Batch cache limit of {cache_limit // 2 ** 20} MiB exceeded.',
                       ValueError(f'Batch cache limit {cache_limit // 2 ** 20} MiB is too small'
                                  f'to store even one batch. Please increase it and try again!'))

    logger.log('Finished analyzing loaded batch size')


def attach_batch_file_loader_json(cls, root, logger, __):
    import json

    cache = cls.simerse_cache

    def load_batch(batch_number):
        if batch_number not in cache.batches:
            try:
                logger.log(f'Reading JSON batch file {batch_number}')
                with open(get_batch_file_path(root, batch_number) + '.json', 'r') as f:
                    data = json.load(f)

                    cache.batches[batch_number] = data
                    cache.batch_queue.append(batch_number)

                    safe_add_batch(batch_number, cache, data, cls.batch_cache_limit, logger)

            except IOError as e:
                logger.log(f'Failed to read JSON batch file {batch_number}', e)
            except json.decoder.JSONDecodeError as e:
                logger.log(f'Failed to parse JSON batch file {batch_number}', e)
            except Exception as e:
                logger.log(f'Failed to parse JSON batch file {batch_number}', e)

        return cache.batches[batch_number]

    cls.load_batch = load_batch


def parse_xml_dimension_value_recursive(node, na_value):
    if len(node.text) != 0:
        try:
            return int(node.text)
        except ValueError:
            try:
                float(node.text)
            except ValueError:
                return node.text
    else:
        if node.find(BatchKey.xml_array_element) is not None:
            return [parse_xml_dimension_value_recursive(array_element, na_value) for array_element in node]
        elif node.find(BatchKey.xml_map_pair) is not None:
            return {
                parse_xml_dimension_value_recursive(map_pair.find(BatchKey.xml_map_key), na_value):
                    parse_xml_dimension_value_recursive(map_pair.find(BatchKey.xml_map_value), na_value)
                for map_pair in node
            }
        else:
            return na_value


def parse_xml_observation_value(node, na_value):
    final_value = {}
    for value in node.iterfind(BatchKey.xml_dimension_value):
        final_value[value.attrib[BatchKey.xml_dimension_name]] = parse_xml_dimension_value_recursive(value, na_value)

    final_value[BatchKey.object_values] = [
        parse_xml_dimension_value_recursive(object_value, na_value) for object_value in
        node.find(BatchKey.object_values)
    ]

    return final_value


def attach_batch_file_loader_xml(cls, root, logger, na_value):
    import xml.etree.ElementTree as ElementTree

    cache = cls.simerse_cache

    def load_batch(batch_number):
        if batch_number not in cache.batches:
            logger.log(f'Reading XML batch file {batch_number}')
            try:
                root_element = ElementTree.parse(get_batch_file_path(root, batch_number) + '.xml').getroot()

                logger.log(f'Collecting data from XML batch file {batch_number}')
                batch = []
                for observation in root_element.iterfind(BatchKey.xml_observation):
                    per_observer_values = {}

                    current_observation = {
                        BatchKey.observation_uid: int(observation.find(BatchKey.observation_uid).text),
                        BatchKey.per_observation_values:
                            parse_xml_observation_value(observation.find(BatchKey.per_observation_values), na_value),
                        BatchKey.per_observer_values: per_observer_values
                    }
                    batch.append(current_observation)

                    for observer_value in observation.find(BatchKey.per_observer_values):
                        per_observer_values[observer_value.attrib[BatchKey.xml_observer_name]] = \
                            parse_xml_observation_value(observer_value, na_value)

                data = {BatchKey.batch_uid: batch_number, BatchKey.observations: batch}

                cache.batches[batch_number] = data
                cache.batch_queue.append(batch_number)

                safe_add_batch(batch_number, cache, data, cls.batch_cache_limit, logger)

            except IOError as e:
                logger.log(f'Failed to read XML batch file {batch_number}', e)
            except ElementTree.ParseError as e:
                logger.log(f'Failed to parse XML batch file {batch_number}', e)
            except Exception as e:
                logger.log(f'Failed to parse XML batch file {batch_number}', e)

        return cache.batches[batch_number]

    cls.load_batch = load_batch


def attach_batch_file_loader_csv(cls, root, logger, na_value):
    import csv
    import json

    cache = cls.simerse_cache

    def load_batch(batch_number):
        if batch_number not in cache.batches:
            logger.log(f'Reading CSV batch file {batch_number}')
            try:
                with open(get_batch_file_path(root, batch_number) + '.csv', 'r') as f:
                    raw_csv_data = list(csv.reader(f))

                logger.log(f'Collecting data from CSV batch file {batch_number}')
                observations_by_uid = {}

                dimensions = raw_csv_data[0]

                for row in raw_csv_data:
                    observation_uid = int(row[BatchKey.csv_observation_uid_col])
                    if observation_uid not in observations_by_uid:
                        observations_by_uid[observation_uid] = {
                            BatchKey.observation_uid: observation_uid,
                            BatchKey.per_observation_values: {
                                BatchKey.object_values: []
                            },
                            BatchKey.per_observer_values: {}
                        }
                    observation = observations_by_uid[observation_uid]
                    if row[BatchKey.csv_observer_col] == na_value:
                        if row[BatchKey.csv_object_uid_col] == na_value:
                            per_observation_values = observation[BatchKey.per_observation_values]
                            for col_index in range(BatchKey.csv_first_dimension_col_index, len(row)):
                                if row[col_index] != na_value:
                                    per_observation_values[dimensions[col_index]] = json.loads(str(row[col_index]))
                        else:
                            per_object_value = {}
                            for col_index in range(BatchKey.csv_first_dimension_col_index, len(row)):
                                if row[col_index] != na_value:
                                    per_object_value[dimensions[col_index]] = json.loads(str(row[col_index]))
                            observation[BatchKey.per_observation_values][BatchKey.object_values].append(
                                per_object_value
                            )
                    else:
                        # noinspection PyTypeChecker
                        observer_value = observation[BatchKey.per_observer_values].setdefault(
                            row[BatchKey.csv_observer_col], {BatchKey.object_values: []}
                        )
                        if row[BatchKey.csv_object_uid_col] == na_value:
                            for col_index in range(BatchKey.csv_first_dimension_col_index, len(row)):
                                if row[col_index] != na_value:
                                    observer_value[dimensions[col_index]] = json.loads(str(row[col_index]))
                        else:
                            per_object_value = {}
                            for col_index in range(BatchKey.csv_first_dimension_col_index, len(row)):
                                if row[col_index] != na_value:
                                    per_object_value[dimensions[col_index]] = json.loads(str(row[col_index]))
                            observer_value[BatchKey.object_values].append(per_object_value)

                data = {BatchKey.batch_uid: batch_number, BatchKey.observations: list(observations_by_uid.values())}

                cache.batches[batch_number] = data
                cache.batch_queue.append(batch_number)

                safe_add_batch(batch_number, cache, data, cls.batch_cache_limit, logger)

            except IOError as e:
                logger.log(f'Failed to read CSV batch file {batch_number}', e)
            except Exception as e:
                logger.log(f'Failed to parse CSV batch file {batch_number}', e)

        return cache.batches[batch_number]

    cls.load_batch = load_batch
