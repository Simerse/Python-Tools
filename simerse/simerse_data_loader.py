
from collections import namedtuple
import os

from simerse import image_util, data_loader

try:
    import torch as array_provider
    array_maker = array_provider.tensor
    image_maker = image_util.to_torch
except ImportError:
    import numpy as array_provider
    array_maker = array_provider.array
    image_maker = image_util.to_numpy

from simerse.box_format import BoxFormat

from simerse.simerse_keys import BuiltinDimension


# ==== Configuration stuff ====

progress_reporter = print

log_verbosity_level = 1

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

meta_dict_key_mappings = {
    'Batch File Format': 'Batch File Format',
    'Mapping File Format': 'Mapping File Format',
    'Dataset Name': 'Dataset Name',
    'Description': 'Description',
    'Summary': 'Summary',
    'Dimensions': 'Dimensions',
    'Total Observations': 'Total Observations',
    'Observation Batch Size': 'Observation Batch Size',
    'Capture Resolution': 'Capture Resolution',
    'Image Coordinates Top Left': 'Image Coordinates Top Left',
    '2D Image-Aligned Bounding Box Format': 'Image-Aligned 2D Bounding Box Format',
    '3D Globally-Aligned Bounding Box Format': 'Globally-Aligned 3D Bounding Box Format',
    'NA Value': 'NA Value',
    'License': 'License'
}

capture_data_key = 'CaptureData'
observations_key = 'Observations'
observation_key = 'Observation'
objects_key = 'ObservationObjects'
visual_ldr_capture_key = BuiltinDimension.visual_ldr
visual_hdr_capture_key = BuiltinDimension.visual_hdr
segmentation_capture_key = BuiltinDimension.segmentation
segmentation_rle_key = BuiltinDimension.segmentation_rle
segmentation_polygon_key = BuiltinDimension.segmentation_polygon
keypoints_key = BuiltinDimension.keypoints
segmentation_outline_capture_key = BuiltinDimension.segmentation_outline
uv_capture_key = BuiltinDimension.uv
depth_capture_key = BuiltinDimension.depth
world_normal_capture_key = BuiltinDimension.world_normal
world_tangent_capture_key = BuiltinDimension.world_tangent
world_bitangent_capture_key = BuiltinDimension.world_bitangent
world_position_capture_key = BuiltinDimension.world_position
world_position_origin_key = BuiltinDimension.world_origin
projection_type_key = BuiltinDimension.camera_projection
view_parameter_key = BuiltinDimension.camera_view
camera_transform_key = BuiltinDimension.camera_transform
object_transform_key = BuiltinDimension.object_transform
time_key = BuiltinDimension.time

image_aligned_2d_total_bounding_box_key = BuiltinDimension.bounding_box_2d_total
image_aligned_2d_contiguous_bounding_box_key = BuiltinDimension.bounding_box_2d_contiguous
locally_aligned_3d_bounding_box_key = BuiltinDimension.bounding_box_3d_local
globally_aligned_3d_bounding_box_key = BuiltinDimension.bounding_box_3d_global
custom_3d_bounding_box_key = BuiltinDimension.bounding_box_3d_custom

observation_object_uid_key = BuiltinDimension.object_uid
batch_number_key = 'BatchFileNumber'
observation_uid_key = 'ObservationUID'
actor_name_key = 'ActorName'
mapping_key = 'Mapping'
relative_object_uid_name_mapping_path = 'Mappings/ObservationObjectUID-ActorName'


capture_data_dimensions = {
    visual_ldr_capture_key,
    visual_hdr_capture_key,
    segmentation_capture_key,
    segmentation_outline_capture_key,
    uv_capture_key,
    depth_capture_key,
    world_position_origin_key,
    world_position_capture_key,
    world_normal_capture_key,
    world_tangent_capture_key,
    world_bitangent_capture_key,
    time_key,
    projection_type_key,
    view_parameter_key,
    camera_transform_key,
}

per_object_dimensions = {
    keypoints_key,
    object_transform_key,
    image_aligned_2d_total_bounding_box_key,
    image_aligned_2d_contiguous_bounding_box_key,
    locally_aligned_3d_bounding_box_key,
    globally_aligned_3d_bounding_box_key,
    custom_3d_bounding_box_key,
    segmentation_rle_key,
    segmentation_polygon_key,
}

json_parse_dimensions = {
    keypoints_key,
    object_transform_key,
    camera_transform_key,
    view_parameter_key,
    time_key,
    segmentation_rle_key,
    segmentation_polygon_key,
    world_position_origin_key,
    image_aligned_2d_total_bounding_box_key,
    image_aligned_2d_contiguous_bounding_box_key,
    locally_aligned_3d_bounding_box_key,
    globally_aligned_3d_bounding_box_key,
    custom_3d_bounding_box_key,
    observation_object_uid_key,
}

default_capture_dimensions = [
    visual_ldr_capture_key,
    visual_hdr_capture_key,
]

vector_capture_dimensions = [
    world_normal_capture_key,
    world_tangent_capture_key,
    world_bitangent_capture_key,
]

integer_capture_dimensions = [
    segmentation_capture_key,
    segmentation_outline_capture_key,
]

bounding_box_dimensions = [
    image_aligned_2d_total_bounding_box_key,
    image_aligned_2d_contiguous_bounding_box_key,
    locally_aligned_3d_bounding_box_key,
    globally_aligned_3d_bounding_box_key,
    custom_3d_bounding_box_key,
]


def get_batch_file_path(root, number):
    return os.path.join(root, 'Batches', f'BatchFile_{number}')


def get_batch_number_observation_uid_mapping_file_path(root, file_format):
    return os.path.join(root, 'Mappings', f'BatchNumber-ObservationUID.{file_format}')


ImageResolution = namedtuple('ImageResolution', ('width', 'height'))
ImageCoordinatesPoint = namedtuple('ImageCoordinatesPoint', ('x', 'y'))
ObjectUIDNameMapping = namedtuple('ObjectUIDNameMapping', ('uid_to_name', 'name_to_uid'))
BatchNumberObservationUIDMapping = namedtuple(
    'BatchNumberObservationUIDMapping', ('batch_number_to_uid', 'uid_to_batch_number', 'uid_indices')
)

box_format_mapping = {
    'Min-Max': BoxFormat.min_max,
    'Min-Extents': BoxFormat.min_extents,
    'Center-Extents': BoxFormat.center_extents,
    'Center-HalfExtents': BoxFormat.center_half
}

# ==== End configuration stuff ====


def get_box_format(meta_format):
    global box_format_mapping

    return box_format_mapping[meta_format]


def get_meta_key(local_key):
    global meta_dict_key_mappings

    return meta_dict_key_mappings[local_key]


def log(message, verbosity=1):
    if progress_reporter and verbosity <= log_verbosity_level:
        progress_reporter(f'Simerse Data Loader: {message}')


def load_object_uid_name_mapping(root, file_format):
    absolute_file_path = f'{root}/{relative_object_uid_name_mapping_path}'
    if file_format == 'JSON':
        import json

        log('Reading JSON ObjectUID-Name mapping file')
        with open(f'{absolute_file_path}.json', 'r') as f:
            mapping_data = json.load(f)

        log('Constructing mapping', 2)
        uid_to_name = {pair[observation_object_uid_key]: pair[actor_name_key] for pair in mapping_data[mapping_key]}
        log('Constructing inverse mapping', 2)
        return ObjectUIDNameMapping(uid_to_name, {name: uid for uid, name in uid_to_name.items()})
    elif file_format == 'XML':
        import xml.etree.ElementTree as ElementTree

        log('Reading XML ObjectUID-Name mapping file')
        et = ElementTree.parse(f'{absolute_file_path}.xml')
        root = et.getroot()

        log('Constructing mapping', 2)
        uid_to_name = {int(pair.find(observation_object_uid_key).text): pair.find(actor_name_key).text for pair in root}
        log('Constructing inverse mapping', 2)
        return ObjectUIDNameMapping(uid_to_name, {name: uid for uid, name in uid_to_name.items()})
    elif file_format == 'CSV':
        import csv

        log('Reading CSV ObjectUID-Name mapping file and constructing mapping')
        with open(f'{absolute_file_path}.csv', 'r') as f:
            uid_to_name = {int(row[0]): row[1] for row in csv.reader(f)}

        log('Constructing inverse mapping', 2)
        return ObjectUIDNameMapping(uid_to_name, {name: uid for uid, name in uid_to_name.items()})
    else:
        raise ValueError(f'ObjectUID-Name mapping format "{file_format}" is not a valid mapping file format; must be'
                         f' one of {supported_output_file_formats}')


def parse_list_of_integers(list_string):
    return list(map(int, list_string.strip('][').split(',')))


def parse_list_of_floats(list_string):
    return list(map(int, list_string.strip('][').split(',')))


def load_batch_number_observation_uid_mapping(root, file_format):
    absolute_file_path = get_batch_number_observation_uid_mapping_file_path(root, file_format)
    if file_format == 'JSON':
        import json

        log('Reading JSON batch number-observation uid mapping file')
        with open(absolute_file_path, 'r') as f:
            json_data = json.load(f)

        log('Constructing mapping', 2)
        batch_number_to_uid = {pair[batch_number_key]: pair[observation_uid_key] for pair in json_data[mapping_key]}

        log('Constructing inverse mapping and uid indices', 2)
        inverse = {}
        uid_indices = {}
        for batch_number, observation_uids in batch_number_to_uid.items():
            inverse.update((uid, batch_number) for uid in observation_uids)
            uid_indices.update((uid, index) for index, uid in enumerate(observation_uids))
        return BatchNumberObservationUIDMapping(batch_number_to_uid, inverse, uid_indices)
    elif file_format == 'XML':
        import xml.etree.ElementTree as ElementTree

        log('Reading XML batch number-observation uid mapping file')
        et = ElementTree.parse(absolute_file_path)
        root = et.getroot()

        log('Constructing mapping', 2)
        batch_number_to_uid = {
            int(pair.find(batch_number_key).text): parse_list_of_integers(pair.find(observation_uid_key).text)
            for pair in root
        }

        log('Constructing inverse mapping and uid indices', 2)
        inverse = {}
        uid_indices = {}
        for batch_number, observation_uids in batch_number_to_uid.items():
            inverse.update((uid, batch_number) for uid in observation_uids)
            uid_indices.update((uid, index) for index, uid in enumerate(observation_uids))
        return BatchNumberObservationUIDMapping(batch_number_to_uid, inverse, uid_indices)
    elif file_format == 'CSV':
        import csv

        log('Reading CSV batch number-observation uid mapping file and constructing mapping')
        batch_number_to_uid = {}
        with open(absolute_file_path, 'r') as f:
            for row in csv.reader(f):
                if len(row) >= 2:
                    batch_number_to_uid[int(row[0])] = [int(uid) for uid in row[1:]]

        log('Constructing inverse mapping and uid indices', 2)
        inverse = {}
        uid_indices = {}
        for batch_number, observation_uids in batch_number_to_uid.items():
            inverse.update((uid, batch_number) for uid in observation_uids)
            uid_indices.update((uid, index) for index, uid in enumerate(observation_uids))
        return BatchNumberObservationUIDMapping(batch_number_to_uid, inverse, uid_indices)
    else:
        raise ValueError(f'Batch Number-Observation UID mapping format "{file_format}" is not a valid mapping file'
                         f' format; must be one of {supported_output_file_formats}')


def process_depth_capture(raw):
    return array_maker([raw[:, :, 0]])


def process_world_position_capture(raw, origin):
    import numpy as np

    origin = np.array(origin)
    return raw[:, :, :3] - origin.reshape((1, 1, 3))


def process_integer_capture(raw):
    import numpy as np

    raw = raw.astype(np.uint32)
    return array_maker((raw[:, :, 0] | (raw[:, :, 1] << 8) | (raw[:, :, 2] << 16)).astype(np.int32))


# noinspection PyPep8Naming
def SimerseDataLoader(meta_file, **custom_loaders):
    import json

    log(f'Reading meta file {meta_file}')
    with open(meta_file, 'r') as m:
        meta_dict = json.loads('{' + m.read().replace('\n', '') + '}')

    root = os.path.dirname(meta_file)

    # noinspection PyMethodParameters
    class SimerseDataLoaderInstance(data_loader.DataLoader):
        name = meta_dict[get_meta_key('Dataset Name')]

        description = meta_dict[get_meta_key('Description')]

        dimensions = meta_dict[get_meta_key('Summary')][get_meta_key('Dimensions')] + ['ObservationObjectUID'] + \
            list(custom_loaders)

        num_observations = meta_dict[get_meta_key('Summary')][get_meta_key('Total Observations')]

        batch_size = meta_dict[get_meta_key('Summary')][get_meta_key('Observation Batch Size')]

        capture_resolution = ImageResolution(*meta_dict[get_meta_key('Summary')][get_meta_key('Capture Resolution')])

        image_coordinate_origin = ImageCoordinatesPoint(
            *meta_dict[get_meta_key('Summary')][get_meta_key('Image Coordinates Top Left')]
        )

        box_format2d = get_box_format(
            meta_dict[get_meta_key('Summary')][get_meta_key('2D Image-Aligned Bounding Box Format')]
        )

        box_format3d = get_box_format(
            meta_dict[get_meta_key('Summary')][get_meta_key('3D Globally-Aligned Bounding Box Format')]
        )

        na_value = meta_dict[get_meta_key('NA Value')]

        license = meta_dict[get_meta_key('License')]

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

Capture Resolution:
    width={self.capture_resolution.width}, height={self.capture_resolution.height}

License:
{self.license}
"""

        @data_loader.cache
        def object_uid_name_mapping():
            return load_object_uid_name_mapping(root, meta_dict[get_meta_key('Mapping File Format')])

        cache = {'batches': {}}

    if meta_dict[get_meta_key('Batch File Format')] == 'JSON':
        attach_batch_file_loader_json(SimerseDataLoaderInstance, root)
    elif meta_dict[get_meta_key('Batch File Format')] == 'XML':
        attach_batch_file_loader_xml(SimerseDataLoaderInstance, root)
    elif meta_dict[get_meta_key('Batch File Format')] == 'CSV':
        attach_batch_file_loader_csv(SimerseDataLoaderInstance, root)
    else:
        raise ValueError(f'meta_dict contains an invalid batch file format: {meta_dict["Batch File Format"]}; batch'
                         f' file format must be one of {supported_output_file_formats}')

    dimensions_set = set(SimerseDataLoaderInstance.dimensions)

    # noinspection PyUnusedLocal
    @data_loader.loader
    def load_object_uid(self, points):
        uids = []
        for point in points:
            observation = SimerseDataLoaderInstance.load_observation(point)
            uids.append(array_maker([obj[observation_object_uid_key] for obj in observation[objects_key]]))
        return uids

    load_object_uid.__set_name__(SimerseDataLoaderInstance, observation_object_uid_key)

    if os.path.exists(
            get_batch_number_observation_uid_mapping_file_path(root, meta_dict[get_meta_key('Mapping File Format')])
    ):
        @data_loader.cache
        def batch_number_observation_uid_mapping():
            return load_batch_number_observation_uid_mapping(root, meta_dict[get_meta_key('Mapping File Format')])

        SimerseDataLoaderInstance.batch_number_observation_uid_mapping = batch_number_observation_uid_mapping

        def load_observation(uid):
            batch_number = SimerseDataLoaderInstance.batch_number_observation_uid_mapping.uid_to_batch_number[uid]
            uid_index = SimerseDataLoaderInstance.batch_number_observation_uid_mapping.uid_indices[uid]
            return SimerseDataLoaderInstance.load_batch(batch_number)[observations_key][uid_index]

        SimerseDataLoaderInstance.load_observation = load_observation
    else:
        def load_observation(uid):
            batch_number = uid // SimerseDataLoaderInstance.batch_size
            uid_index = uid % SimerseDataLoaderInstance.batch_size
            return SimerseDataLoaderInstance.load_batch(batch_number)[observations_key][uid_index]

        SimerseDataLoaderInstance.load_observation = load_observation

    import imageio

    def load_capture_name(point, capture_name):
        log(f'Loading capture {capture_name} for observation {point}', 2)
        observation = SimerseDataLoaderInstance.load_observation(point)
        return observation[capture_data_key][capture_name]

    SimerseDataLoaderInstance.load_capture_name = staticmethod(load_capture_name)

    for default_capture_dimension in default_capture_dimensions:
        if default_capture_dimension not in dimensions_set:
            continue

        # noinspection PyUnusedLocal
        @data_loader.loader
        def loader(self, points, capture_dim=default_capture_dimension):
            return image_maker([
                imageio.imread(f'{root}/{load_capture_name(point, capture_dim)}') for point in points
            ])

        loader.__set_name__(SimerseDataLoaderInstance, default_capture_dimension)

    if depth_capture_key in dimensions_set:
        # noinspection PyUnusedLocal
        @data_loader.loader
        def load_depth(self, points):
            return image_maker(
                [
                    process_depth_capture(imageio.imread(f'{root}/{load_capture_name(point, depth_capture_key)}'))
                    for point in points
                ]
            )

        load_depth.__set_name__(SimerseDataLoaderInstance, depth_capture_key)

    for vector_capture_dimension in vector_capture_dimensions:
        if vector_capture_dimension not in dimensions_set:
            continue

        # noinspection PyUnusedLocal
        @data_loader.loader
        def loader(self, points, vector_capture_dim=vector_capture_dimension):
            return image_maker(
                [
                    imageio.imread(f'{root}/{load_capture_name(point, vector_capture_dim)}')[:, :, :3] * 2 - 1
                    for point in points
                ]
            )

        loader.__set_name__(SimerseDataLoaderInstance, vector_capture_dimension)

    if uv_capture_key in dimensions_set:
        # noinspection PyUnusedLocal
        @data_loader.loader
        def uv(self, points):
            return image_maker(
                [
                    imageio.imread(f'{root}/{load_capture_name(point, uv_capture_key)}')[:, :, :3]
                    for point in points
                ]
            )

        uv.__set_name__(SimerseDataLoaderInstance, uv_capture_key)

    if world_position_capture_key in dimensions_set:
        # noinspection PyUnusedLocal
        @data_loader.loader
        def world_position_loader(self, points):
            ret_value = []
            for point in points:
                observation = SimerseDataLoaderInstance.load_observation(point)
                im_path = f"{root}/{observation[capture_data_key][world_position_capture_key]}"
                origin = observation[capture_data_key][world_position_origin_key]
                log(f'Loading capture WorldPosition_Capture for observation {point}')
                ret_value.append(process_world_position_capture(imageio.imread(im_path), origin))
            return image_maker(ret_value)

        world_position_loader.__set_name__(SimerseDataLoaderInstance, world_position_capture_key)

    for integer_capture_dimension in integer_capture_dimensions:
        if integer_capture_dimension not in dimensions_set:
            continue

        # noinspection PyUnusedLocal
        @data_loader.loader
        def loader(self, points, integer_capture_dim=integer_capture_dimension):
            ret_value = []
            for point in points:
                observation = SimerseDataLoaderInstance.load_observation(point)
                im_path = f"{root}/{observation[capture_data_key][integer_capture_dim]}"
                log(f'Loading capture {integer_capture_dim} for observation {point}', 2)
                ret_value.append(process_integer_capture(imageio.imread(im_path)))
            return array_provider.stack(ret_value, 0)

        loader.__set_name__(SimerseDataLoaderInstance, integer_capture_dimension)

    if segmentation_polygon_key in dimensions_set:
        # noinspection PyUnusedLocal
        @data_loader.loader
        def loader(self, points):
            ret_value = []
            for point in points:
                objects = SimerseDataLoaderInstance.load_observation(point)[objects_key]
                ret_value.append(
                    [[array_maker(polygon) for polygon in obj[segmentation_polygon_key]] for obj in objects]
                )
            return ret_value

        loader.__set_name__(SimerseDataLoaderInstance, segmentation_polygon_key)

    if segmentation_rle_key in dimensions_set:
        # noinspection PyUnusedLocal
        @data_loader.loader
        def loader(self, points):
            return [
                [
                    array_maker(obj[segmentation_rle_key])
                    for obj in SimerseDataLoaderInstance.load_observation(point)[objects_key]
                ]
                for point in points
            ]

        loader.__set_name__(SimerseDataLoaderInstance, segmentation_rle_key)

    def safe_array(value):
        return array_maker(value) if value != SimerseDataLoaderInstance.na_value else value

    for bounding_box_dimension in bounding_box_dimensions:
        if bounding_box_dimension not in dimensions_set:
            continue

        # noinspection PyUnusedLocal
        @data_loader.loader
        def loader(self, points, bounding_box_dim=bounding_box_dimension):
            return [
                [
                    safe_array(obj[bounding_box_dim])
                    for obj in SimerseDataLoaderInstance.load_observation(point)[objects_key]
                ]
                for point in points
            ]

        loader.__set_name__(SimerseDataLoaderInstance, bounding_box_dimension)

    if keypoints_key in dimensions_set:
        # noinspection PyUnusedLocal
        @data_loader.loader
        def loader(self, points):
            return [
                [obj[keypoints_key] for obj in SimerseDataLoaderInstance.load_observation(point)[objects_key]]
                for point in points
            ]

        loader.__set_name__(SimerseDataLoaderInstance, keypoints_key)

    if projection_type_key in dimensions_set:
        # noinspection PyUnusedLocal
        @data_loader.loader
        def camera_view_type(self, points):
            return [
                SimerseDataLoaderInstance.load_observation(point)[capture_data_key][projection_type_key]
                for point in points
            ]

        camera_view_type.__set_name__(SimerseDataLoaderInstance, projection_type_key)

    if view_parameter_key in dimensions_set:
        # noinspection PyUnusedLocal
        @data_loader.loader
        def camera_view_parameter(self, points):
            return [
                SimerseDataLoaderInstance.load_observation(point)[capture_data_key][view_parameter_key]
                for point in points
            ]

        camera_view_parameter.__set_name__(SimerseDataLoaderInstance, view_parameter_key)

    if camera_transform_key in dimensions_set:
        # noinspection PyUnusedLocal
        @data_loader.loader
        def camera_transform(self, points):
            return array_maker([
                SimerseDataLoaderInstance.load_observation(point)[capture_data_key][camera_transform_key]
                for point in points
            ])

        camera_transform.__set_name__(SimerseDataLoaderInstance, camera_transform_key)

    if object_transform_key in dimensions_set:
        # noinspection PyUnusedLocal
        @data_loader.loader
        def object_transformation(self, points):
            ret_value = []
            for point in points:
                ret_value.append(array_maker([
                    obj[object_transform_key] for obj in SimerseDataLoaderInstance.load_observation(point)[objects_key]
                ]))
            return ret_value

        object_transformation.__set_name__(SimerseDataLoaderInstance, object_transform_key)

    if time_key in dimensions_set:
        # noinspection PyUnusedLocal
        @data_loader.loader
        def time(self, points):
            return array_maker([SimerseDataLoaderInstance.load_observation(point)[capture_data_key][time_key]
                                for point in points])

        time.__set_name__(SimerseDataLoaderInstance, time_key)

    for name, loader in custom_loaders.items():
        if name not in dimensions_set:
            continue
        loader = data_loader.loader(loader)
        loader.__set_name__(SimerseDataLoaderInstance, name)

    return SimerseDataLoaderInstance()


def attach_batch_file_loader_json(cls, root):
    import json

    batch_cache = cls.cache['batches']

    def load_batch(batch_number):
        if batch_number not in batch_cache:
            log(f'Reading JSON batch file {batch_number}', 2)
            with open(get_batch_file_path(root, batch_number) + '.json', 'r') as f:
                batch_cache[batch_number] = json.load(f)
        return batch_cache[batch_number]

    cls.load_batch = load_batch


def attach_batch_file_loader_xml(cls, root):
    import xml.etree.ElementTree as ElementTree
    import json

    batch_cache = cls.cache['batches']

    def load_batch(batch_number):
        if batch_number not in batch_cache:
            log(f'Reading XML batch file {batch_number}', 2)
            root_element = ElementTree.parse(get_batch_file_path(root, batch_number) + '.xml').getroot()

            log(f'Collecting data from XML batch file {batch_number}', 2)
            batch = []
            for observation in root_element.iterfind(observation_key):
                capture_data = {}
                observation_objects = []
                current_observation = {
                    observation_uid_key: int(observation.find(observation_uid_key).text),
                    capture_data_key: capture_data,
                    objects_key: observation_objects,
                }
                batch.append(current_observation)

                capture_data_element = observation.find(capture_data_key)
                for capture_data_component in capture_data_element:
                    dimension, value = capture_data_component.tag, capture_data_component.text
                    if dimension in json_parse_dimensions:
                        value = json.loads(value)
                    capture_data[dimension] = value

                observation_objects_element = observation.find(objects_key)
                for observation_object in observation_objects_element:
                    object_dict = {}
                    for dimension_element in observation_object:
                        dimension, value = dimension_element.tag, dimension_element.text
                        if dimension in json_parse_dimensions:
                            value = json.loads(value)
                        object_dict[dimension] = value
                    observation_objects.append(object_dict)

            batch_cache[batch_number] = {observations_key: batch}
        return batch_cache[batch_number]

    cls.load_batch = load_batch


def attach_batch_file_loader_csv(cls, root):
    import csv
    import json

    batch_cache = cls.cache['batches']

    def load_batch(batch_number):
        if batch_number not in batch_cache:
            log(f'Reading CSV batch file {batch_number}', 2)
            with open(get_batch_file_path(root, batch_number) + '.csv', 'r') as f:
                raw_csv_data = list(csv.reader(f))

            log(f'Collecting data from CSV batch file {batch_number}', 2)
            batch = {}

            observation_uid_index = raw_csv_data[0].index(observation_uid_key)
            dims_without_observation_uid = (d for d in raw_csv_data[0] if d != observation_uid_key)

            capture_dims = capture_data_dimensions.intersection(dims_without_observation_uid)
            object_dims = per_object_dimensions.intersection(dims_without_observation_uid)

            object_indices = [i for i in range(len(raw_csv_data[0])) if raw_csv_data[0][i] in object_dims]
            capture_indices = [i for i in range(len(raw_csv_data[0])) if raw_csv_data[0][i] in capture_dims]

            for row in raw_csv_data[1:]:
                observation_uid = row[observation_uid_index]
                if observation_uid not in batch:
                    capture_data = {}
                    for i in capture_indices:
                        value = json.loads(row[i]) if raw_csv_data[0][i] in json_parse_dimensions else row[i]
                        capture_data[raw_csv_data[0][i]] = value
                    batch[observation_uid] = {
                        observation_uid_key: observation_uid, capture_data_key: capture_data, objects_key: []
                    }

                current_object_data = {}
                for i in object_indices:
                    current_object_data[raw_csv_data[0][i]] = row[i]

                batch[observation_uid][objects_key].append(current_object_data)

            batch_cache[batch_number] = {observations_key: list(batch.values())}
        return batch_cache[batch_number]

    cls.load_batch = load_batch
