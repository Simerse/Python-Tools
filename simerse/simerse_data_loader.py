
from collections import namedtuple, deque
import os

from simerse import image_util, data_loader, logtools, iotools

try:
    import torch as array_provider
    array_maker = array_provider.tensor
    image_maker = image_util.to_torch
except ImportError:
    import numpy as array_provider
    array_maker = array_provider.array
    image_maker = image_util.to_numpy

from simerse.box_format import BoxFormat

from simerse.simerse_keys import BuiltinDimension, MetaKey, BatchKey


# ==== Configuration stuff ====

na_value = 'N/A'

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
    return os.path.join(root, 'Batches', f'BatchFile_{batch_uid}')


ImageResolution = namedtuple('ImageResolution', ('width', 'height'))
ImageCoordinatesPoint = namedtuple('ImageCoordinatesPoint', ('x', 'y'))
SimerseLoaderCache = namedtuple('SimerseLoaderCache', ('batches', 'batch_sizes', 'batch_queue', 'total_cache_size'))

box_format_mapping = {
    'MIN_MAX': BoxFormat.min_max,
    'MIN_EXTENTS': BoxFormat.min_extents,
    'CENTER_EXTENTS': BoxFormat.center_extents,
    'CENTER_HALF_EXTENTS': BoxFormat.center_half
}


class CacheLimitExceededWarning(Warning):
    pass

# ==== End configuration stuff ====


def get_box_format(meta_format):
    global box_format_mapping

    return box_format_mapping[meta_format]


def parse_list_of_integers(list_string):
    return list(map(int, list_string.strip('][').split(',')))


def parse_list_of_floats(list_string):
    return list(map(int, list_string.strip('][').split(',')))


def parse_list_of_strings(list_string):
    return list(map(lambda s: s.strip('"'), list_string.strip('][').split(',')))


def process_depth_capture(raw):
    return array_maker([raw[:, :, 0]])


def process_world_position_capture(raw, origin):
    import numpy as np

    origin = np.array(origin)
    return raw[:, :, :3] + origin.reshape((1, 1, 3))


def process_integer_capture(raw):
    import numpy as np

    raw = raw.astype(np.uint32)
    return array_maker((raw[:, :, 0] | (raw[:, :, 1] << 8) | (raw[:, :, 2] << 16)).astype(np.int32))


def load_meta(meta, logger):
    if isinstance(meta, dict):

        for meta_key, value in MetaKey.defaults.items():
            meta.setdefault(meta_key, value)
        summary = meta.setdefault(MetaKey.summary, {})

        for meta_key, value in MetaKey.summary_defaults.items():
            summary.setdefault(meta_key, value)

        try:
            return meta, meta[MetaKey.dataset_name]
        except KeyError:
            logger('Explicit dataset meta must contain MetaKey.dataset_name', KeyError)

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
            logger(f'Failed to read meta file {meta_file}', e)

        logger(f'Read meta file {meta_file}')

        return meta, os.path.dirname(meta_file)


def attach_load_batch_function(meta_dict, clazz, root, logger):
    if meta_dict[MetaKey.batch_file_format] == 'JSON':
        attach_batch_file_loader_json(clazz, root, logger)
    elif meta_dict[MetaKey.batch_file_format] == 'XML':
        attach_batch_file_loader_xml(clazz, root, logger)
    elif meta_dict[MetaKey.batch_file_format] == 'CSV':
        attach_batch_file_loader_csv(clazz, root, logger)
    else:
        logger('', ValueError(
            f'meta_dict contains an invalid batch file format: {meta_dict["Batch File Format"]}; batch'
            f' file format must be one of {supported_output_file_formats}'
        ))


# noinspection PyPep8Naming
Mebibytes = 2 ** 20


# noinspection PyPep8Naming
def SimerseDataLoader(meta, logger=logtools.default_logger, cache_limit=256 * Mebibytes):

    meta_dict, root = load_meta(meta, logger)

    '''
    Declare class for the dataset
    '''
    # noinspection PyMethodParameters
    class SimerseDataLoaderInstance(data_loader.DataLoader):

        batch_cache_limit = cache_limit

        name = meta_dict[MetaKey.dataset_name]

        description = meta_dict[MetaKey.description]

        dimensions = meta_dict[MetaKey.summary][MetaKey.dimensions] + [BuiltinDimension.object_uid]

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

        cache = SimerseLoaderCache(batches={}, batch_queue=deque(), batch_sizes={}, total_cache_size=0)

        @staticmethod
        def load_observation(uid):
            return SimerseDataLoaderInstance.load_batch(
                uid // SimerseDataLoaderInstance.batch_size
            )[BatchKey.observations][uid % SimerseDataLoaderInstance.batch_size]

    # attach the appropriate batch loader for the dataset
    attach_load_batch_function(meta_dict, SimerseDataLoaderInstance, root, logger)

    '''
    Attach load_object_uid function to the     
    '''
    # noinspection PyUnusedLocal
    @data_loader.loader
    def load_object_uid(self, points):
        uids = []
        for point in points:
            observation = SimerseDataLoaderInstance.load_observation(point)
            uids.append(array_maker([obj[observation_object_uid_key] for obj in observation[objects_key]]))
        return uids

    load_object_uid.__set_name__(SimerseDataLoaderInstance, observation_object_uid_key)

    import imageio

    def load_capture_name(point, capture_name):
        log(f'Loading capture {capture_name} for observation {point}', 2)
        observation = SimerseDataLoaderInstance.load_observation(point)
        return observation[capture_data_key][capture_name]

    SimerseDataLoaderInstance.load_capture_name = staticmethod(load_capture_name)

    dimensions_set = set(SimerseDataLoaderInstance.dimensions)

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


def safe_add_batch(batch_number, cache, data, cache_limit, logger):
    logger(f'Beginning analysis of loaded batch size')
    cache.batch_sizes[batch_number] = iotools.get_size_recursive(data)
    cache.total_cache_size += cache.batche_sizes[batch_number]

    if cache.total_cache_size >= cache_limit:
        if len(cache.batch_queue) == 1:
            logger(f'Batch cache limit of {cache_limit // 2 ** 20} MiB exceeded.',
                   ValueError(f'Batch cache limit {cache_limit // 2 ** 20} MiB is too small'
                              f'to store even one batch. Please increase it and try again!'))
        else:
            logger(f'Batch cache limit of {cache_limit // 2 ** 20} MiB exceeded. '
                   f'Attempting to remove oldest batch data to make space', CacheLimitExceededWarning)

        while len(cache.batch_queue) > 1 and cache.total_cache_size >= cache_limit:
            last_batch = cache.batch_queue.popleft()
            cache.batches.pop(last_batch)
            cached_size = cache.batch_sizes.pop(last_batch)
            cache.total_cache_size -= cached_size
            logger(f'Cached batch {last_batch} removed, freeing {cached_size} bytes')

        if cache.total_cache_size >= cache_limit:
            logger(f'Batch cache limit of {cache_limit // 2 ** 20} MiB exceeded.',
                   ValueError(f'Batch cache limit {cache_limit // 2 ** 20} MiB is too small'
                              f'to store even one batch. Please increase it and try again!'))

    logger('Finished analyzing loaded batch size')


def attach_batch_file_loader_json(cls, root, logger):
    import json

    cache = cls.cache

    def load_batch(batch_number):
        if batch_number not in cache.batches:
            try:
                logger(f'Reading JSON batch file {batch_number}')
                with open(get_batch_file_path(root, batch_number) + '.json', 'r') as f:
                    data = json.load(f)

                    cache.batches[batch_number] = data
                    cache.batch_queue.append(batch_number)

                    safe_add_batch(batch_number, cache, data, cls.batch_cache_limit, logger)

            except IOError as e:
                logger(f'Failed to read JSON batch file {batch_number}', e)
            except json.decoder.JSONDecodeError as e:
                logger(f'Failed to parse JSON batch file {batch_number}', e)
            except Exception as e:
                logger(f'Failed to parse JSON batch file {batch_number}', e)

        return cache.batches[batch_number]

    cls.load_batch = load_batch


def parse_xml_dimension_value_recursive(node):
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
            return [parse_xml_dimension_value_recursive(array_element) for array_element in node]
        elif node.find(BatchKey.xml_map_pair) is not None:
            return {
                parse_xml_dimension_value_recursive(map_pair.find(BatchKey.xml_map_key)):
                    parse_xml_dimension_value_recursive(map_pair.find(BatchKey.xml_map_value))
                for map_pair in node
            }
        else:
            return na_value


def parse_xml_observation_value(node):
    final_value = {}
    for value in node.iterfind(BatchKey.xml_dimension_value):
        final_value[value.attrib[BatchKey.xml_dimension_name]] = parse_xml_dimension_value_recursive(value)

    final_value[BatchKey.object_values] = [
        parse_xml_dimension_value_recursive(object_value) for object_value in
        node.find(BatchKey.object_values)
    ]

    return final_value


def attach_batch_file_loader_xml(cls, root, logger):
    import xml.etree.ElementTree as ElementTree

    cache = cls.cache

    def load_batch(batch_number):
        if batch_number not in cache.batches:
            logger(f'Reading XML batch file {batch_number}')
            try:
                root_element = ElementTree.parse(get_batch_file_path(root, batch_number) + '.xml').getroot()

                logger(f'Collecting data from XML batch file {batch_number}')
                batch = []
                for observation in root_element.iterfind(BatchKey.xml_observation):
                    per_observer_values = {}

                    current_observation = {
                        BatchKey.observation_uid: int(observation.find(BatchKey.observation_uid).text),
                        BatchKey.per_observation_values:
                            parse_xml_observation_value(observation.find(BatchKey.per_observation_values)),
                        BatchKey.per_observer_values: per_observer_values
                    }
                    batch.append(current_observation)

                    for observer_value in observation.find(BatchKey.per_observer_values):
                        per_observer_values[observer_value.attrib[BatchKey.xml_observer_name]] = \
                            parse_xml_observation_value(observer_value)

                data = {BatchKey.batch_uid: batch_number, BatchKey.observations: batch}

                cache.batches[batch_number] = data
                cache.batch_queue.append(batch_number)

                safe_add_batch(batch_number, cache, data, cls.batch_cache_limit, logger)

            except IOError as e:
                logger(f'Failed to read XML batch file {batch_number}', e)
            except ElementTree.ParseError as e:
                logger(f'Failed to parse XML batch file {batch_number}', e)
            except Exception as e:
                logger(f'Failed to parse XML batch file {batch_number}', e)

        return cache.batches[batch_number]

    cls.load_batch = load_batch


def attach_batch_file_loader_csv(cls, root, logger):
    import csv
    import json

    cache = cls.cache

    def load_batch(batch_number):
        if batch_number not in cache.batches:
            logger(f'Reading CSV batch file {batch_number}')
            try:
                with open(get_batch_file_path(root, batch_number) + '.csv', 'r') as f:
                    raw_csv_data = list(csv.reader(f))

                logger(f'Collecting data from CSV batch file {batch_number}')
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
                logger(f'Failed to read CSV batch file {batch_number}', e)
            except Exception as e:
                logger(f'Failed to parse CSV batch file {batch_number}', e)

        return cache.batches[batch_number]

    cls.load_batch = load_batch
