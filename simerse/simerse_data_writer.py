
import os

import simerse.data_writer as data_writer
import simerse.logtools as logtools
from simerse.box_format import BoxFormat


mebibytes = 2 ** 20


def load_meta(meta, explicit_folder, logger):
    if isinstance(meta, dict):
        meta_dict = meta
        root = explicit_folder
        if root is None:
            logger('', ValueError('Must provide an explicit dataset folder through the "explicit_folder" argument'
                                  ' for datasets with an explicit meta dict.'))
    else:
        import json
        with open(meta, 'r') as f:
            meta_dict = json.load(f)
        root = os.path.dirname(meta)

    return meta_dict, root


# noinspection PyPep8Naming
def SimerseDataWriter(
    meta, explicit_folder=None,
    logger=logtools.default_logger,
    na_value='N/A',
    cache_limit=256 * mebibytes,
    bounding_box_2d_write_format=BoxFormat.min_max, bounding_box_3d_write_format=BoxFormat.min_max,
    **custom_loaders
):

    meta_dict, root = load_meta(meta, explicit_folder, logger)

    class SimerseDataWriterInstance(data_writer.DataWriter):
        _logger = logger

        @property
        def na_value(self):
            return na_value

        batch_cache_limit = cache_limit

        bounding_box_2d_format = bounding_box_2d_write_format
        bounding_box_3d_format = bounding_box_3d_write_format
