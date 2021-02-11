
from enum import Enum, unique

# This is a checkmark
# ✓


@unique
class Visualize(Enum):
    visual_ldr = 0              #
    visual_hdr = 1              #
    segmentation = 2            #
    bounding_box_2d = 3         #
    bounding_box_3d = 4         #
    keypoints = 5               #
    depth = 6                   #
    uv = 8                      #
    position = 9                #
    normal = 10                 #
    tangent = 11                #
    bitangent = 12              #
    custom = 15                 #


class BuiltinDimension:
    object_uid = 'ObservationObjectUID'

    world_tangent = 'WorldTangent'
    world_bitangent = 'WorldBitangent'
    world_normal = 'WorldNormal'

    visual_ldr = 'VisualLDR'
    visual_hdr = 'VisualHDR'

    segmentation = 'Segmentation'
    segmentation_outline = 'SegmentationOutline'

    keypoint = 'Keypoint'

    uv = 'UV'

    position = 'Position'

    depth = 'Depth'

    local_bounding_box_3d = 'LocallyBoundingBox3D'
    global_bounding_box_3d = 'GlobalBoundingBox3D'
    custom_bounding_box_3d = 'CustomBoundingBox3D'

    total_bounding_box_2d = 'TotalBoundingBox2D'
    connected_bounding_box_2d = 'ConnectedBoundingBox2D'

    time = 'ObservationTime'
    object_transform = 'ObjectGlobalTransform'
    mesh_name = 'StaticMeshName'
    unreal_name = 'UnrealObjectName'
    camera_view = 'CameraView'

    _inverse_mapping = None

    @staticmethod
    def get_standard_name(dimension_name):
        if BuiltinDimension._inverse_mapping is None:
            BuiltinDimension._inverse_mapping = {}
            for dimension, value in BuiltinDimension.__dict__:
                if dimension != '_inverse_mapping':
                    BuiltinDimension._inverse_mapping[value] = dimension
        return BuiltinDimension._inverse_mapping.get(dimension_name, dimension_name)


class MetaKey:
    batch_file_format = 'Batch File Format'
    dataset_name = 'Dataset Name'
    description = 'Description'
    summary = 'Summary'
    dimensions = 'Dimensions'
    observation_count = 'Total Observations'
    batch_size = 'Observation Batch Size'
    license = 'License'
    start_time = 'Generation Start Time'
    end_time = 'Generation End Time'

    defaults = {
        batch_file_format: 'Detect',
        dataset_name: 'Unknown--Dataset Incomplete',
        description: 'An incomplete dataset',
        license: 'No license',
        start_time: '',
        end_time: ''
    }
    summary_defaults = {
        dimensions: [],
        observation_count: 1,
        batch_size: 1
    }

    dataset_folder = 'Dataset Folder'


class BatchKey:
    batch_uid = 'ObservationBatchUID'
    observation_uid = 'ObservationUID'
    object_uid = 'ObservationObjectUID'
    xml_batch = 'ObservationBatch'
    xml_observation = 'Observation'
    observations = 'Observations'
    per_observer_values = 'PerObserverValues'
    per_observation_values = 'PerObservationValues'
    object_values = 'ObservationObjectValues'
    observer_name = 'ObserverName'
    xml_object_value = 'ObservationObjectValue'
    xml_array_element = 'AE'
    xml_map_pair = 'MP'
    xml_map_key = 'K'
    xml_map_value = 'V'
    xml_dimension_value = 'DimensionValue'
    xml_dimension_name = 'Name'
    xml_observer_value = 'Observer'
    xml_observer_name = 'Name'
    csv_observation_uid_col = 0
    csv_observer_col = 1
    csv_object_uid_col = 2
    csv_first_dimension_col_index = 3
    uri = 'URI'
    resolution = 'Resolution'
    position_capture = 'Capture'
    position_coordinates_origin = 'Position Coordinates Origin'
    actor_bounding_box = 'ActorBoundingBox'
    object_bounding_box = 'ObjectBoundingBox'
    camera_transform_matrix = 'Global Transform'
    spline_object_transform = 'Global Transform'
