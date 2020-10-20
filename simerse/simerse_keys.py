
from enum import Enum, unique


@unique
class Visualize(Enum):
    visual_ldr = 0              # ✓
    visual_ldr_video = 16
    visual_hdr = 1              # ✓
    visual_hdr_video = 17
    segmentation = 2            # ✓
    segmentation_video = 18
    bounding_box_2d = 3         # ✓
    bounding_box_2d_video = 19
    bounding_box_3d = 4
    bounding_box_3d_video = 20
    keypoints = 5
    keypoints_video = 21
    depth = 6                   # ✓
    depth_video = 22
    uv = 8                      # ✓
    uv_video = 23
    position = 9
    position_video = 24
    normal = 10                 # ✓
    normal_video = 25
    tangent = 11                # ✓
    tangent_video = 26
    bitangent = 12              # ✓
    bitangent_video = 27
    custom = 15


class BuiltinDimension:
    object_uid = 'ObservationObjectUID'
    visual_ldr = 'VisualLDR_Capture'
    visual_hdr = 'VisualHDR_Capture'
    segmentation = 'Segmentation_Capture'
    segmentation_rle = 'Segmentation_RLE'
    segmentation_outline = 'SegmentationOutline_Capture'
    segmentation_polygon = 'SegmentationOutline_Polygon'
    bounding_box_2d_total = 'ImageAligned2DTotalBoundingBox'
    bounding_box_2d_contiguous = 'ImageAligned2DContiguousBoundingBox'
    bounding_box_3d_local = 'LocallyAligned3DBoundingBox'
    bounding_box_3d_global = 'GloballyAligned3DBoundingBox'
    bounding_box_3d_custom = 'Custom3DBoundingBox'
    keypoints = 'Keypoints'
    depth = 'Depth_Capture'
    camera_transform = 'CameraTransform'
    camera_projection = 'CameraProjectionType'
    camera_view = 'CameraViewParameter'
    uv = 'UV_Capture'
    world_position = 'WorldPosition_Capture'
    world_normal = 'WorldNormal_Capture'
    world_tangent = 'WorldTangent_Capture'
    world_bitangent = 'WorldBitangent_Capture'
    object_transform = 'ObjectTransformation'
    time = 'Time'
    world_origin = 'WorldPositionOrigin'
