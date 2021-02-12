
from simerse import simerse_data_loader
from simerse.simerse_keys import BuiltinDimension
from simerse import logtools
import matplotlib.pyplot as plt
import simerse.image_util


def card_value_loader_builder(cls):
    return simerse_data_loader.build_per_observation_array_loader(
        'CardValue', jagged_by_observation=True
    )


def card_suit_loader_builder(cls):
    return simerse_data_loader.build_per_observation_array_loader(
        'CardSuit', jagged_by_observation=True
    )


with logtools.default_logger.verbosity_temp(logtools.LogVerbosity.EVERYTHING):
    dl = simerse_data_loader.SimerseDataLoader(
        'D:/Projects/UE4/SimersePlayingCards/SimerseObserveOutput/Dataset',
        CardValue=card_value_loader_builder, CardSuit=card_suit_loader_builder
    )

    image, camera_view, card_value, card_suit, uids, bb3d, bb2d = dl.load(0, (
        BuiltinDimension.visual_ldr,
        BuiltinDimension.camera_view,
        'CardValue', 'CardSuit',
        BuiltinDimension.object_uid,
        BuiltinDimension.local_bounding_box_3d,
        BuiltinDimension.total_bounding_box_2d
    ))
    print(uids)
    print()
    print(card_value)
    print()
    print(bb2d)
    plt.imshow(simerse.image_util.to_numpy(image))
    plt.show()
