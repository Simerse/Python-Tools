
from simerse import simerse_data_loader
from simerse.simerse_keys import BuiltinDimension
from simerse import logtools


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

    camera_view, card_value, uids, bb3d = dl.load(range(5), (
        BuiltinDimension.camera_view, 'CardValue', BuiltinDimension.object_uid,
        BuiltinDimension.local_bounding_box_3d
    ))
    print(uids)
    print('\n')
    print(card_value)
    print('\n')
    print(bb3d)

