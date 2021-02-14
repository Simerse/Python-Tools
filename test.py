
from simerse import simerse_data_loader
from simerse.simerse_keys import BuiltinDimension, Visualize
from simerse import logtools
from simerse import simerse_visualize


card_suits = (
    'Clubs',
    'Diamonds',
    'Hearts',
    'Spades'
)

card_values = (
    '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'
)


def get_card_name(suit, value):
    return card_values[int(value)] + ' of ' + card_suits[int(suit)]


class CardNameMap:
    def __init__(self, uids, values, suits):
        self.mapping = {int(uid): get_card_name(suit, value) for uid, suit, value in zip(uids[None], suits, values)}

    def get(self, uid, default_value):
        return self.mapping.get(uid, default_value)


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

    vis_uid = 5

    vis = simerse_visualize.SimerseVisualizer(dl)

    name_map = CardNameMap(*dl.load(vis_uid, (BuiltinDimension.object_uid, 'CardValue', 'CardSuit')))

    vis.visualize(vis_uid, Visualize.bounding_box_2d, line_thickness=4, show_names=name_map)
