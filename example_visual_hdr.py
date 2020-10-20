
# import simerse visualization objects
from simerse import SimerseDataLoader, SimerseVisualizer, Visualize
# Reinhard tone mapping operator
from simerse import Reinhard

# create a data loader
dl = SimerseDataLoader('D:/Training/UE4/Downtown1/meta.txt')

# create a visualizer backed by our data loader
vis = SimerseVisualizer(dl)

# visualize observation 6 using visual hdr visualization mode,
# and apply reinhard tone mapping. The visualization will be
# saved to C:/Users/hauck/Pictures/test.png when the window is closed.
vis.visualize(
    6, Visualize.visual_hdr,
    # white_point_quantile is a number between 0 and 1. higher quantile=darker image, lower quantile=brighter image
    mapping=Reinhard(white_point_quantile=0.88),
    save_on_finish='C:/Users/hauck/Pictures/test.png'
)
