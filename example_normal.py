
# import simerse visualization objects
from simerse import SimerseDataLoader, SimerseVisualizer, Visualize

# create a data loader
dl = SimerseDataLoader('D:/Training/UE4/Downtown1/meta.txt')

# create a visualizer backed by our data loader
vis = SimerseVisualizer(dl)

# visualize observation 6 using world normal visualization mode. Note that
# a normal vector with components (x, y, z) is encoded to color via (r, g, b) = 2 * (x, y, z) - 1,
# so negative values correspond to darker colors and positive values to brighter ones; x values correspond
# to red, y values to green, and z values to blue.
# The visualization will be saved to C:/Users/hauck/Pictures/test.png
# when the window closes.
vis.visualize(
    6, Visualize.normal,
    save_on_finish='C:/Users/hauck/Pictures/test.png'
)

# visualize observation 6 using world normal visualization mode. Here
# we show each component channel separately so that matplotlib doesn't clip
# the negative components. The visualization will be saved to three files
# C:/Users/hauck/Pictures/test_x.png (the x component will be saved here, encoded as gray = 2 * x - 1)
# C:/Users/hauck/Pictures/test_y.png (the y component will be saved here, encoded as gray = 2 * y - 1)
# C:/Users/hauck/Pictures/test_z.png (the z component will be saved here, encoded as gray = 2 * z - 1)
vis.visualize(
    6, Visualize.normal,
    separate_channels=True,
    save_on_finish='C:/Users/hauck/Pictures/test.png'
)
