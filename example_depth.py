
# import simerse visualization objects
from simerse import SimerseDataLoader, SimerseVisualizer, Visualize

# create a data loader
dl = SimerseDataLoader('D:/Training/UE4/Downtown1/meta.txt')

# create a visualizer backed by our data loader
vis = SimerseVisualizer(dl)

# visualize observation 6 using depth visualization mode.
# The visualization will be saved to C:/Users/hauck/Pictures/test.png as a
# grayscale image
vis.visualize(
    6, Visualize.depth,
    save_on_finish='C:/Users/hauck/Pictures/test.png'
)
