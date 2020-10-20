
# import simerse visualization objects
from simerse import SimerseDataLoader, SimerseVisualizer, Visualize

# create a data loader
dl = SimerseDataLoader('D:/Training/UE4/Downtown1/meta.txt')

# create a visualizer backed by our data loader
vis = SimerseVisualizer(dl)


# create a filter that selects objects with 'Vehicle' in their name
def vehicle_filter(name):
    return 'Vehicle' in name


# set visualization color of objects filtered by our vehicle_filter to green
green = (0, 255, 0)
vis.set_color(green, object_name_filter=vehicle_filter)


# visualize observation 6 using segmentation visualization mode. By default,
# this will overlay the segmentation onto the visual ldr image. Here we set overlay_alpha=.6
# to set the overlay alpha level to 60%. The visualization will be saved to
# C:/Users/hauck/Pictures/test.png when the window is closed.
vis.visualize(
    6, Visualize.segmentation,
    overlay_alpha=.6,
    save_on_finish='C:/Users/hauck/Pictures/test.png'
)

# visualize observation 6 using segmentation visualization mode. Here we
# set the mode to 'raw' instead of 'overlay', which will just show the raw
# segmentation (not an overlay onto the visual ldr image)
vis.visualize(
    6, Visualize.segmentation, mode='raw',
)

# visualize observation 6 using segmentation visualization mode. Here we
# filter the vehicles using our vehicle_filter and show the raw segmentation.
vis.visualize(
    6, Visualize.segmentation, mode='raw',
    object_name_filter=vehicle_filter
)

# and the same thing as an overlay
vis.visualize(
    6, Visualize.segmentation, mode='overlay',
    object_name_filter=vehicle_filter
)
