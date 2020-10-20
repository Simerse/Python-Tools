
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

# visualize observation 6 (this corresponds to VisualLDR_Capture_6.png)
# using the bounding_box_2d visualization mode, with objects filtered by our
# vehicle_filter, object names hidden, and bounding box line thickness of 4 pixels.
# The visualization will be saved to C:/Users/hauck/Pictures/test.png when the window is closed.
vis.visualize(
    6, Visualize.bounding_box_2d,
    object_name_filter=vehicle_filter,
    show_names=False, line_thickness=4,
    save_on_finish='C:/Users/hauck/Pictures/test.png'
)

# visualize observation 6 using bounding_box_2d visualization mode, showing all objects
vis.visualize(
    6, Visualize.bounding_box_2d,
    show_names=False, line_thickness=4
)
