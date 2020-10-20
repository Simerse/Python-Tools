
from simerse import SimerseDataLoader, SimerseVisualizer, Visualize, BuiltinDimension

dl = SimerseDataLoader('D:/Training/UE4/VisualizeTest/meta.txt')

vis = SimerseVisualizer(dl)

vis.visualize(6, Visualize.position)
