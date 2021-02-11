
from simerse import simerse_data_loader
from simerse import simerse_visualize
from simerse import simerse_keys


dl = simerse_data_loader.SimerseDataLoader('D:/Projects/UE4/NewPluginTest/'
                                           'DefaultSimerseOutput/DefaultSimerseDatasetMeta.txt')

vis = simerse_visualize.SimerseVisualizer(dl)

vis.visualize(3, simerse_keys.Visualize.segmentation, mode='raw')

