from tools import pivot
import numpy as np

# file = '/home/wouter/Projects/Naturalis/datasets/GBIF_all_ArtsObs_and_iNaturalist_pics_2020-04-07.zip'
file = '/home/wouter/Projects/Naturalis/datasets/GBIF_ALL_bombus_2019-11-28.zip'
#  file = '/home/wouter/Projects/Naturalis/datasets/GBIF_ALL_plantae_2019-11-25.zip'

degrees = 55.0
stepsize = .5
folder = '/home/wouter/Projects/Naturalis/datasets/Bombus_occ'

cells = pivot.get_cells(file)
cells.to_csv(folder + '/cells.csv')

np.savez(folder + '/cells.npz', columns=np.array(cells.columns),
         cells=cells.to_numpy())

layer = dict(np.load(folder + '/cells.npz', allow_pickle=True))
print(layer)
exit(0)

pivot.store_per_latitude(pivot.get_pivot(file), folder, degrees, stepsize)