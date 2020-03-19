import matplotlib.pyplot as plt
from datetime import datetime

import get_environmental_layer as get_env

maps = get_env.get_blocks([(63.3, 10, datetime(2012, 1, 6))], 25, 'esacci',
                          '/home/wouter/Projects/Naturalis/environment/cached_layers')

for position in maps:
    for name, map in position.items():
        plt.imshow(map)
        plt.title(name)
        plt.show()
