from KDEpy import FFTKDE
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Create 2D data of shape (obs, dims)
# data = np.random.randn(2 ** 8, 2)

data = np.array([[63.22781, 10.448906],
                 [63.39833, 10.475869],
                 [63.383062, 10.142838],
                 [63.336437, 10.252575],
                 [63.321634, 10.273659],
                 [63.288643, 10.468861],
                 [63.221194, 10.535841],
                 [63.323943, 10.432559],
                 [63.284281, 10.461251],
                 [63.237375, 10.461179],
                 [63.320712, 10.425381],
                 [63.295731, 10.448884],
                 [63.412998, 10.717],
                 [63.45205, 10.906688],
                 [63.429144, 10.378533],
                 [63.428412, 10.379575],
                 [63.34213, 10.217501],
                 [63.382877, 10.400739],
                 [63.466557, 10.88291],
                 [63.464434, 10.902906],
                 [63.399273, 10.33253],
                 [63.466557, 10.88291],
                 [63.428412, 10.379575],
                 [63.466557, 10.88291],
                 [63.44624, 10.88282],
                 [63.460352, 10.90791],
                 [63.473819, 10.863569],
                 [63.475103, 10.808907],
                 [63.429047, 10.379674],
                 [63.423076, 10.395286],
                 [63.471749, 10.817243],
                 [63.472118, 10.799097],
                 [63.435857, 10.695414],
                 [63.334109, 10.203631],
                 [63.342191, 10.219298],
                 [63.342529, 10.245466],
                 [63.442256, 10.1835],
                 [63.339502, 10.211536],
                 [63.428469, 10.375748],
                 [63.47591, 10.805781],
                 [63.44624, 10.88282],
                 [63.44624, 10.88282],
                 [63.318793, 10.827691],
                 [63.331807, 10.8178],
                 [63.284317, 10.467816],
                 [63.445826, 10.865411],
                 [63.477955, 10.7898],
                 [63.383054, 10.395098],
                 [63.436874, 10.503713],
                 [63.383006, 10.396311],
                 [63.477586, 10.900813],
                 [63.44624, 10.88282],
                 [63.436874, 10.503713],
                 [63.431764, 10.524195],
                 [63.438316, 10.705277],
                 [63.472118, 10.799097],
                 [63.436874, 10.503713],
                 [63.338458, 10.255545],
                 [63.440979, 10.177688],
                 [63.342179, 10.218902],
                 [63.473454, 10.897795],
                 [63.46445, 10.909734],
                 [63.434363, 10.719165],
                 [63.434363, 10.719165],
                 [63.338978, 10.442188],
                 [63.435978, 10.736485],
                 [63.435978, 10.736485],
                 [63.434363, 10.719165],
                 [63.434363, 10.719165],
                 [63.435978, 10.736485],
                 [63.244813, 10.166827],
                 [63.477955, 10.7898],
                 [63.342191, 10.219298],
                 [63.434363, 10.719165],
                 [63.420081, 10.804083],
                 [63.434363, 10.719165],
                 [63.435978, 10.736485],
                 [63.472118, 10.799097],
                 [63.477955, 10.7898],
                 [63.342529, 10.245466],
                 [63.435978, 10.736485],
                 [63.434363, 10.719165],
                 [63.434363, 10.719165],
                 [63.376856, 10.395296],
                 [63.477955, 10.7898],
                 [63.472118, 10.799097],
                 [63.447421, 10.895181],
                 [63.374508, 10.406093],
                 [63.34531, 10.215776],
                 [63.342529, 10.245466],
                 [63.435978, 10.736485],
                 [63.435978, 10.736485],
                 [63.434363, 10.719165],
                 [63.4347, 10.727142],
                 [63.420045, 10.790985],
                 [63.420045, 10.790985],
                 [63.438942, 10.831149],
                 [63.428907, 10.353191],
                 [63.472118, 10.799097],
                 [63.471998, 10.903428],
                 [63.442413, 10.663317],
                 [63.471749, 10.817243],
                 [63.470277, 10.832065],
                 [63.435978, 10.736485],
                 [63.442108, 10.628606],
                 [63.435978, 10.736485],
                 [63.442413, 10.663317],
                 [63.442413, 10.663317],
                 [63.442413, 10.663317],
                 [63.435978, 10.736485],
                 [63.438316, 10.705277],
                 [63.434363, 10.719165],
                 [63.423132, 10.39428],
                 [63.338458, 10.255545],
                 [63.450171, 10.910165],
                 [63.405833, 10.534274],
                 [63.472953, 10.89899],
                 [63.420045, 10.790985],
                 [63.435978, 10.736485],
                 [63.442931, 10.616991],
                 [63.4347, 10.727142],
                 [63.472953, 10.89899],
                 [63.450171, 10.910165],
                 [63.347079, 10.588343],
                 [63.464434, 10.902906],
                 [63.302016, 10.281497],
                 [63.437573, 10.576234],
                 [63.437239, 10.391747],
                 [63.4347, 10.727142],
                 [63.429501, 10.79395],
                 [63.437239, 10.391747],
                 [63.472118, 10.799097],
                 [63.473819, 10.863569],
                 [63.431619, 10.380303],
                 [63.383054, 10.395098],
                 [63.448332, 10.731454],
                 [63.439372, 10.474078],
                 [63.438597, 10.474006],
                 [63.330912, 10.259021],
                 [63.459943, 10.885713],
                 [63.435978, 10.736485],
                 [63.432917, 10.377796],
                 [63.360427, 10.2668],
                 [63.457286, 10.444909],
                 [63.435978, 10.736485],
                 [63.429373, 10.380734],
                 [63.434749, 10.727295],
                 [63.440718, 10.472721],
                 [63.43836, 10.568545],
                 [63.43879, 10.504243],
                 [63.448103, 10.466218],
                 [63.341292, 10.226107],
                 [63.420045, 10.790985],
                 [63.4347, 10.727142],
                 [63.35884, 10.425874],
                 [63.439095, 10.398637],
                 [63.424402, 10.803337],
                 [63.429469, 10.363944],
                 [63.471749, 10.817243],
                 [63.470277, 10.832065],
                 [63.413874, 10.794192],
                 [63.434749, 10.727295],
                 [63.47895, 10.790788],
                 [63.395174, 10.532908],
                 [63.477955, 10.7898],
                 [63.43055, 10.557801],
                 [63.383195, 10.393912],
                 [63.343106, 10.207],
                 [63.435548, 10.738245],
                 [63.477955, 10.7898],
                 [63.343722, 10.191333],
                 [63.434363, 10.719165],
                 [63.477955, 10.7898],
                 [63.372858, 10.409192],
                 [63.435351, 10.807604],
                 [63.332077, 10.259686],
                 [63.330968, 10.184614],
                 [63.427082, 10.371625],
                 [63.468154, 10.889117],
                 [63.435548, 10.738245],
                 [63.439521, 10.47397],
                 [63.443349, 10.432297],
                 [63.431731, 10.379907],
                 [63.43368, 10.384956],
                 [63.338656, 10.207359],
                 [63.439428, 10.474284],
                 [63.439529, 10.624644],
                 [63.330968, 10.184614],
                 [63.431856, 10.384758],
                 [63.4347, 10.727142],
                 [63.431619, 10.380303],
                 [63.434134, 10.735757],
                 [63.431619, 10.380303],
                 [63.437645, 10.725534],
                 [63.435978, 10.736485],
                 [63.335342, 10.189482],
                 [63.441377, 10.49155],
                 [63.473819, 10.863569],
                 [63.470277, 10.832065],
                 [63.435946, 10.726334],
                 [63.4347, 10.727142],
                 [63.439432, 10.473521],
                 [63.341723, 10.211833],
                 [63.342243, 10.218839],
                 [63.478978, 10.790869],
                 [63.478978, 10.790869],
                 [63.478978, 10.790869],
                 [63.314908, 10.177993],
                 [63.454512, 10.459543],
                 [63.472118, 10.799097],
                 [63.343722, 10.191333],
                 [63.477955, 10.7898],
                 [63.473819, 10.863569],
                 [63.342243, 10.218839],
                 [63.314912, 10.172603],
                 [63.472118, 10.799097],
                 [63.341723, 10.211833],
                 [63.441461, 10.506471],
                 [63.341723, 10.211833],
                 [63.380748, 10.398655],
                 [63.4347, 10.727142],
                 [63.342872, 10.216279],
                 [63.473254, 10.897768],
                 [63.428895, 10.376063],
                 [63.413874, 10.794192],
                 [63.441565, 10.48855],
                 [63.472118, 10.799097],
                 [63.472953, 10.89899],
                 [63.436572, 10.498925],
                 [63.345189, 10.215417],
                 [63.429501, 10.79395],
                 [63.432772, 10.557082],
                 [63.438946, 10.504665],
                 [63.336842, 10.261932],
                 [63.336842, 10.261932],
                 [63.438685, 10.469002],
                 [63.471854, 10.90076],
                 [63.444349, 10.476854],
                 [63.472744, 10.900319],
                 [63.319321, 10.279916],
                 [63.439702, 10.472281],
                 [63.284317, 10.467816],
                 [63.422204, 10.110341],
                 [63.434363, 10.719165],
                 [63.435978, 10.736485],
                 [63.435978, 10.736485],
                 [63.4347, 10.727142],
                 [63.435978, 10.736485],
                 [63.4347, 10.727142],
                 [63.4347, 10.727142],
                 [63.434749, 10.727295],
                 [63.438706, 10.815905],
                 [63.437645, 10.725534],
                 [63.431619, 10.380303],
                 [63.435548, 10.738245],
                 [63.382877, 10.400739],
                 [63.378224, 10.401485],
                 [63.383054, 10.395098],
                 [63.345471, 10.216558],
                 [63.424177, 10.398512],
                 [63.370543, 10.901083],
                 [63.466557, 10.88291],
                 [63.464434, 10.902906],
                 [63.477955, 10.7898],
                 [63.475103, 10.808907],
                 [63.477955, 10.7898],
                 [63.477955, 10.7898],
                 [63.475103, 10.808907],
                 [63.472953, 10.89899],
                 [63.433563, 10.739683],
                 [63.429148, 10.38006],
                 [63.429148, 10.38006],
                 [63.330597, 10.258276],
                 [63.330879, 10.254673],
                 [63.432519, 10.355994],
                 [63.426375, 10.401081],
                 [63.370684, 10.241324],
                 [63.370684, 10.241324],
                 [63.370684, 10.241324],
                 [63.370684, 10.241324],
                 [63.366214, 10.561421],
                 [63.38328, 10.396311],
                 [63.341723, 10.211833],
                 [63.341486, 10.239042],
                 [63.479199, 10.907748],
                 [63.444349, 10.476854],
                 [63.444349, 10.476854],
                 [63.444349, 10.476854],
                 [63.435548, 10.738245],
                 [63.434749, 10.727295],
                 [63.479199, 10.907748],
                 [63.342243, 10.218839],
                 [63.331984, 10.255967],
                 [63.444349, 10.476854],
                 [63.433069, 10.524123],
                 [63.380486, 10.610146],
                 [63.463972, 10.904488],
                 [63.472118, 10.799097],
                 [63.35884, 10.425874],
                 [63.432909, 10.360612],
                 [63.434536, 10.5116],
                 [63.382334, 10.396859],
                 [63.386254, 10.286923],
                 [63.42392, 10.395834],
                 [63.437645, 10.725534],
                 [63.445826, 10.865411],
                 [63.439951, 10.62777],
                 [63.435456, 10.407962],
                 [63.226387, 10.948586],
                 [63.2213, 10.949808],
                 [63.338656, 10.207359],
                 [63.419896, 10.807272],
                 [63.4347, 10.727142],
                 [63.475268, 10.79669],
                 [63.331758, 10.163297],
                 [63.431856, 10.384758],
                 [63.340631, 10.25444],
                 [63.4347, 10.727142],
                 [63.341713, 10.511396],
                 [63.341713, 10.511396],
                 [63.341713, 10.511396],
                 [63.362997, 10.704546],
                 [63.341713, 10.511396],
                 [63.341713, 10.511396],
                 [63.362997, 10.704546],
                 [63.341713, 10.511396],
                 [63.362997, 10.704546],
                 [63.341713, 10.511396],
                 [63.362997, 10.704546],
                 [63.341713, 10.511396],
                 [63.341713, 10.511396],
                 [63.341713, 10.511396],
                 [63.341713, 10.511396],
                 [63.362997, 10.704546],
                 [63.341713, 10.511396],
                 [63.362997, 10.704546],
                 [63.362997, 10.704546],
                 [63.362997, 10.704546],
                 [63.362997, 10.704546],
                 [63.362997, 10.704546],
                 [63.362997, 10.704546],
                 [63.341713, 10.511396],
                 [63.362997, 10.704546],
                 [63.362997, 10.704546],
                 [63.341713, 10.511396],
                 [63.341713, 10.511396],
                 [63.341713, 10.511396],
                 [63.341713, 10.511396],
                 [63.362997, 10.704546],
                 [63.362997, 10.704546],
                 [63.341713, 10.511396],
                 [63.341713, 10.511396],
                 [63.434363, 10.719165],
                 [63.447421, 10.895181],
                 [63.339974, 10.225442],
                 [63.338656, 10.207359],
                 [63.435946, 10.726334],
                 [63.435548, 10.738245],
                 [63.433069, 10.524123],
                 [63.445826, 10.865411],
                 [63.435978, 10.736485],
                 [63.433021, 10.747956],
                 [63.477955, 10.7898],
                 [63.435978, 10.736485],
                 [63.441678, 10.475299],
                 [63.435548, 10.738245],
                 [63.438942, 10.831149],
                 [63.431976, 10.379629],
                 [63.434893, 10.512139],
                 [63.435548, 10.738245],
                 [63.280734, 10.467035],
                 [63.432619, 10.370206],
                 [63.4347, 10.727142],
                 [63.322004, 10.199337],
                 [63.431856, 10.384758],
                 [63.345209, 10.216342],
                 [63.431639, 10.530636],
                 [63.437645, 10.725534],
                 [63.478738, 10.792764],
                 [63.437645, 10.725534],
                 [63.479544, 10.789566],
                 [63.437974, 10.819022],
                 [63.441606, 10.831724],
                 [63.339857, 10.208419],
                 [63.437854, 10.393113],
                 [63.439939, 10.512436],
                 [63.435548, 10.738245],
                 [63.338656, 10.207359],
                 [63.42777, 10.107565],
                 [63.342308, 10.218956],
                 [63.434749, 10.727295],
                 [63.344996, 10.546644],
                 [63.241432, 10.539511],
                 [63.47191, 10.846636],
                 [63.342308, 10.218956],
                 [63.342308, 10.218956],
                 [63.444349, 10.476854],
                 [63.435548, 10.738245],
                 [63.342179, 10.218902],
                 [63.342308, 10.218956],
                 [63.314364, 10.167923],
                 [63.342308, 10.218956],
                 [63.343106, 10.207],
                 [63.340329, 10.226763],
                 [63.441164, 10.659472],
                 [63.435548, 10.738245],
                 [63.345451, 10.215399],
                 [63.342953, 10.217752],
                 [63.342872, 10.216279],
                 [63.40122, 10.388917],
                 [63.4347, 10.727142],
                 [63.4347, 10.727142],
                 [63.344943, 10.199876],
                 [63.429959, 10.559202],
                 [63.324544, 10.186805],
                 [63.298066, 10.699752],
                 [63.435548, 10.738245],
                 [63.435548, 10.738245],
                 [63.431478, 10.536268],
                 [63.430855, 10.558421],
                 [63.479544, 10.789566],
                 [63.331758, 10.163297],
                 [63.41498, 10.085844],
                 [63.444425, 10.634652],
                 [63.435548, 10.738245],
                 [63.331984, 10.255967],
                 [63.438215, 10.704999],
                 [63.459489, 10.905736],
                 [63.431619, 10.380303],
                 [63.441344, 10.698423],
                 [63.322847, 10.434965],
                 [63.441164, 10.659472],
                 [63.479247, 10.790186],
                 [63.374142, 10.412399],
                 [63.343722, 10.191333],
                 [63.435548, 10.738245],
                 [63.384386, 10.390975],
                 [63.343722, 10.191333],
                 [63.341514, 10.247127],
                 [63.344178, 10.192492],
                 [63.342308, 10.218956],
                 [63.435548, 10.738245],
                 [63.27646, 10.29658],
                 [63.4376, 10.34395],
                 [63.43446, 10.64448],
                 [63.477518, 10.790473],
                 [63.345209, 10.216342],
                 [63.472118, 10.799097],
                 [63.342872, 10.216279],
                 [63.433069, 10.377509],
                 [63.336193, 10.215668],
                 [63.337741, 10.220708],
                 [63.337741, 10.220708],
                 [63.251392, 10.89201],
                 [63.345846, 10.49853],
                 [63.256126, 10.493877],
                 [63.258079, 10.294765],
                 [63.347805, 10.298789],
                 [63.349481, 10.099031],
                 [63.259748, 10.095618],
                 [63.405849, 10.331452],
                 [63.282875, 10.468346],
                 [63.435564, 10.503219],
                 [63.347805, 10.298789],
                 [63.349481, 10.099031],
                 [63.437529, 10.302859],
                 [63.478397, 10.867854],
                 [63.453652, 10.906356],
                 [63.453652, 10.906356],
                 [63.460826, 10.906841],
                 [63.453652, 10.906356],
                 [63.435564, 10.503219],
                 [63.453733, 10.900346],
                 [63.480242, 10.79368],
                 [63.443823, 10.350263],
                 [63.435186, 10.407989],
                 [63.349481, 10.099031],
                 [63.437529, 10.302859],
                 [63.39307, 10.691156],
                 [63.430791, 10.903841],
                 [63.253899, 10.692961],
                 [63.34361, 10.698234],
                 [63.451858, 10.906239],
                 [63.430791, 10.903841],
                 [63.433318, 10.703543],
                 [63.34361, 10.698234],
                 [63.345846, 10.49853],
                 [63.349481, 10.099031],
                 [63.433318, 10.703543],
                 [63.433318, 10.703543],
                 [63.439216, 10.102463],
                 [63.332246, 10.207979],
                 [63.439372, 10.473871],
                 [63.439372, 10.473871],
                 [63.439047, 10.47406],
                 [63.439372, 10.473871],
                 [63.337741, 10.220708],
                 [63.439372, 10.473871],
                 [63.337741, 10.220708],
                 [63.340538, 10.245097],
                 [63.439372, 10.473871],
                 [63.439368, 10.474114],
                 [63.337741, 10.220708],
                 [63.334641, 10.214761],
                 [63.337741, 10.220708],
                 [63.435548, 10.738245],
                 [63.342308, 10.218956],
                 [63.369742, 10.396374],
                 [63.4376, 10.34395],
                 [63.4376, 10.34395],
                 [63.434363, 10.719165],
                 [63.435548, 10.738245],
                 [63.472953, 10.89899],
                 [63.468595, 10.90023],
                 [63.340538, 10.245097],
                 [63.342308, 10.218956],
                 [63.342179, 10.218902],
                 ])

grid_points = 2 ** 7  # Grid points in each dimension
N = 32  # Number of contours

# Compute the kernel density estimate
kde = FFTKDE(kernel='gaussian', norm=2)
grid, points = kde.fit(data).evaluate(grid_points)

# The grid is of shape (obs, dims), points are of shape (obs, 1)
x, y = np.unique(grid[:, 0]), np.unique(grid[:, 1])
z = points.reshape(grid_points, grid_points).T

# Plot the kernel density estimate
# ax.contour(x, y, z, N, linewidths=0.8, colors='k')
plt.contourf(x, y, z, N, cmap="viridis")
plt.plot(data[:, 0], data[:, 1], 'ok', ms=3)

plt.show()