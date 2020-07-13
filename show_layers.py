import numpy as np


def main():
    data = np.load("/home/laurens/Documents/sdm_dl/data/sdm_dl/out/159464502058.npz", allow_pickle=True)
    print(list(data.keys()))
    features = data["columns"].tolist()
    print(features)
    layers_ = data["layers"]
    print(len(layers_))
    print("point 1200", layers_[1200])
    lat_dim = 71 - 37
    lon_dim = 43 - -25
    print((lat_dim) * (lon_dim))
    layer = features.index("lon_latlon")

    for p in range(2312):
        print(layers_[p][layer])
        print(p, layers_[p][layer][0][0])
    exit(0)

    layer_0 = [float(layers_[p][layer][0][0]) for p in range(2312)]
    layer_0 = np.reshape(layer_0, (lat_dim, lon_dim))
    layer_0 = np.flipud(layer_0)

    import matplotlib.pyplot as plt
    plt.imshow(layer_0)
    plt.gca().set_aspect("equal")
    plt.show()

    # for p in tqdm(range(2312)):
    #     print(layers_[p][layer][0][0])
    print("layer_0", layer_0)


if __name__ == '__main__':
    main()
