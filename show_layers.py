import numpy as np


def main():
    data = np.load("/home/laurens/Documents/sdm_dl/data/sdm_dl/out/out.npz", allow_pickle=True)
    print(list(data.keys()))
    meta = data["meta"].tolist()
    print(meta)
    features = data["columns"].tolist()
    print(features)
    layers_ = data["layers"]
    print(len(layers_))
    print("point 1200", layers_[1200])
    lat_dim = int((meta["max_lat"] - meta["min_lat"]) / meta["cell_size_deg"])
    lon_dim = int((meta["max_lon"] - meta["min_lon"]) / meta["cell_size_deg"])
    print(lat_dim, lon_dim, (lat_dim) * (lon_dim))

    # for p in range(len(layers_)):
    #     print(layers_[p][layer])
    #     print(p, layers_[p][layer][0][0])

    for feature in features:
        layer = features.index(feature)
        layer_0 = [float(layers_[p][layer][0][0]) for p in range(len(layers_))]
        layer_0 = np.reshape(layer_0, (lat_dim, lon_dim))
        layer_0 = np.flipud(layer_0)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        plt.imshow(layer_0)
        plt.gca().set_aspect("equal")
        plt.colorbar()
        plt.show()

        # for p in tqdm(range(2312)):
        #     print(layers_[p][layer][0][0])
        print("layer_0", layer_0)


if __name__ == '__main__':
    main()
