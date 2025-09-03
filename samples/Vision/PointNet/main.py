import os

import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from hbm_runtime import HB_HBMRuntime
np.set_printoptions(precision=5, suppress=True, threshold=5)


DUMP_INPUT = False


def load_data(point_file):
    """
    Load the point cloud data and convert it to ndarray

    Parameters:
        point_file: string, path of .pts data
    Returns:
       point_set: point clound represented in np.array format
    """

    point_set = np.loadtxt(point_file).astype(np.float32)

    # normailization
    point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
    dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
    point_set = point_set / dist  # scale

    return point_set


def visualize(point_set:np.ndarray):
    """
    Create a 3D view for data visualization

    Parameters:
        point_set: np.ndarray, the coordinate data in X Y Z format
    """

    fig = plt.figure(dpi=192, figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')
    X = point_set[:, 0]
    Y = point_set[:, 2]
    Z = point_set[:, 1]

    # Scale the view of each axis to adapt to the coordinate data distribution
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() * 0.5
    mid_x = (X.max() + X.min()) * 0.5
    mid_y = (Y.max() + Y.min()) * 0.5
    mid_z = (Z.max() + Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tick_params(labelsize=5)
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_zlabel('Z', fontsize=10)

    return ax


def main():
    points = load_data('chair.pts')
    X = points[:, 0]
    Y = points[:, 2]
    Z = points[:, 1]
    ax = visualize(points)
    ax.scatter3D(X, Y, Z, s=5, cmap="jet", marker="o", label='chair')
    ax.set_title('3D Visualization')
    plt.legend(loc='upper right', fontsize=8)
    plt.savefig('chair.png', bbox_inches='tight', dpi=192)

    # Parts of a chair
    classes = ['back', 'seat', 'leg', 'arm']

    # Preprocess the input data
    point = points.transpose(1, 0)
    point = np.expand_dims(point, axis=0)

    if DUMP_INPUT:
        np.save('point.npy', point)
        
    inf = HB_HBMRuntime("pointnet.hbm")
    output = inf.run(point)
    pred = output['pointnet']['pred']
    pred = np.argmax(pred[0], axis=1)
    ax = visualize(point)

    for i, name in enumerate(tqdm([0, 1, 2, 3], desc="Labels")):
        XCur = []
        YCur = []
        ZCur = []
        for j, nameCur in enumerate(tqdm(pred, desc=f"Label {name}", leave=False)):
            if name == nameCur:
                XCur.append(X[j])
                YCur.append(Y[j])
                ZCur.append(Z[j])
        XCur = np.array(XCur)
        YCur = np.array(YCur)
        ZCur = np.array(ZCur)

        # add current point of the part
        ax.scatter(XCur, YCur, ZCur, s=5, cmap="jet", marker="o", label=classes[i])

    ax.set_title('3D Segmentation Visualization')
    plt.legend(loc='upper right', fontsize=8)
    plt.savefig('chair_res.png', bbox_inches='tight', dpi=192)


if __name__ == "__main__":
    main()