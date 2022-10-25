import open3d as o3d
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import torch

def get_pts(pcd):
    points = np.asarray(pcd.points)
    X = []
    Y = []
    Z = []
    for pt in range(points.shape[0]):
        X.append(points[pt][0])
        Y.append(points[pt][1])
        Z.append(points[pt][2])
    return np.asarray(X), np.asarray(Y), np.asarray(Z)


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    plot_radius = 0.5*max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_single_pcd(points, save_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    rotation_matrix = np.asarray([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    pcd = pcd.transform(rotation_matrix)
    X, Y, Z = get_pts(pcd)
    t = Z
    ax.scatter(X, Y, Z, c=t, cmap='jet', marker='o', s=0.5, linewidths=0)
    ax.grid(False)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    set_axes_equal(ax)
    plt.axis('off')
    plt.savefig(save_path, format='png', dpi=600)
    plt.close()

def save_pointcloud_ply(pointcloud, filename):
    """Utility function to save to disk a pointcloud in PLY format.
    Args:
        filename (str): the path to save the pointcloud.
        pointcloud (torch.Tensor): tensor containing the pointcloud to save.
          The tensor must be in the shape of :math:`(*, 3)` where the last
          component is assumed to be a 3d point coordinate :math:`(X, Y, Z)`.
    """
    if not isinstance(filename, str) and filename[-3:] == '.ply':
        raise TypeError("Input filename must be a string in with the .ply  "
                        "extension. Got {}".format(filename))
    if not torch.is_tensor(pointcloud):
        raise TypeError("Input pointcloud type is not a torch.Tensor. Got {}"
                        .format(type(pointcloud)))
    if not len(pointcloud.shape) == 3 and pointcloud.shape[-1] == 3:
        raise TypeError("Input pointcloud must be in the following shape "
                        "HxWx3. Got {}.".format(pointcloud.shape))
    # flatten the input pointcloud in a vector to iterate points
    xyz_vec: torch.Tensor = pointcloud.reshape(-1, 3)

    with open(filename, 'w') as f:
        data_str: str = ''
        num_points: int = xyz_vec.shape[0]
        for idx in range(num_points):
            xyz = xyz_vec[idx]
            if not bool(torch.isfinite(xyz).any()):
                continue
            x: float = xyz[0].item()
            y: float = xyz[1].item()
            z: float = xyz[2].item()
            data_str += '{0} {1} {2}\n'.format(x, y, z)

        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment arraiy generated\n")
        f.write("element vertex %d\n" % num_points)
        f.write("property double x\n")
        f.write("property double y\n")
        f.write("property double z\n")
        f.write("end_header\n")
        f.write(data_str)








