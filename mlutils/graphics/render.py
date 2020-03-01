import torch

from mlutils.geometry.mesh import test_occlusion


def render_viewpoint(mesh, points, viewpoint):
    # transform mesh and point cloud to (local) viewpoint coordinate frame
    mesh_local = mesh.copy()
    mesh_local.vertices = viewpoint.world2cam(mesh.vertices)
    points_local = viewpoint.world2cam(points)

    visible_mask = torch.stack([~test_occlusion(mesh_local, point) for point in points_local])
    # points_local: (N, 3)
    # visible_mask: (N, )
    return points_local, visible_mask

