import torch
import numpy as np
from mlutils.utils import batch_triangle_intersect


EPSILON = 0.0000001


class Mesh(object):
    """ Models a triangle mesh. """

    def __init__(self, vertices, indices):
        assert vertices.shape[1] == indices.shape[1] == 3
        self.vertices = vertices.type(torch.float32)
        self.indices = indices.type(torch.long)

    def to(self, device):
        self.vertices = self.vertices.to(device)
        self.indices = self.indices.to(device)
        return self

    def get_triangle(self, face_idx):
        return self.vertices[self.indices[face_idx]]

    def copy(self):
        v = torch.empty_like(self.vertices).copy_(self.vertices)
        i = torch.empty_like(self.indices).copy_(self.indices)
        return Mesh(v, i)

    @property
    def triangles(self):
        return torch.index_select(self.vertices, 0, self.indices.view(-1)).view(-1, 3, 3)

    @property
    def device(self):
        return self.vertices.device

    @property
    def num_vertices(self):
        return len(self.vertices)

    @property
    def num_faces(self):
        return len(self.indices)

    @staticmethod
    def from_off(file, normalize=False):
        v, i = read_off(file)
        v, i = torch.from_numpy(v), torch.from_numpy(i)
        if normalize:
            v = mlutils.utils.preprocess.normalize(v)
        return Mesh(v, i)

    @staticmethod
    def from_obj(file):
        print('reading mesh from .obj not yet implemented')
        return None


def read_off(off):
    """ Reads the vertices and triangle indices from a .off mesh file.
    """
    with open(off, 'r') as file:
        lines = file.readlines()
    lines = [line.strip().replace('OFF', '') for line in lines if line.strip() != 'OFF']
    n_vertices, n_faces, _ = [int(x) for x in lines[0].split()]
    lines = lines[1:]
    vertices = np.array([np.fromstring(verts, sep=' ', dtype=float) for verts in lines[:n_vertices]])
    lines = lines[n_vertices:]
    indices = np.array([np.fromstring(faces[1:], sep=' ', dtype=int) for faces in lines[:n_faces]])
    return vertices, indices


def vertices_from_obj(obj):
    """ Creates an N x 3 numpy array of vertices from a .obj mesh file.
    """
    with open(obj, 'r') as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines if line.startswith('v')]
    points = [line.split()[1:] for line in lines]
    points = [point for point in points if len(point) == 3]
    points = [(float(x), float(y), float(z)) for x, y, z in points]
    return np.array(points)


def triangle_areas(mesh):
    triangles = mesh.triangles
    v1, v2, v3 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    e1, e2 = v2 - v1, v3 - v1
    areas = 0.5 * torch.norm(torch.cross(e1, e2, dim=1), dim=1)
    return areas


def sample_points_on_mesh(mesh, n, return_faces=False):
    """ Samples points uniformly on the surface of the triangle mesh.
        Optionally return the indices of faces on which the points were sampled.

        Approximation for uniform sampling from:
        http://openaccess.thecvf.com/content_cvpr_2018/papers/Le_PointGrid_A_Deep_CVPR_2018_paper.pdf
    """
    areas = triangle_areas(mesh)
    faces = torch.multinomial(areas, n, replacement=True)
    indices = mesh.indices[faces]
    verts = torch.index_select(mesh.vertices, 0, indices.view(-1)).view(-1, 3, 3)
    a, b, c = verts[:, 0], verts[:, 1], verts[:, 2]
    u, v = torch.rand((2, n, 1), dtype=verts.dtype, device=mesh.device)
    points = (1 - u.sqrt()) * a + u.sqrt() * (1 - v) * b + v * u.sqrt() * c
    return (points, faces) if return_faces else points


def test_occlusion(mesh, point):
    """ Tests whether a point is occluded by a triangle in the mesh.
        A ray is cast from the origin in the direction of the point and tested with intersection of every mesh triangle.
        :return True, if the point is occluded by the mesh, False otherwise.
    """
    ray_origin = torch.zeros(3, device=point.device)
    ray_vector = point - ray_origin
    triangles = mesh.triangles
    mask, t = batch_triangle_intersect(ray_origin, ray_vector, triangles)
    candidate_t = torch.masked_select(t, mask)
    # if t = 1, it means that this is the face where the point was sampled from
    # t < 1 means another triangle is closer to the eye and occluding
    return (candidate_t < 1 - EPSILON).any()
