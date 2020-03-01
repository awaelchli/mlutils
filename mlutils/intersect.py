import torch
import time

EPSILON = 0.0000001


def batch_triangle_intersect(ray_origin, ray_vector, triangles):
    """ Ray intersection test with a batch of triangles.
        :param ray_origin: Point of origin of the ray, tensor of shape (3, )
        :param ray_vector: Travel direction of ray, tensor of shape (3, )
        :param triangles: Batch of triangle vertices of shape (N, 3, 3), where N is the batch size,
        the 2nd dimension are the vertices and the last dimension are the x-, y- and z-coordinates.
        :return: A ByteTensor of size (N, ) containing the boolean value of the intersection, and a FloatTensor
        of size (N, ) with the line parameter t where the intersection occurred.
    """
    N = triangles.size(0)
    ray_origin = ray_origin.unsqueeze(0).repeat(N, 1)
    ray_vector = ray_vector.unsqueeze(0).repeat(N, 1)

    vertex0, vertex1, vertex2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]  # (N, 3)
    edge1, edge2 = vertex1 - vertex0, vertex2 - vertex0  # (N, 3)

    h = torch.cross(ray_vector, edge2, dim=1)  # (N, 3)
    a = torch.sum(edge1 * h, 1)  # (N, )
    result = torch.abs(a) > EPSILON

    f = 1.0 / a
    s = ray_origin - vertex0  # (N, 3)
    u = f * torch.sum(s * h, 1)  # (N, )
    result = result & (u >= 0) & (u <= 1)

    q = torch.cross(s, edge1, dim=1)  # (N, 3)
    v = f * torch.sum(ray_vector * q, 1)  # (N, )
    result = result & (v >= 0) & (u + v <= 1)

    t = f * torch.sum(edge2 * q, 1)  # (N, )
    result = result & (t > 0)
    return result, t


def triangle_intersect(ray_origin, ray_vector, triangle):
    # ray_origin: 3
    # ray_vector: 3
    # triangle: 3 x 3 (vertex_id, dim)

    vertex0, vertex1, vertex2 = triangle[0], triangle[1], triangle[2]
    edge1, edge2 = vertex1 - vertex0, vertex2 - vertex0
    h = torch.cross(ray_vector, edge2)
    a = torch.dot(edge1, h)
    if torch.abs(a) < EPSILON:
        return False
    f = 1.0 / a
    s = ray_origin - vertex0
    u = f * torch.dot(s, h)
    if u < 0 or u > 1:
        return False
    q = torch.cross(s, edge1)
    v = f * torch.dot(rayVector, q)
    if v < 0 or u + v > 1:
        return False
    t = f * torch.dot(edge2, q)
    if t > 0:
        return True
    return False


if __name__ == '__main__':
    device = torch.device('cuda')
    rayOrigin = torch.Tensor([0, 1, 1]).to(device)
    rayVector = torch.Tensor([0, 0, -1]).to(device)
    triangle = torch.Tensor([
        [2, 0, 0],
        [0, 2, 0],
        [-2, 0, 0],
    ]).to(device)

    batch = triangle.unsqueeze(0).repeat(10000, 1, 1)
    print('batch', batch.shape)

    start = time.time()
    for i in range(2000):
        intersect = batch_triangle_intersect(rayOrigin, rayVector, batch)
    print('time', time.time() - start)
    print(intersect)
