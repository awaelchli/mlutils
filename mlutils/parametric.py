import torch
from numpy import pi


class Sphere:
    """ A sphere parameterized by longitude and latitude. """

    def __init__(self, radius=1.0):
        self._radius = radius

    @property
    def radius(self):
        return self._radius

    def __call__(self, longitude, latitude):
        return self.sample(longitude, latitude)

    def sample(self, longitude, latitude):
        # longitude/azimuth within range [0, 2pi]
        # latitude within range [-pi/2, pi/2]
        phi = longitude
        theta = latitude + pi / 2  # colatitude/inclination
        r = self.radius
        x = r * torch.sin(theta) * torch.cos(phi)
        y = r * torch.sin(theta) * torch.sin(phi)
        z = r * torch.cos(theta)
        return torch.as_tensor([x, y, z])


class Curve:
    """ A one-dimensional curve. """

    def __call__(self, t):
        return self.sample(t)

    def sample(self, t):
        pass


class Sinusoids(Curve):
    """ Weighted sum of sinusoids without phase shift. """

    def __init__(self, coeffs, period=2*pi):
        super(Sinusoids, self).__init__()
        assert len(coeffs) > 0
        self.coeffs = torch.as_tensor(coeffs)
        self.period = period

    @property
    def order(self):
        return len(self.coeffs)

    def sample(self, t):
        result = 0
        for k in torch.arange(self.order):
            result += self.coeffs[k] * torch.sin(2 * k.float() * pi * t / self.period)
        return result

