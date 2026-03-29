"""Common base interface for all finite/virtual element spaces."""


class SpaceBase:
    """
    Common base class for finite and virtual element spaces.

    Concrete spaces should implement bind(), evaluateLocal(), and interpolate().
    """

    def bind(self, element_or_vertices):
        raise NotImplementedError

    def evaluateLocal(self, x):
        raise NotImplementedError

    def evaluateLocalGradient(self, x):
        raise NotImplementedError

    def interpolate(self, gf):
        raise NotImplementedError