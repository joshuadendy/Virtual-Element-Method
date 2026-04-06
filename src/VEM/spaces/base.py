"""Common base interface for all finite/virtual element spaces."""


class SpaceBase:
    """
    Minimal base class for the first package split.

    We deliberately avoid abc.ABC for now to keep the transition simple.
    Concrete spaces should implement the methods below.
    """

    def bind(self, element_or_vertices):
        raise NotImplementedError

    def evaluateLocal(self, x):
        raise NotImplementedError

    def interpolate(self, gf):
        raise NotImplementedError