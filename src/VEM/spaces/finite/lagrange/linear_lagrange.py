import numpy
from ...base import SpaceBase
from ...common.triangle_geometry import coerce_triangle_vertices

class LinearLagrangeSpace(SpaceBase):
    def __init__(self,view):
        self.localDofs = 3
        self.view   = view
        self.dim    = view.dimension
        self.layout = lambda gt: 1 if gt.dim == 0 else 0
        self.mapper = view.mapper(self.layout)
        self.points = numpy.array( [ [0,0],[1,0],[0,1] ] )
        # by default this is the reference element
        self.vertices = numpy.array( [ [0,0],[1,0],[0,1] ], dtype=float )
    # bind to a different physical triangles
    # - store origin and coordinate axes
    # Use the vertices when setting up the CLS so that we can easily
    # switch between a physical VEM and a reference VEM
    def bind(self, element_or_vertices):
        vertices = coerce_triangle_vertices(element_or_vertices)
        self.vertices[0] = vertices[0]
        self.vertices[1] = vertices[1] - vertices[0]
        self.vertices[2] = vertices[2] - vertices[0]
    # evaluate basis function in local coordinates so
    # (0,0) is vertex[0], (1,0) is vertex[1], (0,1) is vertex[2]
    def evaluateLocal(self, x):
        bary = 1.-x[0]-x[1], x[0], x[1]
        return numpy.array( bary )
    def interpolate(self,gf):
        dofs = numpy.zeros(len(self.mapper))
        for e in self.view.elements:
            dofs[self.mapper(e)] = gf(e,self.points.transpose())
        return dofs