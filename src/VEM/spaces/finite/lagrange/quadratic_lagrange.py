import numpy
from ...base import SpaceBase
from ...common.triangle_geometry import coerce_triangle_vertices

class QuadraticLagrangeSpace(SpaceBase):
    def __init__(self,view):
        self.localDofs = 6
        self.view   = view
        self.dim    = view.dimension
        self.layout = lambda gt: 1 if gt.dim == 0 or gt.dim == 1 else 0
        self.mapper = view.mapper(self.layout)
        self.points = numpy.array( [ [0,0],[1,0],[0,1],
                                   [0.5,0],[0,0.5],[0.5,0.5] ] )
        # by default this is the reference element
        self.vertices = numpy.array( [ [0,0],[1,0],[0,1] ], dtype=float )
    # bind to a different physical triangles
    # - store origin and coordinate axes
    def bind(self, element_or_vertices):
        vertices = coerce_triangle_vertices(element_or_vertices)
        self.vertices[0] = vertices[0]
        self.vertices[1] = vertices[1] - vertices[0]
        self.vertices[2] = vertices[2] - vertices[0]
    def evaluateLocal(self, x):
        bary = 1.-x[0]-x[1], x[0], x[1]
        return numpy.array([ bary[i]*(2.*bary[i]-1.) for i in range(3) ] +\
               [ 4.*bary[(3-j)%3]*bary[(4-j)%3] for j in range(3) ])
    def interpolate(self, gf):
        dofs = numpy.zeros(len(self.mapper))
        points_t = self.points.transpose()
        for e in self.view.elements:
            dofs[self.mapper(e)] = gf(e, points_t)
        return dofs