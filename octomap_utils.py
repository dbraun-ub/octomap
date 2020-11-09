import numpy as np


def constructRay(resolution, origin, node):
    resolution = 0.1#self.worldSize / 2**(self.getMaxDepth())
    O = origin
    P = node.position
    OP = [P[0] - O[0], P[1] - O[1], P[2] - O[2]]
    step = np.multiply(resolution, OP / np.linalg.norm(OP))
    pts = np.array([np.arange(O[0], P[0], step[0]), np.arange(O[1], P[1], step[1]) , np.arange(O[2], P[2], step[2])]).T
    return pts[1:]
