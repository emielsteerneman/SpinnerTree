import math
import numpy as np
from scipy.special import comb
from vector import Vector as Vec
from functools import reduce


def distance(v, w):
	return (v - w).norm()

def toCV(vec):
	return int(vec[0]), int(vec[1])

def lineFromPoints(P, Q):
	a = P[1] - Q[1]
	b = Q[0] - P[0]
	c = P[0]*Q[1] - Q[0]*P[1]

	return a, b, c

def perpendicularBisectorFromLine(P, Q, a, b, c):
	midPoint = (P + Q) * 0.5
	c = -b*midPoint[0] + a*midPoint[1]
	a, b = -b, a
	return a, b, c

def lineLineIntersection(a1, b1, c1, a2, b2, c2):
	determinant = a1*b2 - a2*b1
	if determinant == 0:
		print("[lineLineIntersection] Lines not intersecting!")
		return None
	return Vec((b2*c1 - b1*c2)/determinant, (a1*c2 - a2*c1)/determinant)

def distancePointToSegment(P, V, W):
	proj = projectPointOntoSegment(P, V, W)
	return distance(P, proj)

def intersectionOfSegments(V, W):
	# https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
	p, r = V[0], V[1]-V[0]
	q, s = W[0], W[1]-W[0]
	
	t = (q-p).cross(s) * 1/(r.cross(s))
	u = (p-q).cross(r) * 1/(s.cross(r))

	# If r × s = 0 then lines are either parallel or colinear
	if abs(r.cross(s)) < 1e-3: # 1e-3 arbitrary
		return None
		# if abs((q-p).cross(r)) < 1e-3: # Colinear
		# else:							 # Parallel
	return p + t * r

def projectPointOntoSegment(P, V, W):
	### NOTE !! For some reason the projection in projectPointOntoSegment might be extremely 
	### NOTE !! slightly off, probably due to float imprecision. Problem when t = 0 or t = 1
	length = distance(V, W)
	if V == W:
		return V
	# Find projection of p onto w, by first moving everything to (0, 0) (aka subtracting v)
	f = ((P-V) * (W-V)) / ((W-V)*(W-V))
	# Clamp projection between [0, 1]
	t = max(0, min(f, 1))
	# Calculate the projection
	proj = V + t * (W-V)
	
	return proj

def otherCorner(edge, corner):
	return edge[0] if corner == edge[1] else edge[1]

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals

