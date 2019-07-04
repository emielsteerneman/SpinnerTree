import cv2
import numpy as np
import math
import itertools
from functools import reduce
from vector import Vector as Vec
from triangle import Triangle
from functions import lineFromPoints, perpendicularBisectorFromLine, lineLineIntersection, distance, toCV, distancePointToSegment, projectPointOntoSegment, otherCorner

from dijkstra import dijkstra
from bowyerWatson import bowyerWatson

# 90mm = 500px -> 1mm = 5.5px
def mm(mm):
	# return int(mm * 5.5586) # configured for 15.6" 1920x1080
	return int(mm * 3.6460) # configured for 23.8" 1920x1080

NPLATES = 1
SCALAR = 1 / math.sqrt(NPLATES)

WIDTH = 1000
HEIGHT = 600
R = int(mm(5) * SCALAR)

NPOINTS = 80
img = np.ones((HEIGHT, WIDTH, 3))

def sampleFromNormal():
	M = 100
	xi = np.random.rand(M)
	x = (xi.sum() - M/2) / np.sqrt(M/12)
	return x

def sampleFromNormal2D(x=0, y=0, sigma=25):
	px = sigma * sampleFromNormal() + x
	py = sigma * sampleFromNormal() + y
	return Vec(px, py)

def rotate(x, y, r):
	return np.array([math.cos(r) * x - math.sin(r) * y, math.sin(r) * x + math.cos(r) * y])

def generatePoint(radius=200, mu=0, sigma=25):
	r = np.random.rand() * (2*math.pi * 2/4) - (2*math.pi * 1/4)
	p = Vec(0, -radius).rotate(r) + sampleFromNormal2D(sigma=sigma)
	p = Vec(int(p[0]), int(p[1]))
	return p

def inPlane(P, r, w, h):
	return 0 < P[0]-r and 0 < P[1]-r and P[0]+r < w and P[1]+r < h

def drawLeaf(img, vec):
	cv2.circle(img, toCV(vec), R, (0, 0.5, 0), mm(1), cv2.LINE_AA)

def plotLine(img, a, b, c, colour=(0, 0, 0)):
	# ax + by + c = 0 -> by = -ax - c -> y = (-ax - c) / b
	P = Vec(-100, (-a * -100 - c) / b)
	Q = Vec(1100, (-a * 1100 - c) / b)
	cv2.line(img, toCV(P), toCV(Q), colour, 2)

def drawTriangle(img, T, colour=(0, 0, 0)):
	cv2.line(img, toCV(T.v1), toCV(T.v2), colour, 1)
	cv2.line(img, toCV(T.v2), toCV(T.v3), colour, 1)
	cv2.line(img, toCV(T.v3), toCV(T.v1), colour, 1)

O = np.array([WIDTH//2, HEIGHT - HEIGHT//5])
rootNode = Vec(WIDTH//2, HEIGHT)
cv2.circle(img, toCV(rootNode), 8, (1, 0, 1), -1)



######## GENERATE LEAFS ########
leafs = []
i = 0
while len(leafs) < NPOINTS and i < NPOINTS*20:
	P = generatePoint(radius=300, sigma=40) + O

	minDistance = 2 * R + mm(5)
	if inPlane(P, R, WIDTH, HEIGHT):
		distances = [distance(P, L) < minDistance for L in leafs]
		if not any(distances):
			leafs.append(P)
	i += 1
print(len(leafs), "leafs added")
######## LEAFS GENERATED ########




######## GENERATE PATHS ########
img = np.ones((HEIGHT, WIDTH, 3))
for leaf in leafs:
	drawLeaf(img, leaf)

paths = []

# Generate Delaunay triangulation
triangulation = bowyerWatson(leafs)

### Remove all from triangulation whose center does not lie within another triangle ###
_triangulation = triangulation
triangulation = []
for T in _triangulation:
	if any([_T.containsPoint(T.C) for _T in _triangulation]):
		triangulation.append(T)
### Remove all from triangulation whose center does not lie within another triangle ###

### Add all possible branches as paths ###
for T in triangulation:
	for _T in triangulation:
		if T.sharesVertexWith(_T):
			# Prevent duplicate paths, since line A-B != line B-A
			if (T.C, _T.C) not in paths and (_T.C, T.C) not in paths: 
				paths.append((T.C, _T.C))
print("%d voronoi paths added" % len(paths))
### Add all possible branches as paths ###

print("Drawing %d paths" % len(paths))
for (v1, v2) in paths:
	cv2.line(img, toCV(v1), toCV(v2), (0, 0, 1), 14)

### Add paths from leafs to branches ###
print("\nAdd paths from leafs to branches")
leafPaths = []
for L in leafs:
	minDist, minProj, minLine = float("inf"), None, None
	for P in paths: # Find path closest to leaf
		V, W = P
		proj = projectPointOntoSegment(L, V, W)
		dist = distancePointToSegment(L, V, W)
		if dist < minDist:
			minDist, minProj, minLine = dist, proj, P
	
	## Split up path V-W into paths V-proj, proj-W
	if minProj not in minLine:
		paths.remove(minLine) 					# Remove original segment
		paths.append((minLine[0], minProj))		# Add first subsegment
		paths.append((minProj, minLine[1]))		# Add second subsegment
	leafPaths.append((L, minProj))			# Add line from leaf to segment
paths += leafPaths
### Add paths from leafs to branches ###

### Add path from root node to closest voronoi node ###
distances = [distance(rootNode, T.C) for T in triangulation]
iClosest = distances.index(min(distances))
paths.append((rootNode, triangulation[iClosest].C))

## Draw all paths
print("Drawing %d paths" % len(paths))
for (v1, v2) in paths:
	cv2.line(img, toCV(v1), toCV(v2), (np.random.rand(), np.random.rand(), np.random.rand()), 3)

cv2.imshow("img", img)
cv2.waitKey()	
######## PATHS GENERATED ########



######## FILTER PATHS ########
img = np.ones((HEIGHT, WIDTH, 3))

# Get unique list of all nodes (leafs, voronoi, rootnode)
nodes = list(set([val for sublist in paths for val in sublist]))

for leaf in leafs:
	drawLeaf(img, leaf)

routes = dijkstra(list(nodes), list(paths), rootNode)

paths = []
for node in leafs:
	## Create path for node
	path = []
	previous = node
	while routes[previous] != None:
		path.append(previous)
		previous = routes[previous]
	path.append(rootNode)

	## Draw path
	for i in range(len(path)-1):	
		P = (path[i], path[i+1])
		if P not in paths:
			paths.append(P)

for path in paths:
	cv2.line(img, toCV(path[0]), toCV(path[1]), (0, 0, 0), mm(3))
	cv2.line(img, toCV(path[0]), toCV(path[1]), (0, 0, 1), 1)



cv2.imshow("img", img)
cv2.waitKey()	


# cv2.imshow("img", img)
# cv2.waitKey()	















import numpy as np
import scipy
from scipy.special import comb

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


if __name__ == "__main__":
	from matplotlib import pyplot as plt

	neighbours = []
	N = None
	while True:
		N = nodes[int(np.random.rand() * len(nodes))]
		print(N)
		neighbours = [path for path in paths if N in path]
		print(neighbours)
		if 2 <= len(neighbours):
			break


	p1 = otherCorner(neighbours[0], N)
	p2 = otherCorner(neighbours[1], N)
	
	print(N, p1, p2)

	cv2.circle(img, toCV(p1), 3, (1, 0, 0), -1)
	cv2.circle(img, toCV(p2), 3, (1, 0, 0), -1)
	cv2.circle(img, toCV(N), 3, (1, 0, 0), -1)

	nPoints = 3
	points = [p1, N, p2]
	xpoints = [p[0] for p in points]
	ypoints = [p[1] for p in points]

	xvals, yvals = bezier_curve(points, nTimes=1000)

	z = zip(xvals, yvals)
	for x, y in z:
		# print(x, y)
		img[int(y), int(x)] = (1, 0, 0)
		# cv2.circle(img, (int(x), int(y)), 0, (1, 0, 0), -1)
	cv2.imshow("img", img)
	cv2.waitKey()
	# print(xvals, yvals)

	# plt.plot(xvals, yvals)
	# plt.plot(xpoints, ypoints, "ro")
	# for nr in range(len(points)):
	#     plt.text(points[nr][0], points[nr][1], nr)

	# plt.show()








exit()




























