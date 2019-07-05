import cv2
import numpy as np
from scipy.special import comb
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
rootNode = Vec(WIDTH//2, HEIGHT-1)
cv2.circle(img, toCV(rootNode), 8, (1, 0, 1), -1)



img = np.ones((HEIGHT, WIDTH, 3))
LC = Vec(WIDTH//2, HEIGHT//2)

L1 = Vec(WIDTH//2, HEIGHT//4)
L2 = Vec(3*WIDTH//4, HEIGHT//4)
L3 = Vec(WIDTH//3, 3*HEIGHT//4)
L4 = Vec(2*WIDTH//3, 3*HEIGHT//4)
LC2= Vec(1*WIDTH//4, 2*HEIGHT//4)
E1 = (LC, L1)
E2 = (LC, L2)
E3 = (LC, L3)
E4 = (LC, L4)
E5 = (LC2,L1)
E6 = (LC2,L3)
leafs = [LC, L1, L2, L3, L4, LC2]
edges = [E1, E2, E3, E4, E5, E6]

def getPointsFromEdge(edge, dist):
	V, W = edge
	center = (V+W) * 0.5
	p1 = (dist*(V-W)*(1/(V-W).norm())).rotate(0.5*math.pi) + center
	p2 = (dist*(W-V)*(1/(W-V).norm())).rotate(0.5*math.pi) + center
	return p1, p2

def getPointsFromNode(edges, P, dist, img):
	neighbours = [otherCorner(edge, P) for edge in edges if P in edge] + [P]
	triangulation = bowyerWatson(neighbours)
	
	for T in triangulation:
		if P not in T.corners:
			print("Dafuq???")
			print(P, T)
			print(P[0], P[1])
			for Q in T.corners:
				print(Q[0], Q[1])

			exit()

	points = []
	for T in triangulation:
		# https://www.mathsisfun.com/algebra/trig-solving-sss-triangles.html
		print()
		print(T, P)
		print("%s == %s = %s, %s == %s = %s, %s == %s = %s" % (T.corners[0], P, T.corners[0] == P, T.corners[1], P, T.corners[1] == P, T.corners[2], P, T.corners[2] == P))
		V, W = [c for c in T.corners if c != P]
		a, b, c = (V-W).norm(), (P-V).norm(), (P-W).norm()
		print(a, b, c, (b**2 + c**2 - a**2), (2 * b * c))
		cosA = (b**2 + c**2 - a**2) / (2 * b * c)
		if cosA < -1:
			drawTriangle(img, T, (0, 0, 1))
			cv2.circle(img, toCV(T.C), 8, (1, 0, 0), 4)
			cv2.circle(img, toCV(T.C), T.R, (1, 0, 0), 2)
			return []
		angle = math.acos((b**2 + c**2 - a**2) / (2 * b * c))
		Q = (V-P).rotate(angle/2)
		Q = P + 1/Q.norm() * Q * dist
		points.append(Q)
	return points

# for leaf in leafs:
# 	cv2.circle(img, toCV(leaf), R, (0.5, 1, 0), mm(2))
# for V, W in edges:
# 	cv2.line(img, toCV(V), toCV(W), (1, 0.5, 0), mm(2))

# for edge in edges:
# 	p1, p2 = getPointsFromEdge(edge)
# 	cv2.circle(img, toCV(p1), 4, (0.5, 0.5, 1), -1)
# 	cv2.circle(img, toCV(p2), 4, (0.5, 0.5, 1), -1)

# for P in [LC, LC2]:
# 	points = getPointsFromNode(edges, P)
# 	for Q in points:
# 		cv2.circle(img, toCV(Q), 4, (0.5, 0.5, 1), -1)

# cv2.imshow("img", img)
# cv2.waitKey()
# exit()

# cv2.imshow("img", img)
# cv2.waitKey()
# exit()



######## GENERATE LEAFS ########
leafs = []
i = 0
while len(leafs) < NPOINTS and i < NPOINTS*20:
	## Generate random leaf somewhere in the tree
	P = generatePoint(radius=300, sigma=40) + O

	## Check if its not too close to another leaf
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

edges = []

# Generate Delaunay triangulation
triangulation = bowyerWatson(leafs)

### Remove all triangles from triangulation whose circumcentre does not lie within another triangle ###
_triangulation = triangulation
triangulation = []
for T in _triangulation:
	if any([_T.containsPoint(T.C) for _T in _triangulation]):
		triangulation.append(T)
### Remove all from triangulation whose center does not lie within another triangle ###

### Add all possible branches as edges ###
for T in triangulation:
	for _T in triangulation:
		if T.sharesVertexWith(_T):
			# Prevent duplicate edges, since line A-B != line B-A
			if (T.C, _T.C) not in edges and (_T.C, T.C) not in edges: 
				edges.append((T.C, _T.C))
print("%d voronoi edges added" % len(edges))
### Add all possible branches as edges ###

print("Drawing %d edges" % len(edges))
for (v1, v2) in edges:
	cv2.line(img, toCV(v1), toCV(v2), (0, 0, 1), 14)

### Add edges from leafs to branches ###
print("\nAdd edges from leafs to branches")
leafPaths = []
for L in leafs:
	minDist, minProj, minLine = float("inf"), None, None
	for P in edges: # Find edge closest to leaf
		V, W = P
		proj = projectPointOntoSegment(L, V, W)
		dist = distancePointToSegment(L, V, W)
		if dist < minDist:
			minDist, minProj, minLine = dist, proj, P
	
	## Split up edge V-W into edges V-proj, proj-W
	if minProj not in minLine:
		edges.remove(minLine) 					# Remove original segment
		edges.append((minLine[0], minProj))		# Add first subsegment
		edges.append((minProj, minLine[1]))		# Add second subsegment
	leafPaths.append((L, minProj))			# Add line from leaf to segment
edges += leafPaths
### Add edges from leafs to branches ###

### Add edge from root node to closest voronoi node ###
distances = [distance(rootNode, T.C) for T in triangulation]
iClosest = distances.index(min(distances))
edges.append((rootNode, triangulation[iClosest].C))

## Draw all edges
print("Drawing %d edges" % len(edges))
for (v1, v2) in edges:
	cv2.line(img, toCV(v1), toCV(v2), (np.random.rand(), np.random.rand(), np.random.rand()), 3)

cv2.imshow("img", img)
cv2.waitKey()	
######## PATHS GENERATED ########



######## FILTER PATHS ########
img = np.ones((HEIGHT, WIDTH, 3))

# Get unique list of all nodes (leafs, voronoi, rootnode)
nodes = list(set([val for sublist in edges for val in sublist]))
# Get routes from all leafs to root node
routes = dijkstra(list(nodes), list(edges), rootNode)
## Filter out all unused edges
edges = []
for node in leafs:
	## Create route for node
	route = []
	previous = node
	while routes[previous] != None:
		route.append(previous)
		previous = routes[previous]
	route.append(rootNode)

	## Add each edge of route to edges
	for i in range(len(route)-1):	
		P = (route[i], route[i+1])
		if P not in edges:
			edges.append(P)

for leaf in leafs:
	drawLeaf(img, leaf)
for edge in edges:
	e1, e2 = toCV(edge[0]), toCV(edge[1])
	cv2.line(img, e1, e2, (0, 0, 0), 1)
	cv2.circle(img, e1, 2, (0, 0, 1), -1)
	cv2.circle(img, e2, 2, (0, 0, 1), -1)

lengths = [(V-W).norm() for V, W in edges]
lengths.sort()
print(lengths)

for V, W in edges:
	if (V-W).norm() < 10:
		center = (V+W) * 0.5
		cv2.circle(img, toCV(center), 14, (0, 0, 1), 3)


cv2.imshow("img", img)
cv2.waitKey()	

for edge in edges:
	p1, p2 = getPointsFromEdge(edge, mm(1))
	cv2.circle(img, toCV(p1), 2, (0.5, 0, 1), -1)
	cv2.circle(img, toCV(p2), 2, (0.5, 0, 1), -1)

for node in nodes:
	points = getPointsFromNode(edges, node, mm(2), img)
	for Q in points:
		cv2.circle(img, toCV(Q), 2, (0, 0.5, 1), -1)

cv2.imshow("img", img)
cv2.waitKey()	

exit()





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

img = np.ones((HEIGHT, WIDTH, 3))
for leaf in leafs:
	drawLeaf(img, leaf)

for N in nodes:
	neighbours = []
	neighbours = [edge for edge in edges if N in edge]
	if len(neighbours) < 2:
		continue
	print(len(neighbours))

	perms = list(itertools.permutations(neighbours, 2))
	
	for e1, e2 in perms:
		p1 = otherCorner(e1, N)
		p2 = otherCorner(e2, N)

		print(N, p1, p2)

		# cv2.circle(img, toCV(p1), 5, (1, 0, 0), -1)
		# cv2.circle(img, toCV(p2), 5, (1, 0, 0), -1)
		# cv2.circle(img, toCV(N), 5, (1, 0, 0), -1)

		nPoints = 3
		points = [p1, N, p2]
		xpoints = [p[0] for p in points]
		ypoints = [p[1] for p in points]

		xvals, yvals = bezier_curve(points, nTimes=1000)

		z = zip(xvals, yvals)
		for x, y in z:
			x, y = int(x), int(y)
			try:
				img[y-1:y+2, x-1:x+1] = (1, 0.5, 0)
			except:
				pass
cv2.imshow("img", img)
cv2.waitKey()
