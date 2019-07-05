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

def rotateList(l, n):
    return l[-n:] + l[:-n]

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

def drawTriangle(img, T, colour=(0, 0, 0), stroke=1):
	cv2.line(img, toCV(T.v1), toCV(T.v2), colour, stroke)
	cv2.line(img, toCV(T.v2), toCV(T.v3), colour, stroke)
	cv2.line(img, toCV(T.v3), toCV(T.v1), colour, stroke)

O = np.array([WIDTH//2, HEIGHT - HEIGHT//5])
rootNode = Vec(WIDTH//2, HEIGHT-1)
cv2.circle(img, toCV(rootNode), 8, (1, 0, 1), -1)



img = np.ones((HEIGHT, WIDTH, 3))
LC = Vec(WIDTH//2, HEIGHT//2)

L1 = Vec(WIDTH//2, HEIGHT//4)
L2 = Vec(3*WIDTH//4, HEIGHT//4)
L3 = Vec(WIDTH//3, 3*HEIGHT//4)
L4 = Vec(2*WIDTH//3, 3*HEIGHT//4)
L5 = Vec(WIDTH//2 + 150, HEIGHT//2)
LC2= Vec(1*WIDTH//4, 2*HEIGHT//4)
E1 = (LC, L1)
E2 = (LC, L2)
# E3 = (LC, L3)
E4 = (LC, L4)
E5 = (LC, L5)
E6 = (LC2,L1)
E7 = (LC2,L3)
leafs = [LC, L1, L2, L3, L4, LC2]
edges = [E1, E2, E4, E5, E6, E7]

def getPointsFromEdge(edge, dist=10):
	V, W = edge
	center = (V+W) * 0.5
	p1 = (dist*(V-W)*(1/(V-W).norm())).rotate(0.5*math.pi) + center
	p2 = (dist*(W-V)*(1/(W-V).norm())).rotate(0.5*math.pi) + center
	return p1, p2

def getPointsFromNode(edges, P, dist, img):
	print("\ngetPointsFromNode")
	# Retrieve all neighbouring points
	neighbours = [otherCorner(edge, P) for edge in edges if P in edge]
	# Convert all neighbours from cartesian to polar
	polar = [(n-P).polar() for n in neighbours]
	# Create list (cartesian, polar)
	cartPolarPairs = list(zip(neighbours, polar))
	# Sort list by polar angle
	cartPolarPairs.sort(key=lambda x: x[1][1])
	# Create neighbouring pairs [(cartesian, polar), (cartesian, polar)]
	adjacentPairs = list(zip(cartPolarPairs, rotateList(cartPolarPairs, 1)))
	
	# For each neighbouring pair, create triangle and calculate angle
	triangles = []
	for pair1, pair2 in adjacentPairs:
		cart1, cart2 = pair1[0], pair2[0]
		angle = pair1[1][1] - pair2[1][1]
		if angle < 0:
			angle += 2*math.pi
		triangles.append((Triangle(P, cart1, cart2), angle))

	points = []
	for (T, angleT) in triangles:
		# drawTriangle(img, T)
		# for x in T.corners:
		# 	cv2.circle(img, toCV(x), R, (0, 1, 0), mm(2))

		# https://www.mathsisfun.com/algebra/trig-solving-sss-triangles.html
		V, W = [c for c in T.corners if c != P]				# Get other corners
		a, b, c = (V-W).norm(), (P-V).norm(), (P-W).norm()  # Get lengths of edges
		cosA = (b**2 + c**2 - a**2) / (2 * b * c)			# No idea
		if cosA < -1 or 1 < cosA:							# Sanity check
			print("[getPointsFromNode] cosA out of range. Something is going wrong here..")
			return []
		angle = math.acos(cosA)								# Get angle between the two outgoing edges
		
		# Rotate the other way if pi < angle
		if abs(angleT) < math.pi:							
			Q = (V-P).rotate(angle/2)
		else:
			Q = (V-P).rotate(math.pi - angle/2)
		# scale Q to dist and add to P	
		Q = P + 1/Q.norm() * Q * dist
		
		points.append(Q)
	return points

for leaf in leafs:
	cv2.circle(img, toCV(leaf), R, (0.5, 1, 0), mm(1))
for V, W in edges:
	cv2.line(img, toCV(V), toCV(W), (1, 0.5, 0), mm(1))

for edge in edges:
	p1, p2 = getPointsFromEdge(edge)
	cv2.circle(img, toCV(p1), 4, (0.5, 0.5, 1), -1)
	cv2.circle(img, toCV(p2), 4, (0.5, 0.5, 1), -1)

for P in [LC, LC2]:
	points = getPointsFromNode(edges, P, R, img)
	for Q in points:
		cv2.circle(img, toCV(Q), 4, (1, 0, 0.5), -1)

cv2.imshow("img", img)
cv2.waitKey()
exit()

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

leafCombs = list(itertools.combinations(leafs, 2))
distances = [(V-W).norm() for V, W in leafCombs]
distances.sort()
print("Minimal distance between leaves:", min(distances))
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
for V, W in edges:
	cv2.line(img, toCV(V), toCV(W), (0, 0, 1), 14)

distances = [(V-W).norm() for V, W in edges]
distances.sort()
print("Minimal distance before branches:", min(distances))

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
	
	### This fixes problem where difference between minProj and minLine[x] = 2.8e-14
	### For some reason the projection in projectPointOntoSegment might be extremely slightly off, probably due to float imprecision 
	### Example: (809.5, 337.50000000000006) != (809.5, 337.49999999999994)
	if distance(minProj, minLine[0]) < 1: # 1 is chosen because of 1 pixel!
		minProj = minLine[0]
	if distance(minProj, minLine[1]) < 1: # 1 is chosen because of 1 pixel!
		minProj = minLine[1]

	## Split up edge V-W into edges V-proj, proj-W
	if minProj not in minLine:
		edges.remove(minLine) 					# Remove original segment
		edges.append((minLine[0], minProj))		# Add first subsegment
		edges.append((minProj, minLine[1]))		# Add second subsegment
	leafPaths.append((L, minProj))			# Add line from leaf to segment
edges += leafPaths
### Add edges from leafs to branches ###

distances = [(V-W).norm() for V, W in edges]
distances.sort()
print("Minimal distance  after branches:", min(distances))

### Add edge from root node to closest voronoi node ###
distances = [distance(rootNode, T.C) for T in triangulation]
iClosest = distances.index(min(distances))
edges.append((rootNode, triangulation[iClosest].C))

distances = [(V-W).norm() for V, W in edges]
distances.sort()
print("Minimal distance  after root:", min(distances))

## Replace all branches smaller than 10
edgesRemoved = 0
for i, (V, W) in enumerate(edges):
	# If edge is too small, replace all occurences of W with V and delete edge
	if distance(V, W) < R:
		for j, (P, Q) in enumerate(edges):
			if P == W:
				edges[j] = (V, Q)
			if Q == W:
				edges[j] = (P, V)
		edgesRemoved += 1
## Remove all edges that changed from (V, W) to (V, V), aka with a length of 0.0
edges = [edge for edge in edges if distance(edge[0], edge[1]) != 0]
print("Edges removed:", edgesRemoved)

distances = [(V-W).norm() for V, W in edges]
distances.sort()
print("Minimal distance after replacement:", min(distances))

## Draw all edges
print("Drawing %d edges" % len(edges))
for (v1, v2) in edges:
	cv2.line(img, toCV(v1), toCV(v2), (np.random.rand(), np.random.rand(), np.random.rand()), 3)

# cv2.imshow("img", img)
# cv2.waitKey()	
######## PATHS GENERATED ########



######## FILTER EDGES ########
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

distances = [(V-W).norm() for V, W in edges]
distances.sort()
print("Minimal distance after filtering:", min(distances))

for leaf in leafs:
	drawLeaf(img, leaf)
for V, W in edges:
	e1, e2 = toCV(V), toCV(W)
	cv2.line(img, e1, e2, (0, 0, 0), 1)
	cv2.circle(img, e1, 2, (0, 0, 1), -1)
	cv2.circle(img, e2, 2, (0, 0, 1), -1)

cv2.imshow("img", img)
cv2.waitKey()	
######## EDGES FILTERED ########



for edge in edges:
	p1, p2 = getPointsFromEdge(edge, mm(1))
	cv2.circle(img, toCV(p1), 2, (0.5, 0, 1), -1)
	cv2.circle(img, toCV(p2), 2, (0.5, 0, 1), -1)

cv2.imshow("img", img)
cv2.waitKey()	

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
