import cv2
import numpy as np
import math
import itertools
from functools import reduce
from vector import Vector as Vec
from triangle import Triangle
from functions import lineFromPoints, perpendicularBisectorFromLine, lineLineIntersection, distance, toCV, distancePointToSegment

# 90mm = 500px -> 1mm = 5.5px
def mm(mm): # configured for 15.6" 1920x1080
	return int(mm * 5.5586)

NPLATES = 4
SCALAR = 1 / math.sqrt(NPLATES)
print("Scale : 1px=%0.2fmm" % SCALAR)
# scale : 1px = 1mm
W = 1000
H = 600
R = int(mm(5) * SCALAR)

NPOINTS = 100
img = np.ones((H, W, 3))

def sampleFromNormal():
	M = 100
	xi = np.random.rand(M)
	x = (xi.sum() - M/2) / np.sqrt(M/12)
	return x

def sampleFromNormal2D(x=0, y=0, sigma=25):
	return np.array([sigma*sampleFromNormal()+x, sigma*sampleFromNormal()+y])

def rotate(x, y, r):
	return np.array([math.cos(r) * x - math.sin(r) * y, math.sin(r) * x + math.cos(r) * y])

def generatePoint(radius=200, mu=0, sigma=25):
	r = np.random.rand() * (2*math.pi * 2/4) - (2*math.pi * 1/4)
	p = rotate(0, -radius, r) + sampleFromNormal2D(sigma=sigma)
	return p

def inPlane(x, y, r, w, h):
	return 0 < x-r and 0 < y-r and x+r < w and y+r < h

def drawLeaf(img, vec):
	cv2.circle(img, toCV(vec), R, (0, 0.5, 0), mm(1), cv2.LINE_AA)

def drawTriangle(img, T, colour=(0, 0, 0)):
	cv2.line(img, toCV(T.v1), toCV(T.v2), colour, 1)
	cv2.line(img, toCV(T.v2), toCV(T.v3), colour, 1)
	cv2.line(img, toCV(T.v3), toCV(T.v1), colour, 1)

O = np.array([W//2, H - H//5])


#### GENERATE LEAFS
leafs = []
i = 0
while len(leafs) < NPOINTS and i < NPOINTS*20:
	x, y = generatePoint(radius=300, sigma=40) + O
	x, y = int(x), int(y)
	v = Vec(x, y)

	distances = [distance(v, l) for l in leafs]
	if 0 == len(list(filter(lambda x : x < 2 * R + mm(5), distances))):
		if inPlane(x, y, R, W, H):
			leafs.append(v)
			drawLeaf(img, v)
	i += 1
#### LEAFS GENERATED
print(len(leafs), "leafs added")

# cv2.imshow("img", img)
# cv2.waitKey()	

#### GENERATE PATHS ####
combs = list(itertools.combinations(leafs, 3))
print(len(combs), "possible triangles")

def plotLine(img, a, b, c, colour=(0, 0, 0)):
	# ax + by + c = 0 -> by = -ax - c -> y = (-ax - c) / b
	P = Vec(-100, (-a * -100 - c) / b)
	Q = Vec(1100, (-a * 1100 - c) / b)
	cv2.line(img, toCV(P), toCV(Q), colour, 2)

def bowyerWatson(points):
	# https://en.wikipedia.org/wiki/Bowyer%E2%80%93Watson_algorithm
	print("Running bowyerWatson on %d points" % len(points))
	triangulation = []
	# must be large enough to completely contain all the points in pointList
	megaTriangle = Triangle(Vec(-3000, -3000), Vec(3000, -3000), Vec(0, 3000))

	triangulation.append(megaTriangle)
	
	# add all the points one at a time to the triangulation
	for iP, P in enumerate(points): 
		
		badTriangles = []
		# first find all the triangles that are no longer valid due to the insertion
		for iT, T in enumerate(triangulation):
			if distance(P, T.C) < T.R: # If point inside triangle circumcircle
				badTriangles.append(T) # Triangle is bad
		
		# find the boundary of the polygonal hole
		polygon = []
		for T in badTriangles: 
			for V in T.vertices: # for each edge in triangle
				# if edge is not shared by any other triangles in badTriangles
				if not any([_T.hasVertex(V) for _T in badTriangles if T != _T]):
					polygon.append(V)

		for T in badTriangles:
			triangulation.remove(T)
		
		# re-triangulate the polygonal hole
		for (v1, v2) in polygon:
			triangulation.append(Triangle(P, v1, v2))
	
	# if triangle contains a vertex from original super-triangle
	triangulation = [T for T in triangulation if not T.sharesCornerWith(megaTriangle)]

	return triangulation

# Generate Delaunay triangulation
triangulation = bowyerWatson(leafs)
# for T in triangulation:
# 	drawTriangle(img, T)

# Remove all from triangulation whose center does not lie within another triangle
_triangulation = triangulation
triangulation = []
for T in _triangulation:
	if any([_T.containsPoint(T.C) for _T in _triangulation]):
		triangulation.append(T)

paths = []

## Draw all paths
for T in triangulation:
	for _T in triangulation:
		if T.sharesVertexWith(_T):
			paths.append([T.C, _T.C])
			cv2.line(img, toCV(T.C), toCV(_T.C), (0, 0, 1), 2)

for L in leafs:
	minDistance, minProj = 10000, None
	for V, W in paths:
		# Find projection of p onto w, by first moving everything to (0, 0) (aka subtracting v)
		f = ((L-V) * (W-V)) / ((W-V)*(W-V))
		# Clamp projection between [0, 1]
		t = max(0, min(f, 1))
		# Calculate the projection
		proj = V + t * (W-V)
		if distance(L, proj) < minDistance:
			minDistance = distance(L, proj)
			minProj = proj
	cv2.line(img, toCV(L), toCV(minProj), (0, 0, 1), 2)

cv2.imshow("img", img)
cv2.waitKey()	


exit()

