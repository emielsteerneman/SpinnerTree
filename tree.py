import cv2
import numpy as np
from scipy.special import comb
import math
import itertools
from functools import reduce
from vector import Vector as Vec
from triangle import Triangle
from functions import lineFromPoints, perpendicularBisectorFromLine, lineLineIntersection, distance, toCV, distancePointToSegment, projectPointOntoSegment, otherCorner, bernstein_poly, bezier_curve, intersectionOfSegments

from dijkstra import dijkstra
from bowyerWatson import bowyerWatson

import cairo

# 90mm = 500px -> 1mm = 5.5px
def mm(mm):
	# return int(mm * 5.5586) # configured for 15.6" 1920x1080
	return int(mm * 3.6460) # configured for 23.8" 1920x1080

WIDTH = 1000
HEIGHT = 600
R = int(mm(5))

NPOINTS = 80
img = np.ones((HEIGHT, WIDTH, 3))

def rotateList(l, n):
    return l[-n:] + l[:-n]

def sampleFromNormal():
	M = 100 # number of samples
	xi = np.random.rand(M)
	x = (xi.sum() - M/2) / np.sqrt(M/12)
	return x

def sampleFromNormal2D(x=0, y=0, sigma=25):
	px = sigma * sampleFromNormal() + x
	py = sigma * sampleFromNormal() + y
	return Vec(px, py)

def generatePoint(radius=200, mu=0, sigma=25):
	r = np.random.rand() * (2*math.pi * 2/4) - (2*math.pi * 1/4) # Rotation somewhere between -.5pi and .5pi
	p = Vec(0, -radius).rotate(r) + sampleFromNormal2D(sigma=sigma) # Rotate vector, add random offset sampled from normal distribution
	return p

def inPlane(P, r, w, h):
	return 0 < P[0]-r and 0 < P[1]-r and P[0]+r < w and P[1]+r < h

def drawLeaf(img, vec):
	cv2.circle(img, toCV(vec), R, (0, 0.5, 0), mm(1), cv2.LINE_AA)

O = np.array([WIDTH//2, HEIGHT - HEIGHT//5])
rootNode = Vec(WIDTH//2, HEIGHT-1)

def getPointsFromEdge(edge, dist=10):
	V, W = edge
	center = (V+W) * 0.5
	p1 = (dist*(V-W)*(1/(V-W).norm())).rotate(0.5*math.pi) + center
	p2 = (dist*(W-V)*(1/(W-V).norm())).rotate(0.5*math.pi) + center
	return p1, p2

def getConnectingPairsFromNode(edges, N):
	# Retrieve all neighbouring points
	neighbours = [otherCorner(edge, N) for edge in edges if N in edge]
	# Convert all neighbours from cartesian to polar
	polar = [(n-N).polar() for n in neighbours]
	# Create list (cartesian, polar)
	cartPolarPairs = list(zip(neighbours, polar))
	# Sort list by polar angle
	cartPolarPairs.sort(key=lambda x: x[1][1])
	# Create neighbouring pairs [(cartesian, polar), (cartesian, polar)]
	adjacentPairs = list(zip(cartPolarPairs, rotateList(cartPolarPairs, -1)))

	return adjacentPairs

def getPointsFromNode(edges, P, dist, img):
	print("\ngetPointsFromNode")
	adjacentPairs = getConnectingPairsFromNode(edges, P)

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


def getPointsFromPairs(N, P1, P2):

	(V, (r1, a1)), (W, (r2, a2)) = (P1, P2)


	factorV = 1 - ((N+V)*0.5 - rootNode).norm() / (HEIGHT)
	factorW = 1 - ((N+W)*0.5 - rootNode).norm() / (HEIGHT)

	sizeV = max([mm(1), mm(3)*factorV])
	sizeW = max([mm(1), mm(3)*factorW])

	### Get the point for the center node N, inbetween edges V and W
	# https://www.mathsisfun.com/algebra/trig-solving-sss-triangles.html
	a, b, c = (V-W).norm(), (N-V).norm(), (N-W).norm()  # Get lengths of edges
	cosA = (b**2 + c**2 - a**2) / (2 * b * c)			# Get angle between V and W
	cosA = max(-1, min(cosA, 1))						# Sanity check. Sometimes, cosA is equal to something like -1.0000000000000004
	angle = math.acos(cosA)								# Get angle between the two edges

	if a2-a1 < 0:
		a2 += 2*math.pi

	Q = (V-N).rotate(-angle/2)
	# Rotate the other way if the two edges are concave (if pi < angle)
	if math.pi < abs(a2-a1):
		Q = (W-N).rotate(math.pi-angle/2)
	# scale Q to dist and add to N
	Q = N + 1/Q.norm() * Q * 5

	Vc = (V + N) * 0.5 # center of V
	V1 = sizeV * ((V-N) / (V-N).norm()).rotate(0.5*math.pi) # Point on one side of V
	V2 = sizeV * ((N-V) / (N-V).norm()).rotate(0.5*math.pi) # Point on other side of V

	Wc = (W + N) * 0.5 # center of W
	W1 = sizeW * ((W-N) / (W-N).norm()).rotate(0.5*math.pi) # Point on one side of W
	W2 = sizeW * ((N-W) / (N-W).norm()).rotate(0.5*math.pi) # Point on other side of W

	combs = [(V1, W1), (V1, W2), (V2, W1), (V2, W2)]
	combs.sort(key=lambda E : distance(E[0] + Vc, Q) + distance(E[1] + Wc, Q))
	
	Vp, Wp = combs[0]

	dV = (N + Vp, V + Vp)
	dW = (N + Wp, W + Wp)
	I = intersectionOfSegments(dV, dW)

	if I is None:
		return Vc+Vp, Q, Wc+Wp

	return Vc+Vp, I, Wc+Wp

def drawTree(edges=None, nodes=None, triangles=None, triangleCenters=False, img=None):
	showImg = img == None
	if img == None:
		img = np.ones((HEIGHT, WIDTH, 3)) * 0.1

	if edges != None:
		for V, W in edges:
			cv2.line(img, toCV(V), toCV(W), (0, 0.5, 1), 1, cv2.LINE_AA)

	if nodes != None:
		for N in nodes:
			cv2.circle(img, toCV(N), R, (1, 0, 0.5), 1, cv2.LINE_AA)

	if triangles != None:
		for T in triangles:
			cv2.line(img, toCV(T.v1), toCV(T.v2), (0.5, 1, 0), 1, cv2.LINE_AA)
			cv2.line(img, toCV(T.v2), toCV(T.v3), (0.5, 1, 0), 1, cv2.LINE_AA)
			cv2.line(img, toCV(T.v3), toCV(T.v1), (0.5, 1, 0), 1, cv2.LINE_AA)
			if triangleCenters:
				cv2.circle(img, toCV(T.C), 4, (1, 0.5, 0), -1, cv2.LINE_AA)

	# if showImg:
	# 	cv2.imshow("Tree", img)
	# 	cv2.waitKey()

	return img	

def drawTreeSVG(edges=None, nodes=None, triangles=None, triangleCenters=False):
	surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
	surface = cairo.SVGSurface("img.svg", WIDTH, HEIGHT)
	ctx = cairo.Context(surface)

	if edges != None:
		ctx.set_line_width(1)
		for (x1,y1), (x2, y2) in edges:
			ctx.new_sub_path()
			ctx.move_to(x1, y1)
			ctx.line_to(x2, y2)
			ctx.stroke()

	if nodes != None:
		ctx.set_line_width(1)
		for x,y in nodes:
			ctx.new_sub_path()
			ctx.arc(x, y, R, 0, 2*math.pi)
			ctx.stroke()

	surface.write_to_png("example.png")





# S1 = (Vec(150, 0), Vec(100, 250))
# S2 = (Vec(150, 300), Vec(300, 300))

# img = np.ones((400, 400, 3)) * 0.1
# cv2.line(img, toCV(S1[0]), toCV(S1[1]), (1, 1, 1), 1)
# cv2.line(img, toCV(S2[0]), toCV(S2[1]), (1, 1, 1), 1)

# Q = intersectionOfSegments(S1, S2)

# cv2.circle(img, toCV(Q), 10, (0, 0, 1), 2)
# cv2.imshow("img", img)
# cv2.waitKey()

# getPointsFromPairs(Q, S1, S2)

# exit()




######## GENERATE LEAFS ########
leafs = []
i = 0
while len(leafs) < NPOINTS and i < NPOINTS*200:
	i += 1
	# Generate random leaf somewhere in the tree
	N = generatePoint(radius=HEIGHT/2, sigma=2*R) + O
	# Check if point in plane
	if not inPlane(N, R, WIDTH, HEIGHT):
		continue
	# Check if its not too close to another leaf
	minDistance = 3 * R
	distances = [distance(N, L) < minDistance for L in leafs]
	if not any(distances):
		leafs.append(N)
print("Generated %d/%d leafs" % (len(leafs), NPOINTS))

leafCombs = list(itertools.combinations(leafs, 2))
distances = [(V-W).norm() for V, W in leafCombs]
print("Minimal distance between leaves: %0.2f" % min(distances))
######## LEAFS GENERATED ########





drawTree(nodes=leafs)





######## GENERATE PATHS ########
edges = []

# Generate Delaunay triangulation
triangulation = bowyerWatson(leafs)

drawTree(nodes=leafs, triangles=triangulation, triangleCenters=True)

### Remove all triangles from triangulation whose circumcentre does not lie within another triangle ###
### This is done to make sure that there will be no branches outside of the tree ###
_triangulation = triangulation
triangulation = []
for T in _triangulation:
	if any([_T.containsPoint(T.C) for _T in _triangulation]):
		triangulation.append(T)
### Remove all from triangulation whose center does not lie within another triangle ###

drawTree(nodes=leafs, triangles=triangulation, triangleCenters=True)

### Add all possible branches as edges ###
for T in triangulation:
	for _T in triangulation:
		if T.sharesVertexWith(_T):
			# Prevent duplicate edges, since line A-B != line B-A
			if (T.C, _T.C) not in edges and (_T.C, T.C) not in edges: 
				edges.append((T.C, _T.C))
print("%d voronoi edges added" % len(edges))
### Add all possible branches as edges ###

drawTree(edges=edges, nodes=leafs, triangles=triangulation, triangleCenters=True)
drawTree(edges=edges, nodes=leafs)

distances = [(V-W).norm() for V, W in edges]
print("Minimal distance before branches: %0.2f" % min(distances))

### Add edges from leafs to branches ###
print("\nAdd edges from leafs to branches")
branches = []
for L in leafs:
	### Find the edge that is closest to the leaf
	minDist, closestProj, closestEdge = float("inf"), None, None
	for (V, W) in edges: # For each edge
		proj = projectPointOntoSegment(L, V, W) # Get projection on branch
		dist = distancePointToSegment(L, V, W)  # Get distance to branch
		if dist < minDist:
			minDist, closestProj, closestEdge = dist, proj, (V, W)
	
	### This fixes problem where difference between closestProj and closestEdge[x] = 2.8e-14
	### For some reason the projection in projectPointOntoSegment might be extremely slightly off, probably due to float imprecision 
	### Example: (809.5, 337.50000000000006) != (809.5, 337.49999999999994)
	if distance(closestProj, closestEdge[0]) < 1: # 1 is chosen because of 1 pixel!
		closestProj = closestEdge[0]
	if distance(closestProj, closestEdge[1]) < 1: # 1 is chosen because of 1 pixel!
		closestProj = closestEdge[1]

	## Split up edge V-W into edges V-proj, proj-W
	if closestProj not in closestEdge: # If line has to be split up
		edges.remove(closestEdge) 					# Remove original segment
		edges.append((closestEdge[0], closestProj))		# Add first subsegment
		edges.append((closestProj, closestEdge[1]))		# Add second subsegment
	
	branches.append((L, closestProj))			# Add line from leaf to segment
edges += branches
### Add edges from leafs to branches ###

drawTree(edges=edges, nodes=leafs)

distances = [(V-W).norm() for V, W in edges]
print("Minimal distance  after branches: %0.2f" % min(distances))

### Add edge from root node to closest voronoi node ###
distances = [distance(rootNode, T.C) for T in triangulation]
iClosest = distances.index(min(distances))
leafs.append(rootNode)
edges.append((rootNode, triangulation[iClosest].C))

drawTree(edges=edges, nodes=leafs)

distances = [(V-W).norm() for V, W in edges]
print("Minimal distance  after root: %0.2f" % min(distances))

## Replace all branches smaller than R
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
## Remove all edges that changed from (V, W) to (V, V), aka with a length of 0
edges = [edge for edge in edges if distance(edge[0], edge[1]) != 0]
print("Edges removed:", edgesRemoved)

distances = [(V-W).norm() for V, W in edges]
print("Minimal distance after replacing branches < %0.2f:" % R, min(distances))
######## PATHS GENERATED ########


# drawTreeSVG(edges=edges, nodes=leafs)
# exit()


drawTree(edges=edges, nodes=leafs)





######## FILTER EDGES ########
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
print("Minimal distance after filtering:", min(distances))


nodes = list(set([val for sublist in edges for val in sublist]))
######## EDGES FILTERED ########





drawTree(edges=edges, nodes=leafs)





nodes.sort(key=lambda node : node.norm())

leafs = []
for N in nodes:
	neighbours = [edge for edge in edges if N in edge]
	if len(neighbours) == 1: # If node is a leaf
		leafs.append(N)


surface = cairo.SVGSurface("after.svg", WIDTH, HEIGHT)
ctx = cairo.Context(surface)

ctx.set_source_rgb(0.1, 0.1, 0.1)
ctx.rectangle(0, 0, WIDTH, HEIGHT)
ctx.fill()
ctx.set_source_rgb(1, 1, 1)
ctx.set_line_width(2)

for NODE in nodes:

	if NODE in leafs:
		ctx.new_sub_path()
		ctx.arc(NODE[0], NODE[1], R/2, 0, 2*math.pi)
		ctx.stroke()
		continue

	pairs = getConnectingPairsFromNode(edges, NODE)

	# for ((V, _), (W, _)) in pairs:
	# 	ctx.new_sub_path()
	# 	ctx.move_to(NODE[0], NODE[1])	
	# 	ctx.line_to(V[0], V[1])
	# 	ctx.close_path()

	beginx, beginy = pairs[0][0][0]
	beginx, beginy = (NODE[0]+beginx)/2, (NODE[1]+beginy)/2
	ctx.new_sub_path()
	ctx.move_to(beginx, beginy)
	for iPair, (P1, P2) in enumerate(pairs):
		((x1, y1), (r1, a1)), ((x2, y2), (r2, a2)) = (P1, P2)

		points = getPointsFromPairs(NODE, P1, P2)
		if points is None:
			continue

		(dVx, dVy), (Qx, Qy), (dWx, dWy) = points

		x, y = (NODE[0]+x2)/2, (NODE[1]+y2)/2

		# ctx.line_to(Q[0], Q[1])
		if iPair == 0:
			ctx.move_to(dVx, dVy)
		else:
			ctx.line_to(dVx, dVy)
		ctx.curve_to(Qx, Qy, Qx, Qy, dWx, dWy)
		

		# ctx.line_to(Qx, Qy)
		# ctx.line_to(dWx, dWy)


		# ctx.set_line_width(4)
		# ctx.arc(NODE[0], NODE[1], 2, 0, 2*math.pi)
		# ctx.stroke()		
		# ctx.set_line_width(2)

	ctx.close_path()
ctx.stroke()

surface.write_to_png("after.png")


exit()


















































img = np.ones((HEIGHT, WIDTH, 3)) * 0.1

for N in nodes:
	if N in leafs:
		continue

	pairs = getConnectingPairsFromNode(edges, N)
	for (V, (r1, a1)), (W, (r2, a2)) in pairs:

		factor1 = 1 - ((N+V)*0.5 - rootNode).norm() / (HEIGHT)
		factor2 = 1 - ((N+W)*0.5 - rootNode).norm() / (HEIGHT)
		factorN = max([factor1, factor2])

		size1 = max([mm(1), mm(3)*factor1])
		size2 = max([mm(1), mm(3)*factor2])
		sizeN = max([mm(1), mm(3)*factorN])

		### Get the point for the center node N, inbetween edges V and W
		# https://www.mathsisfun.com/algebra/trig-solving-sss-triangles.html
		a, b, c = (V-W).norm(), (N-V).norm(), (N-W).norm()  # Get lengths of edges
		cosA = (b**2 + c**2 - a**2) / (2 * b * c)			# Get angle between V and W
		cosA = max(-1, min(cosA, 1))						# Sanity check. Sometimes, cosA is equal to something like -1.0000000000000004
		angle = math.acos(cosA)								# Get angle between the two edges

		if a2-a1 < 0:
			a2 += 2*math.pi

		Q = (V-N).rotate(-angle/2)
		# Rotate the other way if the two edges are concave (if pi < angle)
		if math.pi < abs(a2-a1):
			Q = (W-N).rotate(math.pi-angle/2)
		# scale Q to dist and add to N
		Q = N + 1/Q.norm() * Q * (sizeN)

		V1, V2 = getPointsFromEdge((N, V), size1)
		W1, W2 = getPointsFromEdge((N, W), size2)



		if V in leafs:
			V1 = V - ((V-N) * (1/(V-N).norm()) * R).rotate(2*math.pi/20)
			V2 = V - ((V-N) * (1/(V-N).norm()) * R).rotate(-2*math.pi/20)

		if W in leafs:
			W1 = W - ((W-N) * (1/(W-N).norm()) * R).rotate(2*math.pi/20)
			W2 = W - ((W-N) * (1/(W-N).norm()) * R).rotate(-2*math.pi/20)

		dV = [V1, V2][np.argmin([distance(Q,V1), distance(Q,V2)])]
		dW = [W1, W2][np.argmin([distance(Q,W1), distance(Q,W2)])]

		xvals, yvals = bezier_curve([dV, Q, dW], nTimes=10000)
		z = zip(xvals, yvals)
		for x, y in z:
			img[int(y), int(x)] = (0, 0.5, 1)
			# cv2.circle(img, (int(x), int(y)), 1, (0, 0.5, 1), -1)
			# img[y-1:y+2, x-1:x+1] = (0, 0.5, 1)

for N in leafs:
	P, Q = [edge for edge in edges if N in edge][0]
	V = P if N == Q else Q
	angle = (V-N).polar()[1] * (180 / math.pi)

	cv2.circle(img, toCV(N), R, (1, 0, 0.5), 2, cv2.LINE_AA)
	# cv2.ellipse(img, toCV(N), (int(mm(2)), int(mm(2))), 0, 90-angle+15, 90-angle+360-15, (1, 0, 0.5), 2, cv2.LINE_AA)

	# cv2.circle(img, toCV(N), R - mm(0), (1, 0, 0.5), 1, cv2.LINE_AA)
	# cv2.ellipse(img, toCV(N), (int(R), int(R)), 0, 90-angle+15, 90-angle+360-15, (1, 0, 0.5), 1, cv2.LINE_AA)
	# cv2.putText(img, "%0.2f" % angle, toCV(N), cv2.FONT_HERSHEY_SIMPLEX, 1, (1, 1, 1))

# for edge in edges:
# 	if edge[0] in leafs or edge[1] in leafs:
# 		continue

# 	factor = 1 - ((edge[0] + edge[1])*0.5 - rootNode).norm() / HEIGHT
# 	size = max([mm(3), mm(5)*factor])

# 	V, W = getPointsFromEdge(edge, mm(3))
# 	Vs, Ws = getPointsFromEdge(edge, size)
# 	# cv2.line(img, toCV(V), toCV(W), (0, 0.5, 1), 1)	
# 	X = W-V
# 	points = []
# 	points.append(Vs)
# 	points.append(V + X*0.33)
# 	points.append(V + X*0.33)
# 	points.append(V + X*0.33)

# 	points.append(V + X.rotate(0.5*math.pi)*0.66)
# 	points.append(V + X.rotate(0.5*math.pi)*0.66)
# 	points.append(V + X.rotate(0.5*math.pi)*0.66)

# 	points.append(W + X.rotate(0.5*math.pi)*0.66)
# 	points.append(W + X.rotate(0.5*math.pi)*0.66)
# 	points.append(W + X.rotate(0.5*math.pi)*0.66)

# 	points.append(V + X*0.66)
# 	points.append(V + X*0.66)
# 	points.append(V + X*0.66)
# 	points.append(Ws)

# 	xvals, yvals = bezier_curve(points, nTimes=1000)
# 	z = zip(xvals, yvals)
# 	for x, y in z:
# 		# cv2.circle(img, (int(x), int(y)), 1, (0, 0.5, 1), -1)
# 		img[int(y), int(x)] = (0, 0.5, 1)

cv2.imwrite("before.png", img*255)
# cv2.imshow("img", img)
# cv2.waitKey()	
