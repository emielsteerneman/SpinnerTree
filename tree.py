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
	return mm
	# return mm * 5.5586 # configured for 15.6" 1920x1080
	# return mm * 3.6460 # configured for 23.8" 1920x1080

NPOINTS = 80
WIDTH = 4*1000
HEIGHT = 4*600
R = mm(50)

BRANCH_MIN_LENGTH = 1.5*R
BRANCH_MIN_WIDTH = mm(20)
BRANCH_MAX_WIDTH = mm(50)
LEAF_MIN_DISTANCE = 2.5 * R + BRANCH_MIN_WIDTH

RADIUS = 0.8 * HEIGHT // 2
SIGMA = HEIGHT/15

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
rootNode = Vec(WIDTH//2, HEIGHT-2*R)

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

def getPointsFromPairs(N, P1, P2, leafs, branchMin, branchMax):

	(V, (r1, a1)), (W, (r2, a2)) = (P1, P2)

	factorV = 1 - ((N+V)*0.5 - rootNode).norm() / (HEIGHT)
	factorW = 1 - ((N+W)*0.5 - rootNode).norm() / (HEIGHT)

	sizeV = max([0.5*branchMin, 0.5*branchMax*factorV])
	sizeW = max([0.5*branchMin, 0.5*branchMax*factorW])
	sizeN = (sizeV + sizeW) / 2

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
	Q = N + 1/Q.norm() * Q * sizeN

	## Get two points next to V
	Vc, V1, V2 = None, None, None
	if V not in leafs:
		Vc = (V + N) / 2 # center of V
		V1 = sizeV * ((V-N) / (V-N).norm()).rotate(0.5*math.pi) # Point on one side of V
		V2 = sizeV * ((N-V) / (N-V).norm()).rotate(0.5*math.pi) # Point on other side of V
	else:
		V1 = R * ((N-V) / (N-V).norm()).rotate(2*math.pi * 15/360)
		V2 = R * ((N-V) / (N-V).norm()).rotate(2*math.pi *-15/360)
		Vc = V

	## Get two points next to W
	Wc, W1, W2 = None, None, None
	if W not in leafs:
		Wc = (W + N) * 0.5 # center of W
		W1 = sizeW * ((W-N) / (W-N).norm()).rotate(0.5*math.pi) # Point on one side of W
		W2 = sizeW * ((N-W) / (N-W).norm()).rotate(0.5*math.pi) # Point on other side of W
	else:
		W1 = R * ((N-W) / (N-W).norm()).rotate(2*math.pi * 15/360)
		W2 = R * ((N-W) / (N-W).norm()).rotate(2*math.pi *-15/360)
		Wc = W

	## Get the two points closest to Q
	combs = [(V1, W1), (V1, W2), (V2, W1), (V2, W2)]
	combs.sort(key=lambda E : distance(E[0] + Vc, Q) + distance(E[1] + Wc, Q))
	
	Vp, Wp = combs[0]

	dV = (N + Vp, V + Vp)
	dW = (N + Wp, W + Wp)
	I = intersectionOfSegments(dV, dW)

	# img = np.ones((HEIGHT, WIDTH, 3)) * 0.1
	# if V in leafs:
	# 	cv2.line(img, toCV(N), toCV(V), (0, 0, 1), 1)
	# 	cv2.line(img, toCV(N), toCV(W), (0, 0, 1), 1)

	# 	cv2.circle(img, toCV(V), 1, (1, 0, 0), 2)
	# 	cv2.circle(img, toCV(V), R, (1, 0, 0), 1)
	# 	cv2.circle(img, toCV(V+V1), 1, (0, 1, 0), 2)
	# 	cv2.circle(img, toCV(V+V2), 1, (0, 1, 0), 2)

	# 	cv2.line(img, toCV(dV[0]), toCV(dV[1]), (0, 1, 1), 1)
	# 	cv2.line(img, toCV(dW[0]), toCV(dW[1]), (0, 1, 1), 1)

		# cv2.imshow("img", img)
		# cv2.waitKey()

	if I is None:
		# cv2.circle(img, toCV(Q), 1, (1, 1, 0), 1)
		# cv2.circle(img, toCV(Vc+Vp), 1, (1, 1, 0), 1)
		# cv2.circle(img, toCV(Wc+Wp), 1, (1, 1, 0), 1)
		# cv2.imshow("img", img)
		# cv2.waitKey()
		return Vc+Vp, Q, Wc+Wp

	# cv2.circle(img, toCV(I), 1, (1, 1, 0), -1)
	# cv2.circle(img, toCV(Vc+Vp), 1, (1, 1, 0), 1)
	# cv2.circle(img, toCV(Wc+Wp), 1, (1, 1, 0), 1)
	# cv2.imshow("img", img)
	# cv2.waitKey()
	return Vc+Vp, I, Wc+Wp

def drawTree(edges=None, nodes=None, triangles=None, triangleCenters=False, img=None, title="None"):
	showImg = img == None
	if img == None:
		img = np.ones((HEIGHT, WIDTH, 3)) * 0.1

	if edges != None:
		for V, W in edges:
			cv2.line(img, toCV(V), toCV(W), (0, 0.5, 1), 5, cv2.LINE_AA)

	if nodes != None:
		for N in nodes:
			cv2.circle(img, toCV(N), R, (1, 0, 0.5), 5, cv2.LINE_AA)

	if triangles != None:
		for T in triangles:
			cv2.line(img, toCV(T.v1), toCV(T.v2), (0.5, 1, 0), 5, cv2.LINE_AA)
			cv2.line(img, toCV(T.v2), toCV(T.v3), (0.5, 1, 0), 5, cv2.LINE_AA)
			cv2.line(img, toCV(T.v3), toCV(T.v1), (0.5, 1, 0), 5, cv2.LINE_AA)
			if triangleCenters:
				cv2.circle(img, toCV(T.C), 4, (1, 0.5, 0), -1, cv2.LINE_AA)

	# if showImg:
	# 	cv2.imshow("Tree", img)
	# 	cv2.waitKey()

	img = cv2.resize(img, (WIDTH//10, HEIGHT//10), cv2.INTER_LANCZOS4)
	cv2.imwrite(title + ".png", img*255)

	return img	

def drawTreeSVG(edges=None, nodes=None, triangles=None, triangleCenters=False):
	surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
	surface = cairo.SVGSurface("img.svg", WIDTH, HEIGHT)
	ctx = cairo.Context(surface)

	if edges != None:
		ctx.set_line_width(3)
		for (x1,y1), (x2, y2) in edges:
			ctx.new_sub_path()
			ctx.move_to(x1, y1)
			ctx.line_to(x2, y2)
			ctx.stroke()

	if nodes != None:
		ctx.set_line_width(3)
		for x,y in nodes:
			ctx.new_sub_path()
			ctx.arc(x, y, R, 0, 2*math.pi)
			ctx.stroke()

	surface.write_to_png("example.png")





######## GENERATE LEAFS ########
leafs = []
i = 0
while len(leafs) < NPOINTS and i < NPOINTS*200:
	i += 1
	# Generate random leaf somewhere in the tree
	N = generatePoint(radius=RADIUS, sigma=SIGMA) + O
	# Check if point in plane
	if not inPlane(N, R, WIDTH, HEIGHT):
		continue
	# Check if its not too close to another leaf
	distances = [distance(N, L) < LEAF_MIN_DISTANCE for L in leafs]
	if not any(distances):
		leafs.append(N)
print("Generated %d/%d leafs" % (len(leafs), NPOINTS))

leafCombs = list(itertools.combinations(leafs, 2))
distances = [(V-W).norm() for V, W in leafCombs]
print("Minimal distance between leaves: %0.2f" % min(distances))
######## LEAFS GENERATED ########





drawTree(nodes=leafs, title="A")





######## GENERATE PATHS ########
edges = []

# Generate Delaunay triangulation
triangulation = bowyerWatson(leafs)

drawTree(nodes=leafs, triangles=triangulation, triangleCenters=True, title="B")

### Remove all triangles from triangulation whose circumcentre does not lie within another triangle ###
### This is done to make sure that there will be no branches outside of the tree ###
_triangulation = triangulation
triangulation = []
for T in _triangulation:
	if any([_T.containsPoint(T.C) for _T in _triangulation]):
		triangulation.append(T)
### Remove all from triangulation whose center does not lie within another triangle ###

drawTree(nodes=leafs, triangles=triangulation, triangleCenters=True, title="C")

### Add all possible branches as edges ###
for T in triangulation:
	for _T in triangulation:
		if T.sharesVertexWith(_T):
			# Prevent duplicate edges, since line A-B != line B-A
			if (T.C, _T.C) not in edges and (_T.C, T.C) not in edges: 
				edges.append((T.C, _T.C))
print("%d voronoi edges added" % len(edges))
### Add all possible branches as edges ###

drawTree(edges=edges, nodes=leafs, triangles=triangulation, triangleCenters=True, title="D")
drawTree(edges=edges, nodes=leafs, title="E")

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

drawTree(edges=edges, nodes=leafs, title="F")

distances = [(V-W).norm() for V, W in edges]
print("Minimal distance  after branches: %0.2f" % min(distances))

### Add edge from root node to closest voronoi node ###
distances = [distance(rootNode, T.C) for T in triangulation]
iClosest = distances.index(min(distances))
leafs.append(rootNode)
edges.append((rootNode, triangulation[iClosest].C))

drawTree(edges=edges, nodes=leafs, title="G")

distances = [(V-W).norm() for V, W in edges]
print("Minimal distance  after root: %0.2f" % min(distances))

## Replace all branches smaller than BRANCH_MIN_LENGTH
edgesRemoved = 0
for i, (V, W) in enumerate(edges):
	# If edge is too small, replace all occurences of W with V and delete edge
	if distance(V, W) < BRANCH_MIN_LENGTH:
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





drawTree(edges=edges, nodes=leafs, title="H")





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





drawTree(edges=edges, nodes=leafs, title="I")





def generateThoseWeirdPuzzlePieces(V, W, direction):
	E = W-V
	dE = E/5
	r90 = 0.5*math.pi

	points = []
	P = V

	points.append(P); P += 2*dE;
	points.append(P); P += 2*dE.rotate(r90)*direction;
	points.append(P); P -= dE;
	points.append(P); P += 2*dE.rotate(r90)*direction;
	points.append(P); P += 3*dE;
	points.append(P); P -= 2*dE.rotate(r90)*direction;
	points.append(P); P -= dE;
	points.append(P); P -= 2*dE.rotate(r90)*direction;
	points.append(P); P += 2*dE;
	points.append(P);

	return points


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
ctx.set_line_width(1)

img = np.ones((HEIGHT, WIDTH, 3)) * 0.1

for NODE in nodes:

	if NODE in leafs:
		neighbour = [otherCorner(edge, NODE) for edge in edges if NODE in edge][0]
		offsetCairo = -0.5*math.pi
		offsetAngle = 2*math.pi*(15/360)
		angle = (NODE - neighbour).polar()[1]

		ctx.new_sub_path()
		# ctx.arc(NODE[0], NODE[1], R, offsetCairo-angle+offsetAngle, offsetCairo-angle-offsetAngle)
		
		# Get points where circle of leaf begins and ends
		V = Vec(0, -1).rotate(-angle - offsetAngle)
		W = Vec(0, -1).rotate(-angle + offsetAngle)
		
		ctx.move_to((NODE+V*R)[0], (NODE+V*R)[1])

		# Generate weird puzzle piece thing for leaf
		points = generateThoseWeirdPuzzlePieces(NODE + V*R, NODE + W*R, -1)
		# Draw weird puzzle piece thing
		for i in range(len(points)-2):
			p2, p3 = points[i+1], points[i+2]
			pp = (p2 + p3) / 2 # Middle of next path
			# Go to end of path if at last curve
			if i == len(points)-3:
				pp = p3 
			ctx.curve_to(p2[0], p2[1], p2[0], p2[1], pp[0], pp[1])

		offsetAngle = 2*math.pi*(30/360)
		# V = Vec(0, -1).rotate(-angle - offsetAngle)
		W = Vec(0, -1).rotate(-angle + offsetAngle)

		ctx.line_to((NODE+W*mm(20))[0], (NODE+W*mm(20))[1])
		ctx.arc(NODE[0], NODE[1], mm(20), offsetCairo-angle+offsetAngle, offsetCairo-angle-offsetAngle)
		ctx.line_to((NODE+V*R)[0], (NODE+V*R)[1])
		
		V = Vec(0, mm(4)).rotate(offsetCairo) + NODE
		ctx.move_to(V[0], V[1])
		ctx.arc(NODE[0], NODE[1], mm(4), 0, 2*math.pi)

		ctx.close_path()
		ctx.stroke()




		# ctx.new_sub_path()
		# ctx.arc(NODE[0], NODE[1], mm(40), 0, 2*math.pi)
		# ctx.close_path()
		# ctx.stroke()

		# thickness = mm(10)
		# outerR = R - thickness
		# innerR = mm(4) + thickness

		# ctx.move_to((NODE + W*outerR)[0], (NODE + W*outerR)[1])
		# ctx.arc(NODE[0], NODE[1], outerR, offsetCairo-angle+offsetAngle, offsetCairo-angle-offsetAngle)
		# ctx.move_to([0], W[1])

		# ctx.line_to(NODE[0] + V[0]-10, NODE[1] + V[1]-10)
		# ctx.line_to((NODE + V*outerR)[0], (NODE + V*outerR)[1])
		# ctx.arc_negative(NODE[0], NODE[1], innerR, offsetCairo-angle-offsetAngle, offsetCairo-angle+offsetAngle)

		# ctx.line_to((NODE + W*outerR)[0], (NODE + W*outerR)[1])

		# ctx.move_to(NODE[0]+mm(4), NODE[1])
		# ctx.arc(NODE[0], NODE[1], mm(4), 0,  2*math.pi)

		# ctx.close_path()
		# ctx.stroke()

		continue

	pairs = getConnectingPairsFromNode(edges, NODE)

	## Draw lines
	# for ((V, _), (W, _)) in pairs:
	# 	ctx.new_sub_path()
	# 	ctx.move_to(NODE[0], NODE[1])	
	# 	ctx.line_to(V[0], V[1])
	# 	ctx.close_path()

	beginX, beginY, isLeaf = None, None, None # Used to close the path at the end
	ctx.new_sub_path()
	for iPair, (P1, P2) in enumerate(pairs):
		V, W = P1[0], P2[0]
		(dVx, dVy), (Qx, Qy), (dWx, dWy) = getPointsFromPairs(NODE, P1, P2, leafs, BRANCH_MIN_WIDTH, BRANCH_MAX_WIDTH)

		if iPair == 0:
			ctx.move_to(dVx, dVy)
			beginX, beginY = dVx, dVy
			isLeaf = V in leafs
		else:
			x1, y1 = ctx.get_current_point()
			x2, y2 = dVx, dVy
			A, B = Vec(x1, y1), Vec(x2, y2)
			
			## Get direction of puzzle piece
			direction = 1
			if distance(rootNode, A) < distance(rootNode, B) or V in leafs:
				direction = -1
			## Generate puzzle piece points
			points = generateThoseWeirdPuzzlePieces(A, B, direction)
			
			## Draw puzzle piece
			for i in range(len(points)-2):
				p2, p3 = points[i+1], points[i+2]
				pp = (p2 + p3) / 2
				if i == len(points)-3:
					pp = p3
				ctx.curve_to(p2[0], p2[1], p2[0], p2[1], pp[0], pp[1])

		# Draw curve from one edge of node to other
		ctx.curve_to(Qx, Qy, Qx, Qy, dWx, dWy)

	## Draw last puzzle piece
	x1, y1 = ctx.get_current_point()
	x2, y2 = beginX, beginY
	A, B = Vec(x1, y1), Vec(x2, y2)
	
	## Get direction of puzzle piece
	direction = 1
	if distance(rootNode, A) < distance(rootNode, B) or isLeaf:
		direction = -1
	## Generate puzzle piece points
	points = generateThoseWeirdPuzzlePieces(A, B, direction)
	
	## Draw puzzle piece
	for i in range(len(points)-2):
		p2, p3 = points[i+1], points[i+2]
		pp = (p2 + p3) / 2
		if i == len(points)-3:
			pp = p3
		ctx.curve_to(p2[0], p2[1], p2[0], p2[1], pp[0], pp[1])

	ctx.stroke()
	ctx.close_path()

surface.write_to_png("after.png")

# cv2.imshow("img", img)
# cv2.waitKey()

exit()

