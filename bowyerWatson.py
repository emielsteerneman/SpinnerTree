from vector import Vector as Vec
from triangle import Triangle
from functions import distance

### Generate Delaunay triangulation ###
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
#######################################