from functions import distance, lineFromPoints, perpendicularBisectorFromLine, lineLineIntersection

class Triangle(object):
	def __init__(self, v1, v2, v3):
		self.v1 = v1
		self.v2 = v2
		self.v3 = v3
		self.corners = [v1, v2, v3]
		self.vertices = [(v1, v2), (v2, v3), (v3, v1)]

		a, b, c = lineFromPoints(v1, v2)
		e, f, g = lineFromPoints(v2, v3)

		a, b, c = perpendicularBisectorFromLine(v1, v2, a, b, c)
		e, f, g = perpendicularBisectorFromLine(v2, v3, e, f, g)

		self.C = lineLineIntersection(a, b, c, e, f, g)
		self.R = distance(v1, self.C) if self.C != None else None

	def hasVertex(self, V):
		return V[0] in self.corners and V[1] in self.corners

	def sharesVertexWith(self, other):
		return sum([1 if t in self.corners else 0 for t in other.corners]) == 2

	def sharesCornerWith(self, other):
		v1, v2, v3 = other.corners
		return v1 in self.corners or v2 in self.corners or v3 in self.corners

	def containsPoint(self, P):
		(P0, P1, P2) = self.corners
	#   A = 1/2 * (-p1.y * p2.x + p0.y * (-p1.x + p2.x) + p0.x * (p1.y - p2.y) + p1.x * p2.y);
		A = 0.5 * (-P1[1]* P2[0]+ P0[1]* (-P1[0]+ P2[0]) +P0[0]* (P1[1] -P2[1]) +P1[0]* P2[1])
	#   sign = A < 0 ? -1 : 1;
		sign = -1 if A < 0 else 1
	#	s = (p0.y * p2.x - p0.x * p2.y + (p2.y - p0.y) * p.x + (p0.x - p2.x) * p.y) * sign;
		s = (P0[1] *P2[0] -P0[0] *P2[1] +(P2[1] -P0[1]) *P[0] +(P0[0] -P2[0]) *P[1]) *sign
	#   t = (p0.x * p1.y - p0.y * p1.x + (p0.y - p1.y) * p.x + (p1.x - p0.x) * p.y) * sign;
		t = (P0[0] *P1[1] -P0[1] *P1[0] +(P0[1] -P1[1]) *P[0] +(P1[0] -P0[0]) *P[1]) *sign 
	#   return s > 0 && t > 0 && (s + t) < 2 * A * sign;
		return s >0 and t >0 and (s + t) < 2 * A * sign

	def __repr__(self):
		return str(self.corners)

	def __eq__(self, other):
		if type(self) != type(other):
			return False
		return all([c in other.corners for c in self.corners])
