import math
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

	GCD = reduce(lambda x,y: math.gcd(x,y), [a, b, c])
	while GCD != 1:
		a = a // GCD
		b = b // GCD
		c = c // GCD
		GCD = reduce(lambda x,y: math.gcd(x,y), [a, b, c])

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

def distancePointToSegment(V, W, P):
	length = distance(V, W)
	if length == 0:
		return distance(V, P)
	# Find projection of p onto w, by first moving everything to (0, 0) (aka subtracting v)
	f = ((P-V) * (W-V)) / ((W-V)*(W-V))
	# Clamp projection between [0, 1]
	t = max(0, min(f, 1))
	# Calculate the projection
	proj = V + t * (W-V)

	return distance(P, proj)

# https://stackoverflow.com/questions/2049582/how-to-determine-if-a-point-is-in-a-2d-triangle
# def pointInTriangle(P, T):
# 	(P0, P1, P2) = T.corners
# #   A = 1/2 * (-p1.y * p2.x + p0.y * (-p1.x + p2.x) + p0.x * (p1.y - p2.y) + p1.x * p2.y);
# 	A = 0.5 * (-P1[1]* P2[0]+ P0[1]* (-P1[0]+ P2[0]) +P0[0]* (P1[1] -P2[1]) +P1[0]* P2[1])
# #   sign = A < 0 ? -1 : 1;
# 	sign = -1 if A < 0 else 1
# #	s = (p0.y * p2.x - p0.x * p2.y + (p2.y - p0.y) * p.x + (p0.x - p2.x) * p.y) * sign;
# 	s = (P0[1] *P2[0] -P0[0] *P2[1] +(P2[1] -P0[1]) *P[0] +(P0[0] -P2[0]) *P[1]) *sign
# #   t = (p0.x * p1.y - p0.y * p1.x + (p0.y - p1.y) * p.x + (p1.x - p0.x) * p.y) * sign;
# 	t = (P0[0] *P1[1] -P0[1] *P1[0] +(P0[1] -P1[1]) *P[0] +(P1[0] -P0[0]) *P[1]) *sign 
# #   return s > 0 && t > 0 && (s + t) < 2 * A * sign;
# 	return s >0 and t >0 and (s + t) < 2 * A * sign
