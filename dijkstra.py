from functions import distance

### Dijkstra from one source to entire graph ###
def dijkstra(nodes, edges, source):
	print("\nDijkstra : %d nodes, %d edges, %s" % (len(nodes), len(edges), source))
	inf = float("inf")
	
	distances = {node : inf for node in nodes}
	previous = {node : None for node in nodes}

	distances[source] = 0

	while 0 < len(nodes):
		dists = [(distances[node], node) for node in nodes]
		node = min(dists, key = lambda x : x[0])[1]

		# print("  node=%s" % node, "distance=%0.2f" % distances[node])
	
		if distances[node] == inf:
			print("  No possible path from source to all destinations")
			return previous

		nodes.remove(node)
		
		# For each neighbour v of u
		neighbours = [edge for edge in edges if node in edge]
		for neighbour in neighbours:
			# Get other end of edge
			other = neighbour[1] if neighbour[0] == node else neighbour[0]
			alt = distances[node] + distance(node, other)
			if alt < distances[other]:
				distances[other] = alt
				previous[other] = node
			# print("    ->", other, "= %0.2f" % distances[other])
	
	return previous
################################################