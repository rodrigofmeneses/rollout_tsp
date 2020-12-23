#%%
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random as rd
import os
import time

class TSPR():
	'''
		Traveling Salesman Problem,
		whereby a salesman wants to find minimun cost tour.
		That visit each of N given citys once and returns to the city started from.
		Each city i = 1,..., N, a node is associated and an arc (i, j) with travel cost aij is introduced
		for each pair of nodes i and j.
	'''
	def __init__(self, path, seed=0):		
		if seed: np.random.seed(seed)
		# Graph
		self.Graph = self._create_model(path)
		# Cities
		self.num_cities = self.Graph.number_of_nodes()
		self.cities = list(range(self.num_cities))
		# 
		self.tour = None
		self.tour_cost = 0
		self.starting_node = np.random.choice(self.cities)


	'''
		Private Methods
	'''

	def _create_model(self, path):
		"""
			Create Graph representation and MDP variables with number of states designated
		"""
		# 'brazil58.tsp'
		with open(path) as reader:
			# # Complete graph
			G = nx.complete_graph(int(reader.readline()))
			for i in range(G.number_of_nodes()):
				# Read lines to set weights
				nbrs_i = [int(x) for x in reader.readline().split()]
				for j, w in enumerate(nbrs_i, i + 1):
					G.edges[i, j]['w'] = w
					G.edges[j, i]['w'] = w
		return G

	def _create_adj_mat(self, tour):
		# Adjacency matrix
		adj_mat = np.array(nx.adjacency_matrix(self.Graph, weight='w').todense(), dtype=float)
		# set main diagonal to inf
		for i in range(self.num_cities): adj_mat[i][i] = np.inf
		# set elements of current tour to inf
		for i in range(self.num_cities):
			for t in tour[:-1]:
				adj_mat[i][t] = np.inf

		return adj_mat

	def _calc_cost(self, tour):
		cost = 0
		for i in range(self.num_cities):
			cost += self.Graph[tour[i]][tour[i+1]]['w']
		return cost

	def _nearest_neighbor(self, tour):
		'''
			tour: is partial tour of solution, 
			example, 5 cities, tour is list of nodes [0, 1, 2], index 0 is starting city,
			the return is a complete tour, in the form [0, 1, 2, 4, 3, 0] and cost of tour 
		'''
		# Create adjacency matrix
		adj_mat = self._create_adj_mat(tour)
		# Complete tour 
		for _ in range(self.num_cities - len(tour)):
			min_index = np.argmin(adj_mat[tour[-1]])
			for t in tour:
				adj_mat[min_index][t] = np.inf
				adj_mat[t][min_index] = np.inf
			tour.append(min_index)
		tour.append(self.starting_node)
		# Calculate cost
		cost = self._calc_cost(tour)
		return tour, cost

	'''
		Public Methods
	'''

	def run(self):
		print(np.array(nx.adjacency_matrix(self.Graph, weight='w').todense(), dtype=float))
		self.tour = [self.starting_node]
		global_cost = []
		
		for i in range(self.num_cities - 1):
			curr_tour = self.tour.copy()
			best_rol_cost = np.inf
			best_node = None

			# One step look ahead
			for j in set(self.Graph[self.tour[-1]]) - set(self.tour):
				curr_tour.append(j)

				_, rol_cost = self._nearest_neighbor(curr_tour.copy())
				# global_cost.append(rol_cost)
				if rol_cost < best_rol_cost:
					best_rol_cost = rol_cost
					best_node = j

				curr_tour.pop()
			self.tour.append(best_node)
		self.tour.append(self.starting_node)
		
		# print('Custos globais:', global_cost)
		print('Tour rol sol:', self.tour)
		print('Cost rol sol:', self._calc_cost(self.tour))
		return self.tour, self._calc_cost(self.tour)

	def draw_graph(self):
		pos = nx.spring_layout(self.Graph)
		nx.draw(self.Graph, pos, with_labels=True)
		edge_labels = nx.get_edge_attributes(self.Graph, 'w')
		nx.draw_networkx_edge_labels(self.Graph, pos, edge_labels)


#%%
if __name__ == "__main__":
	path = 'data/brazil58.tsp'
	t = TSPR(path, seed=2)
	t.run()