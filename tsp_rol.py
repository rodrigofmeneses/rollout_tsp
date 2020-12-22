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
		whereby a salesman wants to find minimun cost path.
		That visit each of N given citys once and returns to the city started from.
		Each city i = 1,..., N, a node is associated and an arc (i, j) with travel cost aij is introduced
		for each pair of nodes i and j.
	'''
	def __init__(self, num_cities, starting_node, seed=0):		
		if seed: rd.seed(seed)

		self.num_cities = num_cities
		self.cities = set(range(num_cities))
		# self.N = num_cities
		
		self.travel_cost = np.zeros((self.num_cities, self.num_cities))
		# self.a = self.travel_cost
		
		self.Graph = None

		self.starting_node = starting_node
		# self.s = starting_node

		self.tour = [self.starting_node]

		self._createModel()

	'''
		Private Methods
	'''
	def _create_adj_mat(self, path):
		adj_mat = np.array(nx.adjacency_matrix(self.Graph, weight='w').todense(), dtype=float)
		for i in range(self.num_cities): adj_mat[i][i] = np.inf
		for i in range(self.num_cities):
			for t in path[:-1]:
				adj_mat[i][t] = np.inf
		# for t in path[:-1]:
		# 	for i in range(self.num_cities - 1):
		# 		adj_mat[t][i] = np.inf
		# 		adj_mat[i][t] = np.inf
		return adj_mat

	def _calc_cost(self, path):
		cost = 0
		for i in range(self.num_cities):
			cost += self.Graph[path[i]][path[i+1]]['w']
		return cost

	def _nearest_neighbor(self, path):
		'''
			path: is partial tour of solution, 
			example, 5 cities, path is list of nodes [0, 1, 2], index 0 is starting city,
			the return is a complete path, in the form [0, 1, 2, 4, 3, 0] and cost of path 
		'''
		# Create adjacency matrix
		adj_mat = self._create_adj_mat(path)
		# Complete path 
		for _ in range(self.num_cities - len(path)):
			min_index = np.argmin(adj_mat[path[-1]])
			for t in path:
				adj_mat[min_index][t] = np.inf
				adj_mat[t][min_index] = np.inf
			path.append(min_index)
		path.append(0)
		# Calculate cost
		cost = self._calc_cost(path)
		return path, cost


	def _createModel(self):
		"""
			Create Graph representation and MDP variables with number of states designated
		"""
		# Complete graph
		self.Graph = nx.complete_graph(self.num_cities)
		
		# Random weights
		for i, j in self.Graph.edges():
			self.Graph.edges[i, j]['w'] = rd.randint(1, 11)
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

			for j in set(self.Graph[self.tour[-1]]) - set(self.tour):
				curr_tour.append(j)

				_, rol_cost = self._nearest_neighbor(curr_tour.copy())
				
				global_cost.append(rol_cost)

				if rol_cost < best_rol_cost:
					best_rol_cost = rol_cost
					best_node = j

				curr_tour.pop()
			self.tour.append(best_node)
		self.tour.append(0)
		
		print('Custos globais:', global_cost)
		print('Custo rol sol:', self._calc_cost(self.tour))
		return self.tour, self._calc_cost(self.tour)

	def draw_graph(self):
		pos = nx.spring_layout(self.Graph)
		nx.draw(self.Graph, pos, with_labels=True)
		edge_labels = nx.get_edge_attributes(self.Graph, 'w')
		nx.draw_networkx_edge_labels(self.Graph, pos, edge_labels)

#%%
if __name__ == "__main__":
	seed = 0
	num_cities = 10
	starting_node = 0
	t = TSPR(num_cities, starting_node, seed)
	t.run()
# %%
