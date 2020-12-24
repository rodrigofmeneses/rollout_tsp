#%%
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random as rd
import os
import time

from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

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
		self.Graph = self.create_model(path)
		# Cities
		self.num_cities = self.Graph.number_of_nodes()
		self.cities = list(range(self.num_cities))
		
		self.tour = None
		self.tour_cost = 0
		self.starting_node = np.random.choice(self.cities)


	'''
		Private Methods
	'''

	def create_model(self, path):
		"""
			Create Graph representation and MDP variables with number of states designated
		"""
		data_format = path.split('/')[1]
		
		if data_format == 'edge_explicity':
			return self.read_edge_explicity_data(path)

		elif data_format == 'node_coordinate':
			return self.read_node_coordinate_data(path)

	def read_edge_explicity_data(self, path):
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

	def read_node_coordinate_data(self, path):
		with open(path) as reader:
			G = nx.complete_graph(int(reader.readline()))
			X = []
			Y = []
			for _ in range(G.number_of_nodes()):
				x, y = reader.readline().split()
				X.append(int(x))
				Y.append(int(y))
			coordinates = np.array(list(zip(X, Y)))
			dist_matrix = squareform(pdist(coordinates))
			for i in range(G.number_of_nodes()):
				for j in range(G.number_of_nodes()):
					if i != j:
						G.edges[i, j]['w'] = dist_matrix[i, j]
						G.edges[j, i]['w'] = dist_matrix[j, i]
		return G

	def create_adj_mat(self, tour):
		# Adjacency matrix
		adj_mat = np.array(nx.adjacency_matrix(self.Graph, weight='w').todense(), dtype=float)
		# set main diagonal to inf
		for i in range(self.num_cities): adj_mat[i][i] = np.inf
		# set elements of current tour to inf
		for i in range(self.num_cities):
			for t in tour[:-1]:
				adj_mat[i][t] = np.inf

		return adj_mat

	def calc_cost(self, tour):
		cost = 0
		for i in range(self.num_cities):
			cost += self.Graph[tour[i]][tour[i+1]]['w']
		return cost

	def nearest_neighbor(self, tour):
		'''
			tour: is partial tour of solution, 
			example, 5 cities, tour is list of nodes [0, 1, 2], index 0 is starting city,
			the return is a complete tour, in the form [0, 1, 2, 4, 3, 0] and cost of tour 
		'''
		# Create adjacency matrix
		adj_mat = self.create_adj_mat(tour)
		# Complete tour 
		for _ in range(self.num_cities - len(tour)):
			min_index = np.argmin(adj_mat[tour[-1]])
			for t in tour:
				adj_mat[min_index][t] = np.inf
				adj_mat[t][min_index] = np.inf
			tour.append(min_index)
		tour.append(self.starting_node)
		# Calculate cost
		cost = self.calc_cost(tour)
		return tour, cost

	'''
		Public Methods
	'''

	def run(self):
		self.tour = [self.starting_node]
		global_cost = []
		
		for i in range(self.num_cities - 1):
			curr_tour = self.tour.copy()
			best_rol_cost = np.inf
			best_node = None

			# One step look ahead
			for j in set(self.Graph[self.tour[-1]]) - set(self.tour):
				curr_tour.append(j)

				_, rol_cost = self.nearest_neighbor(curr_tour.copy())
				# global_cost.append(rol_cost)
				if rol_cost < best_rol_cost:
					best_rol_cost = rol_cost
					best_node = j

				curr_tour.pop()
			self.tour.append(best_node)
		self.tour.append(self.starting_node)
		
		# print('Custos globais:', global_cost)
		# print('Tour rol sol:', self.tour)
		# print('Cost rol sol:', self.calc_cost(self.tour))
		return self.tour, self.calc_cost(self.tour)

	def draw_graph(self):
		pos = nx.spring_layout(self.Graph)
		nx.draw(self.Graph, pos, with_labels=True)
		edge_labels = nx.get_edge_attributes(self.Graph, 'w')
		nx.draw_networkx_edge_labels(self.Graph, pos, edge_labels)


#%%
if __name__ == "__main__":
	edge_explicity_data = os.listdir('data/edge_explicity')
	node_coordinate_date = os.listdir('data/node_coordinate')

	# Experiments
	path = 'data/edge_explicity/'
	results = dict()
	n_episode = 5
	for sample in edge_explicity_data:
		tsp_cost_mean = 0
		tsp_time_mean = 0
		nn_cost_mean = 0
		nn_time_mean = 0
		for episode in range(n_episode):
			exp = TSPR(path + sample)

			init_time = time.time()
			_, cost = exp.run()
			tsp_time_mean += time.time() - init_time
			tsp_cost_mean += cost

			init_time = time.time()
			_, cost = exp.nearest_neighbor([exp.starting_node])
			nn_time_mean += time.time() - init_time
			nn_cost_mean += cost

		print(f'\nTSP sample: {sample}, in {n_episode} episodes')
		print('\nRollout Algorithm')
		print(f'Time mean: {tsp_time_mean / n_episode}')
		print(f'Cost mean: {tsp_cost_mean / n_episode}')
		print('\nNearest Neighbor Algorithm')
		print(f'Time mean: {nn_time_mean / n_episode}')
		print(f'Cost mean: {nn_cost_mean / n_episode}')



# %%
tour, cost = t.nearest_neighbor([t.starting_node])
print('Tour NN sol:', tour)
print('Tour NN cost:', cost)
# %%
