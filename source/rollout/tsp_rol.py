#%%
import numpy as np
#import matplotlib.pyplot as plt
import networkx as nx
import random as rd
import os
import time
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from numba import jit

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

	def create_model(self, path):
		"""
			Create Graph representation and MDP variables with number of states designated
			path: path of instance
		"""
		data_format = path.split('/')[1]
		
		if data_format == 'edge_explicity':
			return self.read_edge_explicity_instances(path)

		elif data_format == 'node_coordinate':
			return self.read_node_coordinate_instances(path)
		else:
			raise Exception

	def read_edge_explicity_instances(self, path):
		'''
			Read explicity edge instance
			path: path of instance
			ex:

			5   			 <- number of nodes
			1231 232 12 122  <- edges (0,1)(0,2)(0,3)(0,4)
			336 1161 717 	 <- edges (1,2)(1,3)(1,4)
			848 323			 <- edges (2,3)(2,4)
			212				 <- edges (3,4)

			return: Graph with weights
		'''
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

	def read_node_coordinate_instances(self, path):
		'''
			Read node coordinate instances
			path: path of instance
			ex:

			5     <- number of nodes
			1 1   <- node 0, coordinate (x, y) = (1, 1) 
			2 3   <- node 0, coordinate (x, y) = (2, 3) 	
			4 5   <- node 0, coordinate (x, y) = (4, 5) 
			3 3   <- node 0, coordinate (x, y) = (3, 3) 
			7 5   <- node 0, coordinate (x, y) = (7, 5) 

			return: Graph with weights
		'''
		with open(path) as reader:
			G = nx.complete_graph(int(reader.readline()))
			X = []
			Y = []
			for _ in range(G.number_of_nodes()):
				x, y = reader.readline().split()
				X.append(float(x))
				Y.append(float(y))
			coordinates = np.array(list(zip(X, Y)))
			dist_matrix = squareform(pdist(coordinates))
			for i in range(G.number_of_nodes()):
				for j in range(G.number_of_nodes()):
					if i != j:
						G.edges[i, j]['w'] = dist_matrix[i, j]
						G.edges[j, i]['w'] = dist_matrix[j, i]
		return G

	def create_adj_mat(self, tour):
		'''
			Adjacency matrix to nn algorithm purposes
			tour: partial ou complete tour to create adjacency matrix, ex:
			[0, 2, 4]  < for 5 cities [0, 1, 2, 3, 4, 5] and starting city is 0 
			neighbor with travel cost equal inf is never a good option, 
			so cities in tour recieve this travel cost.
			return: adjacency matrix
		'''
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
		'''
			Calc cost of tour.
			tour: tour of traveler, ex:
			[0, 2, 4, 1, 3, 0]  <- for 5 cities and starting city is 0 
			return: cost of tour
		'''
		cost = 0
		for i in range(self.num_cities):
			cost += self.Graph[tour[i]][tour[i+1]]['w']
		return cost

	# @jit(nopython=True)
	def nearest_neighbor(self, tour):
		'''
			Base heuristic to rollout algorithm, the main difference of traditional 
			nearest neighbor algorithm is the beginning  with a partial tour (solution), ex:
			tour is list of nodes [0, 1, 2], for 5 cities and starting city is 0,
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

	def run(self):
		self.tour = [self.starting_node]
		
		for _ in range(self.num_cities - 1):
			curr_tour = self.tour.copy()
			best_rol_cost = np.inf
			best_node = None

			# One step look ahead
			for j in set(self.Graph[self.tour[-1]]) - set(self.tour):
				curr_tour.append(j)

				_, rol_cost = self.nearest_neighbor(curr_tour.copy())
				if rol_cost < best_rol_cost:
					best_rol_cost = rol_cost
					best_node = j

				curr_tour.pop()
			self.tour.append(best_node)
		self.tour.append(self.starting_node)

		return self.tour, self.calc_cost(self.tour)

	def draw_graph(self):
		pos = nx.spring_layout(self.Graph)
		nx.draw(self.Graph, pos, with_labels=True)
		edge_labels = nx.get_edge_attributes(self.Graph, 'w')
		nx.draw_networkx_edge_labels(self.Graph, pos, edge_labels)

#%%
def experiments_with(file_path):
	# Instanciate TSPR
	exp = TSPR(file_path)
	# Experiment with tsp rollout algorithm
	init_time = time.time()
	_, rol_cost = exp.run()
	rol_time = time.time() - init_time

	# Experiment with tsp nearest neighbor algorithm
	init_time = time.time()
	_, nn_cost = exp.nearest_neighbor([exp.starting_node])
	nn_time = time.time() - init_time

	print(_)

	return rol_cost, rol_time, nn_cost, nn_time

if __name__ == "__main__":
	# Read Instances
	folders = os.listdir('instances')
	instances = dict()
	instances['edge_explicity'] = os.listdir('instances/edge_explicity')
	instances['node_coordinate'] = os.listdir('instances/node_coordinate')
	
	results = open(f'experiments/results{time.time_ns()}.txt', 'w')
	results.write('instance_name,rol_cost,rol_time,nn_cost,nn_time\n')
	n_episodes = 1
	#test_inst = ['brazil58.tsp']
	test_inst = ['bier127.tsp']
	for folder in folders:
		for instance in instances[folder]:
			if instance not in test_inst:
				continue
			file_path = f'instances/{folder}/{instance}'
			# result = [instance]
			results.write(instance)
			mean_result = np.zeros(4)
			for e in range(n_episodes):
				mean_result += np.array(experiments_with(file_path))
			mean_result /= n_episodes
			# results.append(result + list(mean_result))
			results.write(f',{mean_result[0]},{mean_result[1]},{mean_result[2]},{mean_result[3]}\n')
	results.close()
			

# %%