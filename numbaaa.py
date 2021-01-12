import numpy as np
import networkx as nx
import time
import os
from numba import jit
from numba.typed import List
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


def read_edge_explicity_instances(path):
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

def read_node_coordinate_instances(path):
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

def create_model(path):
    """
        Create Graph representation and MDP variables with number of states designated
        path: path of instance
    """
    data_format = path.split('/')[-2]
    
    if data_format == 'edge_explicity':
        return read_edge_explicity_instances(path)

    elif data_format == 'node_coordinate':
        return read_node_coordinate_instances(path)
    else:
        raise Exception

@jit(nopython=True)
def update_adj_mat(tour, num_cities, adj_mat):
    '''
        Adjacency matrix to nn algorithm purposes
        tour: partial ou complete tour to create adjacency matrix, ex:
        [0, 2, 4]  < for 5 cities [0, 1, 2, 3, 4, 5] and starting city is 0 
        neighbor with travel cost equal inf is never a good option, 
        so cities in tour recieve this travel cost.
        return: adjacency matrix
    '''
    # Adjacency matrix
    #adj_mat = np.array(nx.adjacency_matrix(G, weight='w').todense(), dtype=float)
    #adj_mat = ADJ_MAT.copy()
    # set main diagonal to inf
    for i in range(num_cities): adj_mat[i][i] = np.inf
    # set elements of current tour to inf
    for i in range(num_cities):
        for t in tour[:-1]:
            adj_mat[i][t] = np.inf
    return adj_mat


@jit(nopython=True)
def calculate_cost(tour, adj_mat):
    '''
        Calc cost of tour.
        tour: tour of traveler, ex:
        [0, 2, 4, 1, 3, 0]  <- for 5 cities and starting city is 0 
        return: cost of tour
    '''
    cost = 0.
    for i in range(len(tour) - 1):
        cost += adj_mat[tour[i]][tour[i+1]]
    return cost


@jit(nopython=True)
def nearest_neighbor(tour, num_cities, adj_mat):
    '''
        Base heuristic to rollout algorithm, the main difference of traditional 
        nearest neighbor algorithm is the beginning  with a partial tour (solution), ex:
        tour is list of nodes [0, 1, 2], for 5 cities and starting city is 0,
        the return is a complete tour, in the form [0, 1, 2, 4, 3, 0] and cost of tour 
        
    '''
    # Complete tour 
    for _ in range(num_cities - len(tour)):
        min_index = np.argmin(adj_mat[tour[-1]])
        for t in tour:
            adj_mat[min_index][t] = np.inf
            adj_mat[t][min_index] = np.inf
        tour.append(min_index)
    tour.append(tour[0])
    # Return complete tour
    return tour

def rollout_algorithm(G, num_cities, starting_node, adj_mat):
    tour = [starting_node]

    for i in range(num_cities - 1):
        #current_tour = tour.copy()

        # Exigência do numba - tranformar em funcao
        current_tour = List()
        [current_tour.append(x) for x in tour]

        best_rollout_cost = np.inf
        best_next_city = None
        
        for j in set(G[tour[-1]]) - set(tour):
            current_tour.append(j)

            # Exigencia numba
            tour_numba = List()
            [tour_numba.append(x) for x in tour]

            current_adj_mat = update_adj_mat(tour_numba, num_cities, adj_mat.copy())
            
            nn_path = nearest_neighbor(current_tour.copy(), num_cities, current_adj_mat.copy())
            
            # Exigência do numba
            nn_path_numba = List()
            [nn_path_numba.append(x) for x in nn_path]
            
            rollout_cost = calculate_cost(nn_path_numba, adj_mat.copy())
            if rollout_cost < best_rollout_cost:
                best_rollout_cost = rollout_cost
                best_next_city = j
            
            current_tour.pop()
        tour.append(best_next_city)
    tour.append(starting_node)
    
    # Exigência numba
    tour_numba = List()
    [tour_numba.append(x) for x in tour]

    return tour, calculate_cost(tour_numba, adj_mat.copy())

def experiments_with(file_path):
    
    G = create_model(file_path)
    adj_mat = np.array(nx.adjacency_matrix(G, weight='w').todense(), dtype=float)
    num_cities = G.number_of_nodes()

    starting_node = int(np.random.choice(range(num_cities)))

    start = time.time()
    rollout_tour, rollout_cost = rollout_algorithm(G, num_cities, starting_node, adj_mat)
    rollout_time = time.time() - start
    
    start = time.time()

    # Exigencia numba
    starting_tour = List()
    starting_tour.append(starting_node)

    nn_tour = nearest_neighbor(starting_tour, num_cities, adj_mat.copy())
    
    # Exigencia numba
    nn_tour_numba = List()
    [nn_tour_numba.append(x) for x in nn_tour]

    nn_cost = calculate_cost(nn_tour_numba, adj_mat.copy())
    nn_time = time.time() - start
    #print('Custo do rollout tour: ', rollout_cost)
    #print('Tempo do rollout algorithm: ', rollout_time)
    #print('Custo do nn tour: ', calculate_cost(nn_tour))
    #print('Tempo do nn algorithm: ', nn_time)
    return rollout_cost, rollout_time, nn_cost, nn_time

#%%
print(experiments_with('instances/edge_explicity/brazil58.tsp'))
#print(experiments_with('instances/node_coordinate/bier127.tsp'))