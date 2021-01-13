#%%
import numpy as np
import networkx as nx
import time
import os
from numba import jit
from numba.typed import List
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
#%%

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

# def nn_core(tour):

#     nn_tour = nearest_neighbor()
#     nn_cost = 
#     return nn_tour, nn_cost

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

def rollout_algorithm(G, adj_mat, num_cities, starting_node):

    
    # Starting tour
    tour = [starting_node]

    for _ in range(num_cities - 1):
        current_tour = tour.copy()
        best_rollout_cost = np.inf
        best_next_city = None
        
        for j in set(G[tour[-1]]) - set(tour):
            current_tour.append(j)

            current_adj_mat = update_adj_mat(current_tour.copy(), num_cities, adj_mat.copy())
            nn_path = nearest_neighbor(current_tour.copy(), num_cities, current_adj_mat.copy())
            
            rollout_cost = calculate_cost(nn_path, adj_mat.copy())
            if rollout_cost < best_rollout_cost:
                best_rollout_cost = rollout_cost
                best_next_city = j
            
            current_tour.pop()
        tour.append(best_next_city)
    tour.append(starting_node)

    return tour, calculate_cost(tour, adj_mat.copy())

def experiments_with(file_path):
    # Create Model (Graph)
    G = create_model(file_path)

    # Create Adjacency Matrix
    adj_mat = np.array(nx.adjacency_matrix(G, weight='w').todense(), dtype=float)
    # number of cities
    num_cities = G.number_of_nodes()
    # random starting node
    starting_node = int(np.random.choice(range(num_cities)))

    # execute rollout algorithm and calculate time
    start = time.time()
    rollout_tour, rollout_cost = rollout_algorithm(G, adj_mat, num_cities, starting_node)
    rollout_time = time.time() - start
    
    # execute nearest neighbor algorithm and calculate time
    start = time.time()
    current_adj_mat = update_adj_mat([starting_node], num_cities, adj_mat.copy())
    nn_tour = nearest_neighbor([starting_node], num_cities, current_adj_mat.copy())
    nn_cost = calculate_cost(nn_tour, adj_mat.copy())
    nn_time = time.time() - start

    return rollout_cost, rollout_time, nn_cost, nn_time

#%%
# Read Instances
folders = os.listdir('instances')[:1]
# Create Results file
results = open(f'experiments/results{time.strftime("%d%b%Y_%H_%M_%S", time.gmtime())}.txt', 'w')
# Write header
results.write('instance_name,rol_cost,rol_time,nn_cost,nn_time\n')


# Start tests
n_episodes = 10

for folder in folders:
    for instance in os.listdir(f'instances/{folder}'):
        file_path = f'instances/{folder}/{instance}'
        # Write instance name
        results.write(instance)
        # Initialize mean array with 0
        mean_result = np.zeros(4)

        # Run experiments n_episodes times
        for e in range(n_episodes):
            mean_result += np.array(experiments_with(file_path))
        # Calculate mean 
        mean_result /= n_episodes
        # Write instance results
        results.write(f',{mean_result[0]},{mean_result[1]},{mean_result[2]},{mean_result[3]}\n')
# Close file
results.close()
# %%
