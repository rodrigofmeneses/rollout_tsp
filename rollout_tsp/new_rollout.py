#%%
import os
import numpy as np
import tsplib95
# from numba import jit
#%%

def get_distance_matrix(problem):
    n = problem.dimension
    plus = 0
    if next(problem.get_nodes()) == 1:
        plus = 1

    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + plus, n):
            distance_matrix[i, j] = problem.get_weight(i + plus, j + plus)
    
    return distance_matrix

# @jit(nopython=True)
def calculate_tour_cost(tour, distance_matrix):
    '''
        Calc cost of tour.
        tour: tour of traveler, ex:
        [0, 2, 4, 1, 3, 0]  <- for 5 cities and starting city is 0 
        return: cost of tour
    '''
    cost = 0.
    for i in range(len(tour) - 1):
        cost += distance_matrix[tour[i]][tour[i+1]]
    return cost

# @jit(nopython=True)
def nearest_neighbor(tour, distance_matrix):
    '''
        Core of nearest neighbor, numba require only loops and numpy operations.
        Update adjacency matrix to nn algorithm purposes, ex:
        [0, 2, 4]  < for 5 cities [0, 1, 2, 3, 4, 5] and starting city is 0 
        neighbor with travel cost equal inf is never a good option, 
        so cities in tour recieve this travel cost.
        
    '''
    num_cities = distance_matrix.shape[0]
    # Update adjacency matrix
    # set main diagonal to inf
    for i in range(num_cities): distance_matrix[i][i] = np.inf
    # set elements of current tour to inf
    for i in range(num_cities):
        for t in tour[:-1]: 
            distance_matrix[i][t] = np.inf

    # Complete tour 
    for _ in range(num_cities - len(tour)):
        min_index = np.argmin(distance_matrix[tour[-1]])
        for t in tour:
            distance_matrix[min_index][t] = np.inf
            distance_matrix[t][min_index] = np.inf
        tour.append(min_index)
    tour.append(tour[0])
    # Return complete tour
    return tour

def rollout_algorithm(problem, starting_node=0):
    # Distance Matrix
    distance_matrix = get_distance_matrix(problem)
    # Number of cities
    num_cities = problem.dimension
    # Initial tour
    tour = [starting_node]

    # Rollout Algorithm run for num_cities - 1 steps
    for _ in range(num_cities - 1):
        # Initialize a copy to current tour
        current_tour = tour.copy()
        # What we want optimize, rollout cost
        best_rollout_cost = np.inf
        # Best next city!
        best_next_city = None
        
        # Run over cities not visiteds
        for j in set(range(num_cities)) - set(tour):
            # Adding candidate next city
            current_tour.append(j)

            # Run Base Policy, Nearest Neighbor
            nn_tour = nearest_neighbor(current_tour.copy(), distance_matrix.copy())
            rollout_cost = calculate_tour_cost(nn_tour, distance_matrix)
            # Tests to optimize costs.
            if rollout_cost < best_rollout_cost:
                best_rollout_cost = rollout_cost
                best_next_city = j
            
            # Remove cadidate
            current_tour.pop()
        # Adding best next city
        tour.append(best_next_city)
    # End of algorithm with start city
    tour.append(starting_node)

    return tour

def experiments_with(problem):

    # Core of experiments
    # execute rollout algorithm and calculate time
    start = time.time()
    rollout_tour = rollout_algorithm(problem)
    rollout_time = time.time() - start
    rollout_cost = calculate_tour_cost(rollout_tour)
    
    # execute nearest neighbor algorithm and calculate time
    start = time.time()
    nn_tour = nearest_neighbor([0], distance_matrix.copy())
    nn_time = time.time() - start
    nn_cost = calculate_tour_cost(nn_tour)

    return rollout_cost, rollout_time, nn_cost, nn_time

#%%
# for folder in folders:
for instance in os.listdir(f'../instances/tsp_data')[:4]:
    if instance[-3:] != 'tsp': continue

    file_path = f'../instances/tsp_data/{instance}'
    
    # print(instance)
    problem = tsplib95.load(file_path)
    
    if problem.dimension > 500: continue

    distance_matrix = get_distance_matrix(problem)

    print('OK ', instance)
    print(distance_matrix[0][:5])


# %%
