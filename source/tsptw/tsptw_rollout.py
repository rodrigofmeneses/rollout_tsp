import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

def read_data(path):
    instance_author = path.split('/')[-2]
    coordinates = []
    intervals = []

    if instance_author == 'DaSilvaUrrutia':
        with open(path, 'r') as reader:
            reader.readline()
            for line in reader:
                _, x, y, _, a, b, _ = line.split()
                
                coordinates.append((float(x), float(y)))
                intervals.append((float(a), float(b)))

    elif instance_author == 'DumasEtAl':
        with open(path, 'r') as reader:        
            [reader.readline() for i in range(6)]
            for line in reader:
                _, x, y, _, a, b, _ = line.split()
                    
                coordinates.append((float(x), float(y)))
                intervals.append((float(a), float(b)))
            
            coordinates.pop()
            intervals.pop()

    elif instance_author == 'GendreauEtAl':
        with open(path, 'r') as reader:        
            [reader.readline() for i in range(6)]
            for line in reader:
                _, x, y, _, a, b, _ = line.split()
                    
                coordinates.append((float(x), float(y)))
                intervals.append((float(a), float(b)))
            
            coordinates.pop()
            intervals.pop()

    elif instance_author == 'OhlmannThomas':
        with open(path, 'r') as reader:        
            [reader.readline() for i in range(6)]
            for line in reader:
                _, x, y, _, a, b, _ = line.split()
                    
                coordinates.append((float(x), float(y)))
                intervals.append((float(a), float(b)))
            
            coordinates.pop()
            intervals.pop()

    problem = {
        'coordinates': coordinates,
        'intervals': intervals,
        'distance_matrix': squareform(pdist(coordinates)),
        'dimension': len(coordinates)
    }
    
    return problem



def rollout_algorithm(problem, starting_node=0):
    # Distance Matrix
    distance_matrix = problem['distance_matrix']
    # Number of cities
    num_cities = problem['dimension']
    # Initial solution
    # solution = List([starting_node])
    solution = [starting_node]

    # Rollout Algorithm run for num_cities - 1 steps
    for _ in range(num_cities - 1):
        # Initialize a copy to current solution
        current_solution = solution.copy()
        # What we want optimize, rollout cost
        best_rollout_cost = np.inf
        # Best next city!
        best_next_city = None
        
        # Run over cities not visiteds
        for j in set(range(num_cities)) - set(solution):
            # Adding candidate next city
            current_solution.append(j)

            # Run Base Policy, Nearest Neighbor
            nn_solution = nearest_neighbor(problem, current_solution.copy(), distance_matrix.copy())
            rollout_cost = calculate_solution_cost(nn_solution, List(distance_matrix))
            # Tests to optimize costs.
            if rollout_cost < best_rollout_cost:
                best_rollout_cost = rollout_cost
                best_next_city = j
            
            # Remove cadidate
            current_solution.pop()
        # Adding best next city
        solution.append(best_next_city)
    # End of algorithm with start city
    solution.append(starting_node)

    return solution