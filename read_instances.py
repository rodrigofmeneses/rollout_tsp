import networkx as nx

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