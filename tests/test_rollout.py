import unittest
from . import rollout
import tsplib95

class TestRolloutFunctions(unittest.TestCase):
    self.problem1 = tsplib95.load('../instances/tsp_data/bays29.tsp')
    self.problem2 = tsplib95.load('../instances/tsp_data/brazil58.tsp')
    self.problem3 = tsplib95.load('../instances/tsp_data/eil101.tsp')

    def test_get_distance_matrix(self):
        self.assertEqual(self.problem1.n, 29)
        pass

    def test_nearest_neighbor(self):
        # rollout.new_rollout.nearest_neighbor
        pass
