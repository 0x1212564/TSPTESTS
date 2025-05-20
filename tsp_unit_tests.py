import unittest
import matplotlib.pyplot as plt
from tsp_algorithms import *

class TSPAlgorithmTests(unittest.TestCase):
    def setUp(self):
        """Set up test cases"""
        # Simple test case with known optimal solution (square with kitchen at center)
        self.test_case_small = [
            (0, 0),    # kitchen
            (1, 1),    # table 1
            (1, -1),   # table 2
            (-1, -1),  # table 3
            (-1, 1)    # table 4
        ]
        
        # Known optimal solution for small test case
        # From kitchen (0,0) -> table 1 -> table 2 -> table 3 -> table 4 -> kitchen
        # This forms a clockwise or counterclockwise route around the square
        self.optimal_route_small = [0, 1, 2, 3, 4, 0]  # clockwise
        self.alternative_optimal_route_small = [0, 4, 3, 2, 1, 0]  # counterclockwise
        
        # Generate restaurant datasets
        self.datasets = generate_restaurant_datasets()
    
    def test_euclidean_distance(self):
        """Test euclidean distance calculation"""
        p1 = (0, 0)
        p2 = (3, 4)
        self.assertEqual(euclidean_distance(p1, p2), 5.0)
    
    def test_calculate_route_distance(self):
        """Test route distance calculation"""
        distances = create_distance_matrix(self.test_case_small)
        
        # Test clockwise route
        route_distance = calculate_route_distance(self.optimal_route_small, distances)
      
        expected_distance = 2 * np.sqrt(2) + 6
        self.assertAlmostEqual(route_distance, expected_distance, places=5)
        
        # Test counterclockwise route (should be same distance)
        route_distance = calculate_route_distance(self.alternative_optimal_route_small, distances)
        self.assertAlmostEqual(route_distance, expected_distance, places=5)
    
    def test_brute_force_tsp(self):
        """Test brute force algorithm on small instance with known solution"""
        solution, time = brute_force_tsp(self.test_case_small)
        
        # Check if solution is one of the two optimal routes
        is_optimal = ((solution.route == self.optimal_route_small) or 
                       (solution.route == self.alternative_optimal_route_small))
        self.assertTrue(is_optimal)
        
        # Check if distance is correct
    
        expected_distance = 2 * np.sqrt(2) + 6
        self.assertAlmostEqual(solution.distance, expected_distance, places=5)
        
        # Test that it throws no errors on small dataset
        brute_force_tsp(self.datasets['small'][:5])  # Use just 4 tables for speed
    
    def test_nearest_neighbor_tsp(self):
        """Test nearest neighbor algorithm for feasibility"""
        for dataset_name, dataset in self.datasets.items():
            solution, time = nearest_neighbor_tsp(dataset)
            
            # Check if route is valid
            self.assertTrue(self._is_valid_route(solution.route, len(dataset)))
            
            # Check if distance calculation matches the route
            distances = create_distance_matrix(dataset)
            calculated_distance = calculate_route_distance(solution.route, distances)
            self.assertAlmostEqual(solution.distance, calculated_distance)
    
    def test_two_opt_tsp(self):
        """Test 2-opt local search algorithm for feasibility and improvement"""
        for dataset_name, dataset in self.datasets.items():
            # Get nearest neighbor solution first
            nn_solution, _ = nearest_neighbor_tsp(dataset)
            
            # Apply 2-opt
            two_opt_solution, time = two_opt_tsp(dataset)
            
            # Check if route is valid
            self.assertTrue(self._is_valid_route(two_opt_solution.route, len(dataset)))
            
            # Check if distance calculation matches the route
            distances = create_distance_matrix(dataset)
            calculated_distance = calculate_route_distance(two_opt_solution.route, distances)
            self.assertAlmostEqual(two_opt_solution.distance, calculated_distance)
            
            # Check if 2-opt improves on nearest neighbor solution or equals it
            self.assertLessEqual(two_opt_solution.distance, nn_solution.distance)
    
    def test_genetic_algorithm_tsp(self):
        """Test genetic algorithm for feasibility"""
        for dataset_name, dataset in self.datasets.items():
            # Run multiple times due to stochastic nature
            for _ in range(3):
                solution, time = genetic_algorithm_tsp(
                    dataset, 
                    pop_size=50,  # Smaller for tests
                    max_generations=50  # Fewer generations for tests
                )
                
                # Check if route is valid
                self.assertTrue(self._is_valid_route(solution.route, len(dataset)))
                
                # Check if distance calculation matches the route
                distances = create_distance_matrix(dataset)
                calculated_distance = calculate_route_distance(solution.route, distances)
                self.assertAlmostEqual(solution.distance, calculated_distance)
    
    def test_simulated_annealing_tsp(self):
        """Test simulated annealing for feasibility"""
        for dataset_name, dataset in self.datasets.items():
            # Run multiple times due to stochastic nature
            for _ in range(3):
                solution, time = simulated_annealing_tsp(
                    dataset,
                    initial_temp=100,  # Lower for tests
                    iterations_per_temp=10  # Fewer iterations for tests
                )
                
                # Check if route is valid
                self.assertTrue(self._is_valid_route(solution.route, len(dataset)))
                
                # Check if distance calculation matches the route
                distances = create_distance_matrix(dataset)
                calculated_distance = calculate_route_distance(solution.route, distances)
                self.assertAlmostEqual(solution.distance, calculated_distance)
    
    def test_deterministic_consistency(self):
        """Test that deterministic algorithms produce consistent results"""
        # Test brute force on very small dataset
        tiny_dataset = self.test_case_small
        
        # Run brute force twice
        solution1, _ = brute_force_tsp(tiny_dataset)
        solution2, _ = brute_force_tsp(tiny_dataset)
        
        # Check routes and distances are the same
        self.assertEqual(solution1.route, solution2.route)
        self.assertEqual(solution1.distance, solution2.distance)
        
        # Test nearest neighbor
        solution1, _ = nearest_neighbor_tsp(self.datasets['small'])
        solution2, _ = nearest_neighbor_tsp(self.datasets['small'])
        
        # Check routes and distances are the same
        self.assertEqual(solution1.route, solution2.route)
        self.assertEqual(solution1.distance, solution2.distance)
        
        # Test 2-opt
        solution1, _ = two_opt_tsp(self.datasets['small'])
        solution2, _ = two_opt_tsp(self.datasets['small'])
        
        # Check routes and distances are the same
        self.assertEqual(solution1.route, solution2.route)
        self.assertEqual(solution1.distance, solution2.distance)
    
    def test_performance_bounds(self):
        """Test that algorithms complete within reasonable time limits"""
        # Test brute force on small problem (should be fast)
        start_time = time.time()
        brute_force_tsp(self.test_case_small)
        elapsed = time.time() - start_time
        self.assertLess(elapsed, 1.0)  # Should complete in under 1 second
        
        # Test nearest neighbor on large problem
        start_time = time.time()
        nearest_neighbor_tsp(self.datasets['large'])
        elapsed = time.time() - start_time
        self.assertLess(elapsed, 1.0)  # Should complete in under 1 second
        
        # Test 2-opt on large problem
        start_time = time.time()
        two_opt_tsp(self.datasets['large'])
        elapsed = time.time() - start_time
        self.assertLess(elapsed, 10.0)  # Should complete in under 10 seconds
        
        # Genetic algorithm with reduced parameters
        start_time = time.time()
        genetic_algorithm_tsp(self.datasets['large'], pop_size=20, max_generations=20)
        elapsed = time.time() - start_time
        self.assertLess(elapsed, 10.0)  # Should complete in under 10 seconds
        
        # Simulated annealing with reduced parameters
        start_time = time.time()
        simulated_annealing_tsp(self.datasets['large'], iterations_per_temp=10)
        elapsed = time.time() - start_time
        self.assertLess(elapsed, 10.0)  # Should complete in under 10 seconds
    
    def _is_valid_route(self, route, n_vertices):
        """
        Check if a route is valid:
        1. Starts and ends at kitchen (vertex 0)
        2. Visits each vertex exactly once (except kitchen)
        3. Contains the correct number of vertices
        """
        # Check if route starts and ends at kitchen (vertex 0)
        if route[0] != 0 or route[-1] != 0:
            return False
        
        # Check if route has correct length
        if len(route) != n_vertices + 1:  # n_vertices + 1 because kitchen appears twice
            return False
        
        # Check if route visits each vertex exactly once (except kitchen)
        visited = set(route)
        if len(visited) != n_vertices:
            return False
        
        # Check if all required vertices are in the route
        for i in range(n_vertices):
            if i not in visited:
                return False
        
        return True

class TSPVisualizationTests(unittest.TestCase):
    def setUp(self):
        """Set up visualization tests"""
        self.datasets = generate_restaurant_datasets()
    
    def test_visualization(self):
        """Test visualization of routes"""
        # Only run on small dataset for speed
        dataset = self.datasets['small']
        
        # Get solutions from different algorithms
        nn_solution, _ = nearest_neighbor_tsp(dataset)
        two_opt_solution, _ = two_opt_tsp(dataset)
        ga_solution, _ = genetic_algorithm_tsp(dataset, pop_size=50, max_generations=50)
        sa_solution, _ = simulated_annealing_tsp(dataset, iterations_per_temp=10)
        
        # Visualize routes (commented out for CI but can be used manually)
        
        self._visualize_route(dataset, nn_solution.route, "Nearest Neighbor")
        self._visualize_route(dataset, two_opt_solution.route, "2-opt Local Search")
        self._visualize_route(dataset, ga_solution.route, "Genetic Algorithm")
        self._visualize_route(dataset, sa_solution.route, "Simulated Annealing")
        
        
        # Just a placeholder assertion since we're not actually showing the plots
        self.assertTrue(True)
    
    def _visualize_route(self, coordinates, route, title):
        """Visualize a route on a 2D plot"""
        plt.figure(figsize=(10, 8))
        
        # Extract x and y coordinates
        x = [coordinates[i][0] for i in route]
        y = [coordinates[i][1] for i in route]
        
        # Plot the route
        plt.plot(x, y, 'b-', linewidth=1)
        plt.plot(x, y, 'ro', markersize=5)
        
        # Highlight kitchen
        plt.plot(coordinates[0][0], coordinates[0][1], 'gs', markersize=10)
        
        # Add labels
        for i, point in enumerate(route):
            plt.annotate(str(point), (coordinates[point][0], coordinates[point][1]))
        
        plt.title(f"{title} - Route Distance: {calculate_route_distance(route, create_distance_matrix(coordinates)):.2f}")
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    unittest.main()