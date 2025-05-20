import numpy as np
import time
import random
import itertools
import math
from typing import List, Tuple

# Data structures
class TSPSolution:
    """Class to represent a TSP solution"""
    def __init__(self, route: List[int], distance: float):
        self.route = route
        self.distance = distance
    
    def __repr__(self):
        return f"Route: {self.route}, Distance: {self.distance:.2f}"

# Helper functions
def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points"""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def calculate_route_distance(route: List[int], distances: np.ndarray) -> float:
    """Calculate the total distance of a route"""
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += distances[route[i]][route[i+1]]
    return total_distance

def create_distance_matrix(coordinates: List[Tuple[float, float]]) -> np.ndarray:
    """Create a distance matrix from coordinates"""
    n = len(coordinates)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                distances[i][j] = euclidean_distance(coordinates[i], coordinates[j])
    return distances

# 1. Brute Force Algorithm
def brute_force_tsp(coordinates: List[Tuple[float, float]]) -> tuple[TSPSolution, float]:
    """
    Solve TSP using brute force approach by checking all possible permutations.
    Args:
        coordinates: List of (x, y) coordinates where index 0 is kitchen and rest are tables
    Returns:
        TSPSolution object with the best route and distance
    """
    start_time = time.time()
    distances = create_distance_matrix(coordinates)
    n = len(coordinates)
    
    # Generate all possible permutations of tables (excluding kitchen at index 0)
    table_indices = list(range(1, n))
    
    best_route = None
    best_distance = float('inf')
    
    # Check all permutations
    for perm in itertools.permutations(table_indices):
        # Complete route starts and ends at the kitchen (index 0)
        route = [0] + list(perm) + [0]
        distance = calculate_route_distance(route, distances)
        
        if distance < best_distance:
            best_distance = distance
            best_route = route
    
    computation_time = time.time() - start_time
    
    return TSPSolution(best_route, best_distance), computation_time

# 2. Nearest Neighbor Algorithm
def nearest_neighbor_tsp(coordinates: List[Tuple[float, float]]) -> tuple[TSPSolution, float]:
    """
    Solve TSP using the nearest neighbor heuristic.
    Args:
        coordinates: List of (x, y) coordinates where index 0 is kitchen and rest are tables
    Returns:
        TSPSolution object with the route and distance
    """
    start_time = time.time()
    distances = create_distance_matrix(coordinates)
    n = len(coordinates)
    
    # Start at the kitchen (vertex 0)
    current_vertex = 0
    unvisited = set(range(1, n))  # All tables except kitchen
    route = [0]  # Start with kitchen
    
    # While there are unvisited tables
    while unvisited:
        # Find closest unvisited vertex
        next_vertex = min(unvisited, key=lambda x: distances[current_vertex][x])
        route.append(next_vertex)
        unvisited.remove(next_vertex)
        current_vertex = next_vertex
    
    # Return to kitchen
    route.append(0)
    
    # Calculate total distance
    total_distance = calculate_route_distance(route, distances)
    
    computation_time = time.time() - start_time
    
    return TSPSolution(route, total_distance), computation_time

# 3. 2-opt Local Search Algorithm
def two_opt_swap(route: List[int], i: int, j: int) -> List[int]:
    """
    Perform a 2-opt swap by reversing the segment between positions i and j in the route.
    Note: This doesn't include the first and last element (kitchen) in the swap.
    """
    # Route segment before position i
    new_route = route[:i]
    # Reversed segment between positions i and j
    new_route.extend(reversed(route[i:j + 1]))
    # Route segment after position j
    new_route.extend(route[j + 1:])
    return new_route

def two_opt_tsp(coordinates: List[Tuple[float, float]]) -> tuple[TSPSolution, float]:
    """
    Solve TSP using 2-opt local search starting from a Nearest Neighbor solution.
    Args:
        coordinates: List of (x, y) coordinates where index 0 is kitchen and rest are tables
    Returns:
        TSPSolution object with the improved route and distance
    """
    start_time = time.time()
    
    # Get initial solution using Nearest Neighbor
    initial_solution, _ = nearest_neighbor_tsp(coordinates)
    route = initial_solution.route
    
    distances = create_distance_matrix(coordinates)
    n = len(route)
    
    # 2-opt improvement
    improvement = True
    while improvement:
        improvement = False
        for i in range(1, n - 2):  # Skip first element (kitchen)
            for j in range(i + 1, n - 1):  # Skip last element (kitchen)
                # Skip adjacent edges
                if j - i == 1:
                    continue
                    
                # Calculate change in distance if we perform the 2-opt swap
                current_distance = (distances[route[i-1]][route[i]] + 
                                    distances[route[j]][route[j+1]])
                new_distance = (distances[route[i-1]][route[j]] + 
                                distances[route[i]][route[j+1]])
                
                if new_distance < current_distance:
                    # Perform the swap if it reduces distance
                    route = two_opt_swap(route, i, j)
                    improvement = True
                    break  # Restart the outer loop
            if improvement:
                break
    
    # Calculate final route distance
    total_distance = calculate_route_distance(route, distances)
    
    computation_time = time.time() - start_time
    
    return TSPSolution(route, total_distance), computation_time

# 4. Genetic Algorithm
def initialize_population(size: int, n_vertices: int) -> List[List[int]]:
    """Initialize a population of random routes"""
    population = []
    for _ in range(size):
        # Create a random permutation of tables (excluding kitchen)
        tables = list(range(1, n_vertices))
        random.shuffle(tables)
        # Complete route starts and ends at the kitchen (index 0)
        route = [0] + tables + [0]
        population.append(route)
    return population

def fitness(route: List[int], distances: np.ndarray) -> float:
    """Calculate fitness as inverse of route distance (higher is better)"""
    distance = calculate_route_distance(route, distances)
    return 1.0 / distance

def selection(population: List[List[int]], fitnesses: List[float], n_parents: int) -> List[List[int]]:
    """Select parents using tournament selection"""
    parents = []
    for _ in range(n_parents):
        # Tournament selection (size 3)
        candidates = random.sample(range(len(population)), 3)
        best_candidate = max(candidates, key=lambda idx: fitnesses[idx])
        parents.append(population[best_candidate])
    return parents

def order_crossover(parent1: List[int], parent2: List[int]) -> List[int]:
    """
    Perform order crossover (OX) between two parents.
    Note: Preserves the positions of kitchen (first and last element)
    """
    # Get only the table part of the route (exclude kitchen at start and end)
    p1_tables = parent1[1:-1]
    p2_tables = parent2[1:-1]
    
    size = len(p1_tables)
    
    # Select random subsequence from parent1
    start, end = sorted(random.sample(range(size), 2))
    
    # Create child with elements from parent1 subsequence
    child_tables = [None] * size
    for i in range(start, end + 1):
        child_tables[i] = p1_tables[i]
    
    # Fill remaining positions with elements from parent2 in order
    j = 0
    for i in range(size):
        if child_tables[i] is None:
            # Find next element in parent2 that's not already in child
            while p2_tables[j] in child_tables:
                j += 1
            child_tables[i] = p2_tables[j]
            j += 1
    
    # Add kitchen at start and end
    return [0] + child_tables + [0]

def swap_mutation(route: List[int], mutation_rate: float) -> List[int]:
    """
    Apply swap mutation with given probability.
    Note: Only mutates table positions (excludes kitchen at start and end)
    """
    if random.random() < mutation_rate:
        # Get only the table part of the route
        tables = route[1:-1]
        if len(tables) > 1:  # Need at least 2 tables to swap
            i, j = random.sample(range(len(tables)), 2)
            tables[i], tables[j] = tables[j], tables[i]
            return [0] + tables + [0]
    return route

def genetic_algorithm_tsp(coordinates: List[Tuple[float, float]], 
                         pop_size: int = 100, 
                         max_generations: int = 200, 
                         crossover_rate: float = 0.8, 
                         mutation_rate: float = 0.02) -> tuple[TSPSolution, float]:
    """
    Solve TSP using a genetic algorithm.
    Args:
        coordinates: List of (x, y) coordinates where index 0 is kitchen and rest are tables
        pop_size: Population size
        max_generations: Maximum number of generations
        crossover_rate: Probability of crossover
        mutation_rate: Probability of mutation
    Returns:
        TSPSolution object with the best route and distance
    """
    start_time = time.time()
    distances = create_distance_matrix(coordinates)
    n_vertices = len(coordinates)
    
    # Initialize population
    population = initialize_population(pop_size, n_vertices)
    
    best_route = None
    best_distance = float('inf')
    
    # Evolution process
    for _ in range(max_generations):
        # Calculate fitness for each individual
        fitnesses = [fitness(route, distances) for route in population]
        
        # Find best individual
        best_idx = fitnesses.index(max(fitnesses))
        current_best = population[best_idx]
        current_distance = 1.0 / fitnesses[best_idx]
        
        # Update global best
        if current_distance < best_distance:
            best_distance = current_distance
            best_route = current_best
        
        # Selection
        parents = selection(population, fitnesses, pop_size)
        
        # Create new population
        new_population = []
        
        # Elitism - keep the best individual
        new_population.append(current_best)
        
        # Crossover and mutation
        for i in range(0, pop_size - 1, 2):
            if i + 1 < len(parents):
                if random.random() < crossover_rate:
                    # Crossover
                    child1 = order_crossover(parents[i], parents[i+1])
                    child2 = order_crossover(parents[i+1], parents[i])
                else:
                    # No crossover
                    child1 = parents[i][:]
                    child2 = parents[i+1][:]
                
                # Mutation
                child1 = swap_mutation(child1, mutation_rate)
                child2 = swap_mutation(child2, mutation_rate)
                
                new_population.extend([child1, child2])
        
        # Ensure population size remains constant
        population = new_population[:pop_size]
    
    computation_time = time.time() - start_time
    
    return TSPSolution(best_route, best_distance), computation_time

# 5. Simulated Annealing
def random_neighbor(route: List[int]) -> List[int]:
    """
    Generate a random neighbor using a 2-opt move.
    Note: Keeps kitchen (index 0) at start and end of route
    """
    # Select two random positions in the route (excluding first and last)
    i, j = sorted(random.sample(range(1, len(route) - 1), 2))
    
    # Apply 2-opt move
    return two_opt_swap(route, i, j)

def simulated_annealing_tsp(coordinates: List[Tuple[float, float]], 
                           initial_temp: float = 1000,
                           cooling_rate: float = 0.95,
                           iterations_per_temp: int = 100) -> tuple[TSPSolution, float]:
    """
    Solve TSP using simulated annealing.
    Args:
        coordinates: List of (x, y) coordinates where index 0 is kitchen and rest are tables
        initial_temp: Initial temperature
        cooling_rate: Cooling rate
        iterations_per_temp: Number of iterations at each temperature
    Returns:
        TSPSolution object with the best route and distance
    """
    start_time = time.time()
    distances = create_distance_matrix(coordinates)
    
    # Get initial solution using Nearest Neighbor
    initial_solution, _ = nearest_neighbor_tsp(coordinates)
    current_route = initial_solution.route
    current_distance = initial_solution.distance
    
    best_route = current_route[:]
    best_distance = current_distance
    
    # Simulated annealing process
    temp = initial_temp
    final_temp = 0.01  # Stop when temperature is very low
    
    while temp > final_temp:
        for _ in range(iterations_per_temp):
            # Generate neighbor solution
            new_route = random_neighbor(current_route)
            new_distance = calculate_route_distance(new_route, distances)
            
            # Calculate change in energy (distance)
            delta_e = new_distance - current_distance
            
            # Accept if better, or with probability e^(-delta_e/temp)
            if (delta_e < 0) or (random.random() < math.exp(-delta_e / temp)):
                current_route = new_route
                current_distance = new_distance
                
                # Update best solution if needed
                if current_distance < best_distance:
                    best_route = current_route[:]
                    best_distance = current_distance
        
        # Cool down
        temp *= cooling_rate
    
    computation_time = time.time() - start_time
    
    return TSPSolution(best_route, best_distance), computation_time

# Generate test datasets
def generate_restaurant_datasets():
    """Generate test datasets for restaurants of different sizes"""
    # Random seed for reproducibility
    np.random.seed(42069)
    
    # Kitchen position is fixed
    kitchen = (0, 0)
    
    # Small restaurant: 6 tables
    small_tables = [(np.random.uniform(-10, 10), np.random.uniform(-10, 10)) for _ in range(6)]
    small_restaurant = [kitchen] + small_tables
    
    # Medium restaurant: 12 tables
    medium_tables = [(np.random.uniform(-15, 15), np.random.uniform(-15, 15)) for _ in range(12)]
    medium_restaurant = [kitchen] + medium_tables
    
    # Large restaurant: 20 tables
    large_tables = [(np.random.uniform(-20, 20), np.random.uniform(-20, 20)) for _ in range(20)]
    large_restaurant = [kitchen] + large_tables

    # Larger restaurant: 50 tables
    larger_tables = [(np.random.uniform(-50, 50), np.random.uniform(-50, 50)) for _ in range(50)]
    larger_restaurant = [kitchen] + larger_tables

    # Mega restaurant: 100 tables
    mega_tables = [(np.random.uniform(-100, 100), np.random.uniform(-100, 100)) for _ in range(100)]
    mega_restaurant = [kitchen] + mega_tables




    return {
        'small': small_restaurant,
        'medium': medium_restaurant,
        'large': large_restaurant,
        'larger': larger_restaurant,
        'mega': mega_restaurant,

    }