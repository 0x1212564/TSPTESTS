import argparse
import matplotlib.pyplot as plt
from tsp_algorithms import *
from tsp_benchmark import run_benchmark

def visualize_route(coordinates, route, title, save_path=None):
    """
    Visualize a route on a 2D plot.
    
    Args:
        coordinates: List of (x, y) coordinates
        route: List of vertex indices representing the route
        title: Title for the plot
        save_path: If provided, save the plot to this file
    """
    plt.figure(figsize=(15, 12))
    
    # Extract x and y coordinates
    x = [coordinates[i][0] for i in route]
    y = [coordinates[i][1] for i in route]
    
    # Plot the route
    plt.plot(x, y, 'b-', linewidth=1.5, label='Route')
    
    # Plot vertices
    plt.scatter([coord[0] for coord in coordinates], 
                [coord[1] for coord in coordinates], 
                c='red', s=50, zorder=5)
    
    # Highlight kitchen
    plt.scatter(coordinates[0][0], coordinates[0][1], c='green', s=100, marker='s', zorder=6, label='Kitchen')
    
    # Highlight route direction
    arrow_positions = list(range(1, len(route)-1, max(1, len(route)//10)))
    for i in arrow_positions:
        dx = x[i+1] - x[i]
        dy = y[i+1] - y[i]
        # Calculate midpoint for arrow placement
        midx = (x[i] + x[i+1]) / 2
        midy = (y[i] + y[i+1]) / 2
        plt.arrow(midx - dx/8, midy - dy/8, dx/4, dy/4, 
                  head_width=0.5, head_length=0.7, fc='blue', ec='blue', zorder=7)
    
    # Add labels
    for i, coord in enumerate(coordinates):
        plt.annotate(f"{i}", (coord[0], coord[1]), xytext=(5, 5), 
                     textcoords='offset points', fontsize=10, zorder=8)
    
    # Calculate route distance
    distances = create_distance_matrix(coordinates)
    distance = calculate_route_distance(route, distances)
    
    plt.title(f"{title}\nRoute Distance: {distance:.2f}")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal')
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Route visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def run_all_algorithms(dataset_name='small', visualize=True):
    """
    Run all TSP algorithms on a given dataset and optionally visualize the results.
    
    Args:
        dataset_name: Name of the dataset ('small', 'medium', or 'large')
        visualize: Whether to visualize the routes
    """
    # Generate datasets
    datasets = generate_restaurant_datasets()
    dataset = datasets.get(dataset_name)
    if not dataset:
        print(f"Error: Dataset '{dataset_name}' not found.")
        return
    
    print(f"Running all algorithms on {dataset_name} restaurant dataset ({len(dataset)} vertices)...")
    
    # Algorithms to run
    algorithms = {
        'Brute Force': brute_force_tsp,
        'Nearest Neighbor': nearest_neighbor_tsp,
        '2-opt Local Search': two_opt_tsp,
        'Genetic Algorithm': genetic_algorithm_tsp,
        'Simulated Annealing': simulated_annealing_tsp
    }

    # Skip brute force for large datasets
    if dataset_name != 'small' and len(dataset) > 10:
        print("Skipping Brute Force algorithm (too many vertices).")
        algorithms.pop('Brute Force')

    results = []
    
    # Run each algorithm
    for name, algo in algorithms.items():
        print(f"  Running {name}...")
        start_time = time.time()
        
        if name in ['Genetic Algorithm', 'Simulated Annealing']:
            # Run stochastic algorithms multiple times
            best_solution = None
            best_distance = float('inf')
            
            for i in range(10):
                solution, _ = algo(dataset)
                if solution.distance < best_distance:
                    best_distance = solution.distance
                    best_solution = solution
            
            solution = best_solution
        else:
            # Run deterministic algorithm once
            solution, _ = algo(dataset)
        
        elapsed_time = time.time() - start_time
        
        print(f"    Completed in {elapsed_time:.15f} seconds")
        print(f"    Route: {solution.route}")
        print(f"    Distance: {solution.distance:.2f}")
        
        results.append((name, solution, elapsed_time))
        
        # Visualize route
        if visualize:
            visualize_route(dataset, solution.route, 
                          f"{name} - {dataset_name.capitalize()} Restaurant",
                          save_path=f"route_{dataset_name}_{name.replace(' ', '_').lower()}.png")
    
    # Return results
    return results

def main():
    """Main function to parse arguments and run the appropriate function"""
    parser = argparse.ArgumentParser(description='TSP Algorithm Comparison for Restaurant Delivery Routes')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark comparing all algorithms')
    parser.add_argument('--restaurant', choices=['small', 'medium', 'large', 'mega', 'larger'], default='large',
                        help='Restaurant size for algorithm testing')
    parser.add_argument('--no-visualize', action='store_true', help='Disable route visualization')
    
    args = parser.parse_args()
    
    if args.benchmark:
        # Run benchmark
        run_benchmark()
    else:
        # Run algorithms on specified restaurant size
        run_all_algorithms(dataset_name=args.restaurant, visualize=not args.no_visualize)

if __name__ == "__main__":
    main()