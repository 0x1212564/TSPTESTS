import pandas as pd
import matplotlib.pyplot as plt
from tsp_algorithms import *

def run_benchmark():
    """
    Run a benchmark comparison of all TSP algorithms on different restaurant sizes.
    Results are printed as a table and saved as CSV.
    """
    # Generate datasets
    datasets = generate_restaurant_datasets()
    
    # Initialize results dictionary
    results = {
        'Algorithm': [],
        'Restaurant Size': [],
        'Computation Time (s)': [],
        'Route Distance': [],
        'Vertices': []
    }
    
    # Algorithms to benchmark
    algorithms = {
        'Brute Force': brute_force_tsp,
        'Nearest Neighbor': nearest_neighbor_tsp,
        '2-opt Local Search': two_opt_tsp,
        'Genetic Algorithm': genetic_algorithm_tsp,
        'Simulated Annealing': simulated_annealing_tsp
    }
    
    # Run benchmarks
    for size_name, dataset in datasets.items():
        n_vertices = len(dataset)
        print(f"\nRunning benchmarks for {size_name} restaurant ({n_vertices} vertices):")
        
        for algo_name, algo_func in algorithms.items():
            print(f"  Running {algo_name}...", end="", flush=True)
            
            # Skip brute force for medium and large restaurants
            if algo_name == 'Brute Force' and size_name != 'small' and n_vertices > 10:
                print(" SKIPPED (too large for brute force)")
                continue
            
            # Run algorithm multiple times for stochastic algorithms
            if algo_name in ['Genetic Algorithm', 'Simulated Annealing']:
                times = []
                distances = []
                
                # Number of runs for stochastic algorithms
                n_runs = 10
                
                for _ in range(n_runs):
                    solution, computation_time = algo_func(dataset)
                    times.append(computation_time)
                    distances.append(solution.distance)
                
                avg_time = np.mean(times)
                avg_distance = np.mean(distances)
                best_distance = min(distances)
                
                print(f" done in {avg_time:.4f}s (avg of {n_runs} runs), " +
                      f"Avg Dist: {avg_distance:.2f}, Best Dist: {best_distance:.2f}")
                
                # Record average results
                results['Algorithm'].append(f"{algo_name} (Avg)")
                results['Restaurant Size'].append(size_name)
                results['Computation Time (s)'].append(avg_time)
                results['Route Distance'].append(avg_distance)
                results['Vertices'].append(n_vertices)
                
                # Record best results
                results['Algorithm'].append(f"{algo_name} (Best)")
                results['Restaurant Size'].append(size_name)
                results['Computation Time (s)'].append(avg_time)  # Same computation time
                results['Route Distance'].append(best_distance)
                results['Vertices'].append(n_vertices)
                
            else:
                # Run deterministic algorithm once
                solution, computation_time = algo_func(dataset)
                
                print(f" done in {computation_time:.4f}s, Distance: {solution.distance:.2f}")
                
                # Record results
                results['Algorithm'].append(algo_name)
                results['Restaurant Size'].append(size_name)
                results['Computation Time (s)'].append(computation_time)
                results['Route Distance'].append(solution.distance)
                results['Vertices'].append(n_vertices)
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Print results table
    print("\nBenchmark Results:")
    print(df.to_string(index=False))
    
    # Save results to CSV
    df.to_csv('tsp_benchmark_results.csv', index=False)
    print("\nResults saved to tsp_benchmark_results.csv")
    
    # Plot results
    plot_benchmark_results(df)
    
    return df

def plot_benchmark_results(df):
    """Plot benchmark results"""
    # Create figure with subplots for the main comparison charts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Filter out averaged results for Genetic Algorithm and Simulated Annealing
    df_filtered = df[~df['Algorithm'].str.contains('Avg')]
    
    # Plot Computation Time comparison
    for size in ['small', 'medium', 'large', 'larger', 'mega']:
        size_df = df_filtered[df_filtered['Restaurant Size'] == size]
        ax1.bar(size_df['Algorithm'] + f" ({size})", size_df['Computation Time (s)'])
    
    ax1.set_title('Computation Time Comparison')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_xlabel('Algorithm')
    ax1.set_yscale('log')  # Log scale for better visualization
    ax1.tick_params(axis='x', rotation=90)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Plot Route Quality comparison
    for size in ['small', 'medium', 'large', 'larger', 'mega']:
        size_df = df_filtered[df_filtered['Restaurant Size'] == size]
        ax2.bar(size_df['Algorithm'] + f" ({size})", size_df['Route Distance'])
    
    ax2.set_title('Route Quality Comparison')
    ax2.set_ylabel('Route Distance')
    ax2.set_xlabel('Algorithm')
    ax2.tick_params(axis='x', rotation=90)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('tsp_benchmark_results.png')
    print("Benchmark plots saved to tsp_benchmark_results.png")
    plt.close()
    
    # Create a separate figure for the growth rate graph
    plt.figure(figsize=(12, 8))
    
    # Get unique algorithms and create a color map
    algorithms = df_filtered['Algorithm'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(algorithms)))
    
    # Plot each algorithm's distance growth
    for i, algo in enumerate(algorithms):
        algo_df = df_filtered[df_filtered['Algorithm'] == algo]
        if len(algo_df) > 1:  # Only plot if we have multiple data points
            # Sort by number of vertices
            algo_df = algo_df.sort_values('Vertices')
            plt.plot(algo_df['Vertices'], algo_df['Route Distance'], 
                    marker='o', linestyle='-', linewidth=2, 
                    color=colors[i], label=algo)
    
    plt.title('Route Distance Growth by Problem Size', fontsize=14)
    plt.xlabel('Number of Vertices', fontsize=12)
    plt.ylabel('Route Distance', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best', fontsize=10)
    
    # Add annotations for each data point
    for algo in algorithms:
        algo_df = df_filtered[df_filtered['Algorithm'] == algo]
        if len(algo_df) > 1:
            for _, row in algo_df.iterrows():
                plt.annotate(f"{row['Restaurant Size']}", 
                           (row['Vertices'], row['Route Distance']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8)
    
    plt.tight_layout()
    plt.savefig('tsp_growth_rate.png')
    print("Growth rate plot saved to tsp_growth_rate.png")
    plt.close()