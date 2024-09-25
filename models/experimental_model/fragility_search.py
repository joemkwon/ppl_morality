import numpy as np
import random
from collections import deque
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import csv
import argparse

# Global variables
example_gridworld_this_is_not_used_in_the_model = [
    ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S'],
    ['S', 'G', 'G', 'G', 'S', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'G', 'G', 'G', 'S', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'G', 'G', 'G', 'S', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'G', 'G', 'G', 'S', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'G', 'G', 'G', 'S', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'G', 'G', 'G', 'S', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'G', 'G', 'G', 'S', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'G', 'G', 'G', 'S', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S']
]

# Number of grass steps before the grass dies. In the future, this will be sampled from a distribution
grass_capacity = 100
# Number of agents that walk through the gridworld on a given day
population = 100
# Cost to community when the grass dies
grass_community_cost = 10000

# Define the goals and their corresponding proportions based on experiment data
goal_proportions = {
    'a friend': 50.5,
    'pain': 34.9,
    'ice cream': 42.6,
    'vac clinic': 28.7,
    'porta-potty': 30.7,
    'police car': 11.1
}

# Normalize the proportions so they sum to 1 (since we are sampling probabilities)
total_proportion = sum(goal_proportions.values())
goal_probabilities = {goal: proportion / total_proportion for goal, proportion in goal_proportions.items()}

# Define the goals and their utilities (mean and SE from the experiment data)
goal_utilities = {
    'a friend': (52.7, 3.87),
    'pain': (82.6, 3.87),
    'ice cream': (46.0, 3.87),
    'vac clinic': (48.0, 3.87),
    'porta-potty': (65.8, 3.87),
    'police car': (63.1, 3.87)
}


# Below are the functions for the model

# Calculate the utility of a given trajectory
def calculate_plan_utility(trajectory, goal_utility):
    reward_for_goal = goal_utility if trajectory[-1]['type'] == 'F' else 0
    cost_for_path_length = 0
    
    for step in trajectory:
        action = step['action']
        
        # Check if the movement is diagonal
        if action not in ['north', 'east', 'west', 'south']:
            cost_for_path_length += np.sqrt(2)
        else:
            cost_for_path_length += 1
    
    return reward_for_goal - cost_for_path_length

# Calculate the universalized utility of a plan
def universalized_plan_utility(gridworld, my_goal_utility, my_grass_shortcutiness, grass_capacity, population):
    total_grass_steps = 0
    total_utility = 0
    
    # Use grass_capacity and population from parameters, not global variables
    for agent_id in range(population):
        goal = sample_goal()
        goal_utility = sample_goal_utility(goal)

        # Only simulating agents which have goal utilities greater than or equal to own
        if goal_utility < my_goal_utility:
            continue

        starting_location = sample_location(gridworld)
        goal_location = sample_location(gridworld)
        
        # Resample goal_location if it is the same as starting_location
        while goal_location == starting_location:
            goal_location = sample_location(gridworld)
        
        agent_utility = 0
        agent_utility += goal_utility

        # One way of doing it, which is confusing to me -- discuss with Lio
        # Instead of doing the full shortest path plan, we approximate the number of grass steps by using the the rule-following length of sidewalk-only and the grass shortcutiness parameter
        length_strict_rule_following = len(full_rule_following_trajectory(gridworld, starting_location, goal_location))
        grass_steps = my_grass_shortcutiness * length_strict_rule_following

        total_grass_steps += int(grass_steps)
        total_utility += agent_utility
    
    # Use grass_capacity from parameters
    if total_grass_steps > grass_capacity:
        total_utility -= grass_community_cost

    return total_utility / population

def get_neighbors(gridworld, x, y, include_grass=False):
    directions = [
        ('west', -1, 0), ('east', 1, 0), 
        ('north', 0, -1), ('south', 0, 1),
        ('north-west', -1, -1), ('north-east', 1, -1),
        ('south-west', -1, 1), ('south-east', 1, 1)
    ]
    neighbors = []
    for direction, dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < len(gridworld[0]) and 0 <= ny < len(gridworld):
            if include_grass or gridworld[ny][nx] == 'S':
                neighbors.append((nx, ny, direction))
    return neighbors

# Calculate the full rule-following trajectory, when only moving along sidewalk squares, using BFS
def full_rule_following_trajectory(gridworld, start, goal):
    start_x, start_y = start
    goal_x, goal_y = goal
    queue = deque([(start_x, start_y, None, [])])  # Added None for initial action
    visited = set((start_x, start_y))

    while queue:
        x, y, action, path = queue.popleft()
        if (x, y) == (goal_x, goal_y):
            # Append the final step to the path
            path.append({'coordinate': (x, y), 'type': gridworld[y][x], 'action': action})
            return path

        for nx, ny, direction in get_neighbors(gridworld, x, y):
            if (nx, ny) not in visited:
                visited.add((nx, ny))
                new_path = path + [{'coordinate': (x, y), 'type': gridworld[y][x], 'action': direction}]
                queue.append((nx, ny, direction, new_path))

    return []  # Return an empty path if no valid path is found

# Calculate the shortest path trajectory using BFS except we don't care if we go through grass or not
def shortest_path_trajectory(gridworld, start, goal):
    start_x, start_y = start
    goal_x, goal_y = goal
    queue = deque([(start_x, start_y, None, [])])
    visited = set((start_x, start_y))

    while queue:
        x, y, action, path = queue.popleft()
        if (x, y) == (goal_x, goal_y):
            path.append({'coordinate': (x, y), 'type': gridworld[y][x], 'action': action})
            return path

        for nx, ny, direction in get_neighbors(gridworld, x, y, include_grass=True):
            if (nx, ny) not in visited:
                visited.add((nx, ny))
                new_path = path + [{'coordinate': (x, y), 'type': gridworld[y][x], 'action': direction}]
                queue.append((nx, ny, direction, new_path))

    return []

def calculate_shortcutiness(gridworld, trajectory):
    start = trajectory[0]['coordinate']
    end = trajectory[-1]['coordinate']
    
    rule_following_traj = full_rule_following_trajectory(gridworld, start, end)
    shortest_traj = shortest_path_trajectory(gridworld, start, end)
    
    given_length = len(trajectory)
    rule_following_length = len(rule_following_traj)
    shortest_length = len(shortest_traj)
    
    # Normalize shortcutiness
    if rule_following_length == shortest_length:
        return 0  
    else:
        shortcutiness = (rule_following_length - given_length) / (rule_following_length - shortest_length)
        return max(0, min(1, shortcutiness))  


# Sample a location from the gridworld, specifically a sidewalk square 'S' from the perimeter for now
def sample_location(gridworld):
    # The starting location can be any sidewalk square 'S' that is on the perimeter of the gridworld
    perimeter_sidewalk_squares = []

    width = len(gridworld[0])  # Number of columns
    height = len(gridworld)    # Number of rows

    # Top and bottom rows
    for x in range(width):
        if gridworld[0][x] == 'S':
            perimeter_sidewalk_squares.append((x, 0))
        if gridworld[height - 1][x] == 'S':
            perimeter_sidewalk_squares.append((x, height - 1))

    # Left and right columns (excluding corners already checked)
    for y in range(1, height - 1):
        if gridworld[y][0] == 'S':
            perimeter_sidewalk_squares.append((0, y))
        if gridworld[y][width - 1] == 'S':
            perimeter_sidewalk_squares.append((width - 1, y))

    return random.choice(perimeter_sidewalk_squares) if perimeter_sidewalk_squares else None

# Sample a goal from the 6 possible goals based on their probabilities
def sample_goal():
    goals = list(goal_probabilities.keys())
    probabilities = list(goal_probabilities.values())
    return np.random.choice(goals, p=probabilities)

# Sample a goal utility from the goal distribution (mean and standard error)
def sample_goal_utility(goal):
    mean, se = goal_utilities[goal]
    return np.random.normal(mean, se)

def save_forward_model_table(data, file_path):
    serializable_data = {}
    for metadata_key, table in data.items():
        # Convert the metadata key from a string back to a dictionary
        metadata = json.loads(metadata_key)
        # Convert numpy float64 to regular float for JSON serialization
        serializable_table = {
            str(k): float(v) for k, v in table.items()
        }
        serializable_data[json.dumps(metadata)] = serializable_table
    
    with open(file_path, 'w') as f:
        json.dump(serializable_data, f)

def load_forward_model_table(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Convert the loaded data back to the original format
    return {
        metadata_key: {
            tuple(map(float, k.strip('()').split(','))): v for k, v in table.items()
        }
        for metadata_key, table in data.items()
    }

def load_gridworld_and_trajectory(file_name):
    with open(file_name, 'r') as file:
        data = json.load(file)
    return data['gridworld'], data['trajectory']

def plot_myopia_inference(myopia_values, posterior_probabilities, file_name, goal_utility, goal_name):
    plt.figure(figsize=(10, 6))
    plt.plot(myopia_values, posterior_probabilities)
    plt.xlabel('Myopia Parameter')
    plt.ylabel('Posterior Probability')
    plt.title(f'Inferred Posterior Distribution of Myopia Parameter for Goal: {goal_name}')
    
    # Set y-axis limits
    plt.ylim(0, 1)
    
    # Add text box with additional information
    info_text = f"File: {file_name}\nBins: {len(myopia_values)}\nGoal Utility: {goal_utility:.2f}"
    plt.text(0.95, 0.95, info_text, transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Define the output filename and directory structure
    base_name = os.path.splitext(os.path.basename(file_name))[0]
    output_filename = f'myopia_inference_{goal_name}_{base_name}.png'
    
    # Create subdirectories based on metadata
    output_dir = os.path.join('myopia_inference_plots', 
                              f'goal_{goal_name}',
                              f'bins_{len(myopia_values)}')
    output_path = os.path.join(output_dir, output_filename)

    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved as {output_path}")

def parameter_sweep(gridworld, goal_utility, grass_shortcutiness,
                    grass_capacity_values=None, population_values=None):
    results = []
    
    # Set default values if none provided
    if grass_capacity_values is None:
        grass_capacity_values = range(10, 201, 10)  # Default grass capacity range
    if population_values is None:
        population_values = range(10, 201, 10)  # Default population range
    
    for grass_cap in grass_capacity_values:
        for pop in population_values:
            # Temporarily modify global variables
            global grass_capacity, population
            grass_capacity, population = grass_cap, pop
    
            utility = universalized_plan_utility(gridworld, goal_utility, grass_shortcutiness, grass_cap, pop)
            results.append({
                'grass_capacity': grass_cap,
                'population': pop,
                'utility': utility
            })
    
    return results

def print_sweep_results(results):
    print("Grass Capacity | Population | Universalized Utility")
    print("-" * 50)
    for result in results:
        print(f"{result['grass_capacity']:14d} | {result['population']:10d} | {result['utility']:20.2f}")

def save_sweep_results_to_csv(results, file_path):
    # Prepare the CSV header
    headers = ['grass_capacity', 'population', 'utility']

    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()

        for result in results:
            writer.writerow(result)

    print(f"Parameter sweep results saved to {file_path}")

def infer_myopia_parameter(trajectory, forward_model_table):
    # Get the first (and only) key from the forward_model_table
    metadata_key = list(forward_model_table.keys())[0]
    
    # Parse the metadata_key as JSON
    metadata = json.loads(metadata_key)
    gridworld = metadata['gridworld']
    grass_capacity = metadata.get('grass_capacity', '')
    population = metadata.get('population', '')
    goal_name = metadata.get('goal', '')
    
    # Calculate the actual shortcut ratio from the given trajectory
    actual_shortcut = calculate_shortcutiness(gridworld, trajectory)
    
    # Get the inner dictionary
    inner_dict = forward_model_table[metadata_key]
    
    # Find the closest shortcut value in our discretized space
    shortcut_values = sorted(set(k[1] for k in inner_dict.keys()))
    closest_shortcut = min(shortcut_values, key=lambda x: abs(x - actual_shortcut))
    
    # Calculate likelihoods
    myopia_values = sorted(set(k[0] for k in inner_dict.keys()))
    likelihoods = [inner_dict.get((myopia, closest_shortcut), 0) for myopia in myopia_values]
    
    # Apply Bayes' rule
    # p(myopia | shortcut) ∝ p(shortcut | myopia) * p(myopia)
    # We assume a uniform prior for p(myopia)
    prior = 1.0 / len(myopia_values)
    posterior = [likelihood * prior for likelihood in likelihoods]
    
    # Normalize posterior
    total_posterior = sum(posterior)
    if total_posterior > 0:
        posterior = [p / total_posterior for p in posterior]
    else:
        posterior = [1.0 / len(myopia_values)] * len(myopia_values)
    
    return myopia_values, posterior
    

def generate_forward_model_table(gridworld, trajectory, my_goal_utility, goal_name,
                                 grass_capacity, population, num_bins=100, num_agents=100, num_simulations=10):
    myopia_values = np.linspace(0, 1, num_bins)
    shortcut_values = np.linspace(0, 1, num_bins)
    
    start = trajectory[0]['coordinate']
    end = trajectory[-1]['coordinate']

    rule_following_trajectory = full_rule_following_trajectory(gridworld, start, end)
    rule_following_trajectory_utility = calculate_plan_utility(rule_following_trajectory, my_goal_utility)
    
    results = {}
    
    for myopia in tqdm(myopia_values, desc="Generating table with forward model"):
        for shortcut in shortcut_values:
            # Calculate My_Trajectory_utility
            my_trajectory_utility = (rule_following_trajectory_utility - goal_utility) * (1 - shortcut) + goal_utility
            
            # Run multiple simulations and average the results
            universalized_utilities = [universalized_plan_utility(
                gridworld, my_goal_utility, shortcut, grass_capacity, population
            ) for _ in range(num_simulations)]
        
            avg_universalized_utility = np.mean(universalized_utilities)
            
            # Calculate Overall_trajectory_utility
            overall_utility = myopia * my_trajectory_utility + (1 - myopia) * avg_universalized_utility
            
            results[(myopia, shortcut)] = overall_utility
    
    # Calculate probabilities using softmax
    max_utility = max(results.values()) # Get the maximum utility value
    exp_utilities = {k: np.exp(v - max_utility) for k, v in results.items()} # Exponentiate the utilities; max normalization 
    total_exp_utility = sum(exp_utilities.values()) # Sum the exponentiated utilities   
    probabilities = {k: v / total_exp_utility for k, v in exp_utilities.items()} # Divide each exponential utility by the total exponential utility to get the probabilities
    probabilities = {(float(k[0]), float(k[1])): float(v) for k, v in probabilities.items()}

    # Generate a unique key for this configuration
    metadata_key = json.dumps({
        'num_bins': num_bins,
        'num_agents': num_agents,
        'num_simulations': num_simulations,
        'gridworld': gridworld,
        'my_goal_utility': my_goal_utility,
        'start': trajectory[0]['coordinate'],
        'end': trajectory[-1]['coordinate'],
        'grass_capacity': grass_capacity,
        'population': population,
        'goal': goal_name
    })
    
    # Save the probabilities to CSV
    csv_file_path = f'forward_model_table_{goal}_{grass_capacity}_{population}_{run + 1}.csv'
    save_forward_model_table_to_csv({metadata_key: probabilities}, csv_file_path)

    return {metadata_key: probabilities}

def save_forward_model_table_to_csv(forward_model_table, file_path):
    # Since the forward_model_table has a complex structure,
    # we'll flatten it for the CSV.
    # Each row will have: myopia, shortcutiness, probability

    # Get the first (and only) key from the forward_model_table
    metadata_key = list(forward_model_table.keys())[0]
    probabilities = forward_model_table[metadata_key]

    # Load the metadata from the key
    metadata = json.loads(metadata_key)

    # Prepare the CSV header
    headers = ['myopia', 'shortcutiness', 'probability', 'goal', 'grass_capacity', 'population']

    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        for (myopia, shortcutiness), probability in probabilities.items():
            writer.writerow([
                myopia,
                shortcutiness,
                probability,
                metadata.get('goal', ''),
                metadata.get('grass_capacity', ''),
                metadata.get('population', '')
            ])

    print(f"Forward model table saved to {file_path}")

def save_averaged_posterior_to_csv(myopia_values, averaged_posterior_probabilities, file_path):
    # Prepare the CSV header
    headers = ['myopia', 'averaged_posterior_probability']

    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        for myopia, probability in zip(myopia_values, averaged_posterior_probabilities):
            writer.writerow([myopia, probability])

    print(f"Averaged posterior probabilities saved to {file_path}")

def run_parameter_sweep(goal, grass_capacity, population):
    # Load the gridworld and trajectory
    file_name = 'trajectories/map1_8.json'
    gridworld, trajectory = load_gridworld_and_trajectory(file_name)
    
    # Sample goal utility
    goal_utility = sample_goal_utility(goal)
    
    # Calculate grass shortcutiness
    grass_shortcutiness = calculate_shortcutiness(gridworld, trajectory)
    
    # Perform parameter sweep
    print(f"\nPerforming parameter sweep for goal: {goal}")
    
    sweep_results = parameter_sweep(
        gridworld, goal_utility, grass_shortcutiness,
        grass_capacity_values=range(10, 201, 10),
        population_values=[population]
    )
    
    sweep_csv_file_path = f'parameter_sweep_results_{goal}.csv'
    save_sweep_results_to_csv(sweep_results, sweep_csv_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run parameter sweep or full simulation.')
    parser.add_argument('--run_sweep', action='store_true', help='Run only the parameter sweep.')
    args = parser.parse_args()

    if args.run_sweep:
        # Run only the parameter sweep code
        file_name = 'trajectories/map1_8.json'
        gridworld, trajectory = load_gridworld_and_trajectory(file_name)
        
        # Extract all goals
        all_goals = list(goal_probabilities.keys())
        
        # Set parameters for the sweep
        grass_capacity_values = range(10, 201, 10)  # Sweep over grass capacities
        population_values = [100]  # Fix population at 100
        
        # Loop over each goal
        for goal in all_goals:
            print(f"\nPerforming parameter sweep for goal: {goal}")
            goal_utility = sample_goal_utility(goal)
            grass_shortcutiness = calculate_shortcutiness(gridworld, trajectory)
            
            sweep_results = parameter_sweep(
                gridworld, goal_utility, grass_shortcutiness,
                grass_capacity_values=grass_capacity_values,
                population_values=population_values
            )
            
            sweep_csv_file_path = f'parameter_sweep_results_{goal}.csv'
            save_sweep_results_to_csv(sweep_results, sweep_csv_file_path)
    else:
        # Run the main code as before
        file_name = 'trajectories/map1_8.json'
        gridworld, trajectory = load_gridworld_and_trajectory(file_name)
        
        # Extract all goals
        all_goals = list(goal_probabilities.keys())
        num_runs_per_goal = 3
        
        # Set or retrieve the grass_capacity and population values
        grass_capacity = 100  # Or any value you're using
        population = 100      # Or any value you're fixing it at
        
        # Loop over each goal
        for goal in all_goals:
            print(f"\nProcessing goal: {goal}")
            posterior_probabilities_runs = []
            myopia_values_list = []
            
            # Run the simulation multiple times to reduce noise
            for run in range(num_runs_per_goal):
                print(f"  Run {run + 1}/{num_runs_per_goal}")
                # Sample goal utility for the current goal
                goal_utility = sample_goal_utility(goal)
                
                # Load or create the forward_model_table.json file
                forward_model_table_path = f'forward_model_table_{goal}.json'
                if os.path.exists(forward_model_table_path):
                    forward_model_tables = load_forward_model_table(forward_model_table_path)
                else:
                    forward_model_tables = {}
                
                # Generate a unique key for this configuration
                metadata_key = json.dumps({
                    'num_bins': 100,
                    'num_agents': 100,
                    'num_simulations': 10,
                    'gridworld': gridworld,
                    'my_goal_utility': goal_utility,
                    'start': trajectory[0]['coordinate'],
                    'end': trajectory[-1]['coordinate'],
                    'grass_capacity': grass_capacity,
                    'population': population,
                    'goal': goal
                })
                
                if metadata_key in forward_model_tables:
                    print("    Loading existing forward model table for the current configuration...")
                    forward_model_table = forward_model_tables[metadata_key]
                else:
                    print("    Generating new forward model table for the current configuration...")
                    new_table = generate_forward_model_table(
                        gridworld, trajectory, goal_utility, goal,
                        grass_capacity, population, num_bins=100, num_agents=100, num_simulations=10
                    )
                    
                    # Add the new table to the dictionary
                    forward_model_tables.update(new_table)
                    
                    # Save the updated forward model tables
                    save_forward_model_table(forward_model_tables, forward_model_table_path)
                
                # Save the forward model table to CSV
                csv_file_path = f'forward_model_table_{goal}_{run + 1}.csv'
                save_forward_model_table_to_csv({metadata_key: forward_model_tables[metadata_key]}, csv_file_path)
                
                myopia_values, posterior_probabilities = infer_myopia_parameter(
                    trajectory, {metadata_key: forward_model_tables[metadata_key]}
                )
                posterior_probabilities_runs.append(posterior_probabilities)
                myopia_values_list.append(myopia_values)
            
            # Average the posterior probabilities across runs
            averaged_posterior_probabilities = np.mean(posterior_probabilities_runs, axis=0)
            # Save the averaged results to CSV
            averaged_csv_file_path = f'averaged_posterior_probabilities_{goal}.csv'
            save_averaged_posterior_to_csv(myopia_values_list[0], averaged_posterior_probabilities, averaged_csv_file_path)
            
            # Plot the averaged posterior probabilities
            plot_myopia_inference(
                myopia_values_list[0], averaged_posterior_probabilities, file_name, goal_utility, goal_name=goal
            )
            
            # Perform parameter sweep and save results to CSV
            print(f"\nPerforming parameter sweep for goal: {goal}")
            sweep_results = parameter_sweep(
                gridworld, goal_utility, calculate_shortcutiness(gridworld, trajectory),
                grass_capacity_values=range(10, 201, 10),
                population_values=[population]
            )
            sweep_csv_file_path = f'parameter_sweep_results_{goal}.csv'
            save_sweep_results_to_csv(sweep_results, sweep_csv_file_path)