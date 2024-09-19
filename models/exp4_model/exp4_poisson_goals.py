import numpy as np
import random
from collections import deque
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

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
def universalized_plan_utility(gridworld, my_goal_utility, my_grass_shortcutiness, goal_probabilities, average_goals_per_agent, population):
    total_grass_steps = 0
    total_utility = 0
    total_goal_events = 0

    # Simulate each agent
    for agent_id in range(population):
        # Sample the number of goals for this agent
        num_goals = np.random.poisson(average_goals_per_agent)
        if num_goals == 0:
            continue  # The agent has no goals

        for _ in range(num_goals):
            total_goal_events += 1

            goal = sample_goal(goal_probabilities)
            goal_utility = sample_goal_utility(goal)

            # Only simulating goals which have utilities greater than or equal to own
            if goal_utility < my_goal_utility:
                continue

            starting_location = sample_location(gridworld)
            goal_location = sample_location(gridworld)
            
            # Resample goal_location if it is the same as starting_location
            while goal_location == starting_location:
                goal_location = sample_location(gridworld)
            
            agent_utility = goal_utility

            # Approximate the number of grass steps using the grass shortcutiness parameter
            length_strict_rule_following = len(full_rule_following_trajectory(gridworld, starting_location, goal_location))
            grass_steps = my_grass_shortcutiness * length_strict_rule_following

            total_grass_steps += int(grass_steps)
            total_utility += agent_utility

    # If total grass steps exceed capacity, incur grass community cost
    if total_grass_steps > grass_capacity:
        total_utility -= grass_community_cost

    # Normalize the utility by the total number of goal events
    if total_goal_events > 0:
        return total_utility / total_goal_events
    else:
        return 0

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

# Sample a goal from the possible goals based on their probabilities
def sample_goal(goal_probabilities):
    goals = list(goal_probabilities.keys())
    probabilities = list(goal_probabilities.values())
    return np.random.choice(goals, p=probabilities)

# Sample a goal utility from the goal distribution (mean and standard error)
def sample_goal_utility(goal):
    mean, se = goal_utilities[goal]
    return np.random.normal(mean, se)

def save_forward_model_table(data, file_path):
    serializable_data = {}
    for scenario_name, scenario_data in data.items():
        serializable_scenario_data = {}
        for metadata_key, table in scenario_data.items():
            # Convert the metadata key from a string back to a dictionary
            metadata = json.loads(metadata_key)
            # Convert numpy float64 to regular float for JSON serialization
            serializable_table = {
                str(k): float(v) for k, v in table.items()
            }
            serializable_scenario_data[json.dumps(metadata)] = serializable_table
        serializable_data[scenario_name] = serializable_scenario_data
    
    with open(file_path, 'w') as f:
        json.dump(serializable_data, f)

def load_forward_model_table(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    loaded_data = {}
    for scenario_name, scenario_data in data.items():
        loaded_scenario_data = {
            metadata_key: {
                tuple(map(float, k.strip('()').split(','))): v for k, v in table.items()
            }
            for metadata_key, table in scenario_data.items()
        }
        loaded_data[scenario_name] = loaded_scenario_data
    return loaded_data

def load_gridworld_and_trajectory(file_name):
    with open(file_name, 'r') as file:
        data = json.load(file)
    return data['gridworld'], data['trajectory']

def plot_myopia_inference(myopia_values, posterior_probabilities, file_name, goal_utility, scenario_name, grass_capacity):
    plt.figure(figsize=(10, 6))
    plt.plot(myopia_values, posterior_probabilities)
    plt.xlabel('Myopia Parameter')
    plt.ylabel('Posterior Probability')
    plt.title(f'Inferred Posterior Distribution of Myopia Parameter\nScenario: {scenario_name}')
    
    # Set y-axis limits
    plt.ylim(0, 1)
    
    # Add text box with additional information
    info_text = f"File: {file_name}\nBins: {len(myopia_values)}\nGoal Utility: {goal_utility:.2f}\nGrass Capacity: {grass_capacity}"
    plt.text(0.95, 0.95, info_text, transform=plt.gca().transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Define the output filename and directory structure
    base_name = os.path.splitext(os.path.basename(file_name))[0]
    output_filename = f'myopia_inference_{base_name}_{scenario_name}.png'
    
    # Create subdirectories based on metadata
    output_dir = os.path.join('myopia_inference_plots', 
                              f'goal_utility_{goal_utility:.2f}',
                              f'grass_capacity_{grass_capacity}',
                              f'bins_{len(myopia_values)}',
                              scenario_name)
    output_path = os.path.join(output_dir, output_filename)

    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved as {output_path}")

def infer_myopia_parameter(trajectory, forward_model_table):
    # Get the first (and only) key from the forward_model_table
    metadata_key = list(forward_model_table.keys())[0]
    
    # Parse the metadata_key as JSON
    metadata = json.loads(metadata_key)
    gridworld = metadata['gridworld']
    
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

def calculate_average_goals_per_agent(goal_proportions, population):
    total_goals = sum(goal_proportions.values())
    average_goals_per_agent = total_goals / population
    return average_goals_per_agent

def generate_forward_model_table(
    gridworld, 
    trajectory, 
    my_goal_utility, 
    goal_proportions, 
    scenario_name, 
    num_bins=100, 
    num_agents=100, 
    num_simulations=10, 
    grass_capacity=200
):
    # Calculate the average number of goals per agent
    average_goals_per_agent = calculate_average_goals_per_agent(goal_proportions, num_agents)

    # Normalize the proportions so they sum to 1
    total_proportion = sum(goal_proportions.values())
    goal_probabilities = {goal: proportion / total_proportion for goal, proportion in goal_proportions.items()}

    myopia_values = np.linspace(0, 1, num_bins)
    shortcut_values = np.linspace(0, 1, num_bins)
    
    start = trajectory[0]['coordinate']
    end = trajectory[-1]['coordinate']

    rule_following_trajectory = full_rule_following_trajectory(gridworld, start, end)
    rule_following_trajectory_utility = calculate_plan_utility(rule_following_trajectory, my_goal_utility)
    
    results = {}
    
    for myopia in tqdm(myopia_values, desc=f"Generating table with forward model for {scenario_name}"):
        for shortcut in shortcut_values:
            # Calculate My_Trajectory_utility
            my_trajectory_utility = (rule_following_trajectory_utility - my_goal_utility) * (1 - shortcut) + my_goal_utility
            
            # Run multiple simulations and average the results
            universalized_utilities = [
                universalized_plan_utility(
                    gridworld, 
                    my_goal_utility, 
                    shortcut, 
                    goal_probabilities,
                    average_goals_per_agent,
                    num_agents  # Pass num_agents as population
                ) for _ in range(num_simulations)
            ]
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
        'goal_proportions': goal_proportions,  # Store raw, unnormalized values
        'grass_capacity': grass_capacity
    })
    
    return {metadata_key: probabilities}

if __name__ == "__main__":
    # List of file names to process
    file_names = [
        'trajectories/map1_8.json',
        'trajectories/map1_4.json',
        'trajectories/map1_1.json'
        # Add more file names as needed
    ]

    num_goal_utility_simulations = 5  # Number of times to sample goal utility and then average over
    grass_capacity = 200  
    grass_community_cost = 10000
    num_bins = 50
    num_simulations = 4  # This refers to the number of simulations in the universalization step
    num_agents = 100
    population = num_agents  # Ensure consistency

    # Define the goals and their utilities (mean and SE from the experiment data)
    goal_utilities = {
        'a friend': (52.7, 3.87),
        'pain': (82.6, 3.87),
        'ice cream': (46.0, 3.87),
        'vac clinic': (48.0, 3.87),
        'porta-potty': (65.8, 3.87),
        'police car': (63.1, 3.87)
    }
    
    # Define the list of goal proportions with scenario names
    goal_proportions_scenarios = [
        ('Condition_0', {
            'a friend': 44,
            'pain': 35,
            'ice cream': 44,
            'vac clinic': 23,
            'porta-potty': 34,
            'police car': 9
        }),
        ('Condition_1', {
            'a friend': 44,
            'pain': 60,
            'ice cream': 44,
            'vac clinic': 23,
            'porta-potty': 50,
            'police car': 9
        }),
        ('Condition_2', {
            'a friend': 44,
            'pain': 0,
            'ice cream': 44,
            'vac clinic': 23,
            'porta-potty': 1,
            'police car': 9
        }),
        ('Condition_3', {
            'a friend': 60,
            'pain': 0,
            'ice cream': 70,
            'vac clinic': 23,
            'porta-potty': 1,
            'police car': 9
        }),
        ('Condition_4', {
            'a friend': 4,
            'pain': 0,
            'ice cream': 96,
            'vac clinic': 0,
            'porta-potty': 0,
            'police car': 2
        })
    ]
        
    # Load or create the forward_model_table.json file
    forward_model_table_path = 'forward_model_table.json'
    if os.path.exists(forward_model_table_path):
        forward_model_tables = load_forward_model_table(forward_model_table_path)
    else:
        forward_model_tables = {}

    for file_name in file_names:
        print(f"\nProcessing file: {file_name}")
        gridworld, trajectory = load_gridworld_and_trajectory(file_name)

        for scenario_name, goal_proportions in goal_proportions_scenarios:
            print(f"\nRunning model for {scenario_name}")

            # Calculate average goals per agent
            average_goals_per_agent = calculate_average_goals_per_agent(goal_proportions, population)

            # Initialize lists to store results for each goal utility simulation
            all_myopia_values = []
            all_posterior_probabilities = []
            sampled_goal_utilities = []

            for _ in range(num_goal_utility_simulations):
                # Sample goal utility for the 'porta-potty' goal
                goal_utility = sample_goal_utility('porta-potty')
                sampled_goal_utilities.append(goal_utility)

                # Initialize scenario in forward_model_tables if not present
                if scenario_name not in forward_model_tables:
                    forward_model_tables[scenario_name] = {}
                
                scenario_tables = forward_model_tables[scenario_name]

                # Generate a unique key for this configuration
                metadata_key = json.dumps({
                    'num_bins': num_bins,
                    'num_agents': num_agents,
                    'num_simulations': num_simulations,
                    'gridworld': gridworld,
                    'my_goal_utility': goal_utility,
                    'start': trajectory[0]['coordinate'],
                    'end': trajectory[-1]['coordinate'],
                    'goal_proportions': goal_proportions,
                    'grass_capacity': grass_capacity
                })

                if metadata_key in scenario_tables:
                    print("Loading existing forward model table for the current configuration...")
                    forward_model_table = scenario_tables[metadata_key]
                else:
                    print("Generating new forward model table for the current configuration...")
                    new_table = generate_forward_model_table(
                        gridworld, 
                        trajectory, 
                        goal_utility, 
                        goal_proportions, 
                        scenario_name, 
                        num_bins=num_bins, 
                        num_agents=num_agents, 
                        num_simulations=num_simulations, 
                        grass_capacity=grass_capacity
                    )
                    
                    # Add the new table to the scenario's dictionary
                    scenario_tables.update(new_table)
                    
                    # Save the updated forward model tables
                    save_forward_model_table(forward_model_tables, forward_model_table_path)

                # Retrieve the forward model table for the current configuration
                forward_model_table = scenario_tables[metadata_key]
                
                myopia_values, posterior_probabilities = infer_myopia_parameter(
                    trajectory, 
                    {metadata_key: forward_model_table}
                )
                all_myopia_values.append(myopia_values)
                all_posterior_probabilities.append(posterior_probabilities)

            # Calculate average results
            avg_myopia_values = np.mean(all_myopia_values, axis=0)
            avg_posterior_probabilities = np.mean(all_posterior_probabilities, axis=0)
            avg_goal_utility = np.mean(sampled_goal_utilities)

            # Plot and save the averaged results
            plot_myopia_inference(
                avg_myopia_values, 
                avg_posterior_probabilities, 
                file_name, 
                avg_goal_utility, 
                scenario_name, 
                grass_capacity
            )