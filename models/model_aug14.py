import numpy as np
import random
from functools import lru_cache
import json

def save_trajectories_with_simulation_data(agents_data, gridworld, simulation_parameters, file_name="trajectories_policies.jsonl"):
    # Ensure gridworld is serializable
    gridworld_serializable = [[str(cell) for cell in row] for row in gridworld]
    
    # Convert numpy objects and booleans in simulation_parameters to JSON-compatible formats
    simulation_parameters_serializable = {}
    for k, v in simulation_parameters.items():
        if isinstance(v, np.ndarray):
            simulation_parameters_serializable[k] = v.tolist()  # Convert NumPy arrays to lists
        elif isinstance(v, bool):
            simulation_parameters_serializable[k] = int(v)  # Convert booleans to integers
        else:
            simulation_parameters_serializable[k] = v  # Keep everything else as is
    
    # Convert booleans to integers in agent-specific data
    for agent in agents_data:
        if 'is_rule_following' in agent:
            agent['is_rule_following'] = int(agent['is_rule_following'])  # Convert boolean to integer

    # Create the data structure to store simulation details and agents data
    data = {
        "gridworld": gridworld_serializable,
        "simulation_parameters": simulation_parameters_serializable,
        "agents": agents_data  # A list of agents and their unique data
    }
    
    # Write the entire simulation data to the JSON file in one line
    with open(file_name, "a") as f: 
        json.dump(data, f)  # Write the JSON object as a single line without indentation
        f.write("\n")  # Add a newline to separate different simulation runs


# Actual simulation code begins here
# Define the gridworld map with only 'S' (sidewalk) and 'G' (grass)
gridworld = [
    ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S'],
    ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S']
]

# Possible directions the agent can move, including diagonal directions
directions = ['west', 'east', 'north', 'south', 'north-west', 'north-east', 'south-west', 'south-east']

# Utility distribution parameters for the goal
GOAL_MEAN_UTILITY = 100
GOAL_UTILITY_VARIANCE = 10

# Utility functions
def goal_utility():
    return np.random.normal(GOAL_MEAN_UTILITY, GOAL_UTILITY_VARIANCE)

def motion_utility(location_type, motion_type):
    if location_type in ['G', 'S']:
        if motion_type == 'is_walking':
            return -1
        if motion_type == 'is_walking_diagonal':
            return -np.sqrt(2)
        if motion_type == 'is_staying':
            return 0
    return 0

def rule_follower_motion_utility(location_type, motion_type):
    if location_type == 'G':
        if motion_type == 'is_staying':
            return 0
        else:
            return -1000
    if location_type == 'S':
        if motion_type == 'is_walking':
            return -1
        if motion_type == 'is_walking_diagonal':
            return -np.sqrt(2)
        if motion_type == 'is_staying':
            return 0
    return 0

def location_utility(state_x, state_y, goal_x, goal_y):
    if (state_x, state_y) == (goal_x, goal_y):
        return goal_utility()
    else:
        return 0

def get_gridworld_at(gridworld, x, y):
    return gridworld[y][x]

def set_gridworld_at(gridworld, x, y, value):
    gridworld[y][x] = value

def x_increment(direction):
    if direction == 'west':
        return -1
    if direction == 'east':
        return 1
    if direction in ['north', 'south']:
        return 0
    if direction in ['north-west', 'south-west']:
        return -1
    if direction in ['north-east', 'south-east']:
        return 1
    if direction == 'stay':
        return 0

def y_increment(direction):
    if direction == 'north':
        return -1
    if direction == 'south':
        return 1
    if direction in ['west', 'east']:
        return 0
    if direction in ['north-west', 'north-east']:
        return -1
    if direction in ['south-west', 'south-east']:
        return 1
    if direction == 'stay':
        return 0

def gridworld_max_x(gridworld):
    return len(gridworld[0])

def gridworld_max_y(gridworld):
    return len(gridworld)

def gridworld_transition(gridworld, current_x, current_y, action):
    next_x = current_x + x_increment(action)
    if next_x >= gridworld_max_x(gridworld) or next_x < 0:
        next_x = current_x
    next_y = current_y + y_increment(action)
    if next_y >= gridworld_max_y(gridworld) or next_y < 0:
        next_y = current_y
    return {'location': get_gridworld_at(gridworld, next_x, next_y), 'x': next_x, 'y': next_y}

def utility_function(gridworld, state_x, state_y, action, goal_x, goal_y, rule_following=False):
    location_type = get_gridworld_at(gridworld, state_x, state_y)
    state_location_utility = location_utility(state_x, state_y, goal_x, goal_y)
    motion_type = 'is_walking_diagonal' if action in ['north-west', 'north-east', 'south-west', 'south-east'] else 'is_walking'
    if rule_following:
        state_motion_utility = rule_follower_motion_utility(location_type, motion_type)
    else:
        state_motion_utility = motion_utility(location_type, motion_type)
    return state_location_utility + state_motion_utility

# Value iteration functions
@lru_cache(maxsize=None)
def value_function(curr_iteration, state_x, state_y, goal_x, goal_y, max_iterations, initial_x, initial_y, rule_following=False):
    if curr_iteration == -1:
        return 0
    prev_optimal_action_value = optimal_action_value(curr_iteration - 1, state_x, state_y, goal_x, goal_y, max_iterations, initial_x, initial_y, rule_following)
    return prev_optimal_action_value['value']

@lru_cache(maxsize=None)
def available_actions_to_values(curr_iteration, state_x, state_y, goal_x, goal_y, max_iterations, initial_x, initial_y, rule_following=False):
    action_values = []
    for action in directions:
        utility = utility_function(gridworld, state_x, state_y, action, goal_x, goal_y, rule_following)
        next_state = gridworld_transition(gridworld, state_x, state_y, action)
        next_state_value = value_function(curr_iteration, next_state['x'], next_state['y'], goal_x, goal_y, max_iterations, initial_x, initial_y, rule_following)
        action_values.append({'action': action, 'value': utility + next_state_value})
    return action_values

@lru_cache(maxsize=None)
def optimal_action_value(curr_iteration, state_x, state_y, goal_x, goal_y, max_iterations, initial_x, initial_y, rule_following=False):
    actions_to_values = available_actions_to_values(curr_iteration, state_x, state_y, goal_x, goal_y, max_iterations, initial_x, initial_y, rule_following)
    return max(actions_to_values, key=lambda a: a['value'])

@lru_cache(maxsize=None)
def should_terminate(state_x, state_y, goal_x, goal_y, max_iterations, initial_x, initial_y):
    if value_function(max_iterations, initial_x, initial_y, goal_x, goal_y, max_iterations, initial_x, initial_y) <= 0:
        return True
    return (state_x, state_y) == (goal_x, goal_y)

def optimal_policy_from_initial_state(state_x, state_y, curr_iteration, max_iterations, goal_x, goal_y, initial_x, initial_y, rule_following=False):
    if should_terminate(state_x, state_y, goal_x, goal_y, max_iterations, initial_x, initial_y):
        return []
    curr_optimal_action_value = optimal_action_value(curr_iteration, state_x, state_y, goal_x, goal_y, max_iterations, initial_x, initial_y, rule_following)
    curr_optimal_action = curr_optimal_action_value['action']
    next_state = gridworld_transition(gridworld, state_x, state_y, curr_optimal_action)
    remaining_policy = optimal_policy_from_initial_state(next_state['x'], next_state['y'], curr_iteration + 1, max_iterations, goal_x, goal_y, initial_x, initial_y, rule_following)
    return [curr_optimal_action] + remaining_policy

def trajectory_from_initial_state(state_x, state_y, curr_iteration, max_iterations, goal_x, goal_y, initial_x, initial_y, rule_following=False):
    if should_terminate(state_x, state_y, goal_x, goal_y, max_iterations, initial_x, initial_y):
        return []
    curr_optimal_action_value = optimal_action_value(curr_iteration, state_x, state_y, goal_x, goal_y, max_iterations, initial_x, initial_y, rule_following)
    curr_optimal_action = curr_optimal_action_value['action']
    next_state = gridworld_transition(gridworld, state_x, state_y, curr_optimal_action)
    remaining_trajectory = trajectory_from_initial_state(next_state['x'], next_state['y'], curr_iteration + 1, max_iterations, goal_x, goal_y, initial_x, initial_y, rule_following)
    return [next_state['location']] + remaining_trajectory

def optimal_policy(initial_state_x, initial_state_y, max_iterations, goal_x, goal_y, rule_following=False):
    return [('start', 'start')] + optimal_policy_from_initial_state(initial_state_x, initial_state_y, 0, max_iterations, goal_x, goal_y, initial_state_x, initial_state_y, rule_following)

def optimal_trajectory(initial_state_x, initial_state_y, max_iterations, goal_x, goal_y, rule_following=False):
    return [get_gridworld_at(gridworld, initial_state_x, initial_state_y)] + trajectory_from_initial_state(initial_state_x, initial_state_y, 0, max_iterations, goal_x, goal_y, initial_state_x, initial_state_y, rule_following)

def optimal_policy_with_trajectory(initial_state_x, initial_state_y, max_iterations, goal_x, goal_y, rule_following=False):
    policy = optimal_policy(initial_state_x, initial_state_y, max_iterations, goal_x, goal_y, rule_following)
    trajectory = optimal_trajectory(initial_state_x, initial_state_y, max_iterations, goal_x, goal_y, rule_following)
    return list(zip(policy, trajectory))


# Pre-simulation functions
def softmax(x):
    e_x = np.exp(x - np.max(x))  
    return e_x / e_x.sum()

def pre_simulation(rule_following, max_grass_capacity, grass_dying_cost, main_agent_utility, initial_x, initial_y, goal_x, goal_y, max_iterations):
    total_utility = 0
    total_grass_traffic = 0
    simulated_agents = 0
    
    num_agents = 100
    agent_ids = range(num_agents)

    actual_simulated_agents = 0  # Track actual simulated agents

    for agent_id in agent_ids:
        if simulated_agents >= 100:
            break  # Stop once 100 agents have been simulated

        # Sample goal utility for pre-simulated agent
        sampled_goal_utility = goal_utility()

        # Increment the count of simulated agents before checking the utility threshold
        simulated_agents += 1

        # Skip agents with utility less than the main agent's utility
        if sampled_goal_utility < main_agent_utility:
            continue

        actual_simulated_agents += 1  # Increment actual simulated agents

        policy_trajectory = optimal_policy_with_trajectory(initial_x, initial_y, max_iterations, goal_x, goal_y, rule_following)
        
        utility = 0
        agent_grass_traffic = 0
        
        for step in policy_trajectory:
            action = step[0]
            location = step[1]
            if location == 'G':
                total_grass_traffic += 1
                agent_grass_traffic += 1
            
            motion_type = 'is_walking_diagonal' if action in ['north-west', 'north-east', 'south-west', 'south-east'] else 'is_walking'
            utility += motion_utility(location, motion_type)
        
        final_location = policy_trajectory[-1][1]
        if final_location == 'F':
            utility += sampled_goal_utility
        
        total_utility += utility
    
    if total_grass_traffic > max_grass_capacity:
        total_utility -= grass_dying_cost
    
    return total_utility, total_grass_traffic, actual_simulated_agents  # Return actual simulated agents

# Simulation function
def simulate(
    initial_x=0,
    initial_y=0,
    goal_x=9,
    goal_y=9,
    MAX_ITERATIONS=10,
    num_agents=100,
    max_grass_capacity=300,
    grass_dying_cost=10000,
    seed=None
):
    # Set the random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        print(f"Random seed set to: {seed}")

    # Set the goal location in the gridworld
    set_gridworld_at(gridworld, goal_x, goal_y, 'F')

    total_grass_traffic = 0
    total_utility = 0

    print(f"Number of agents: {num_agents}")
    print(f"Starting location of each agent: ({initial_x}, {initial_y})")
    print(f"Goal location: ({goal_x}, {goal_y})")
    agent_ids = range(num_agents)

    simulated_agents_per_main_agent = []
    agents_data = []  # Collect agents data for final JSON output

    # Separate totals for main and simulated agents
    rule_following_total_utility_main = 0
    rule_exceptional_total_utility_main = 0
    rule_following_total_utility_simulated = 0
    rule_exceptional_total_utility_simulated = 0

    rule_following_agents = 0
    rule_exceptional_agents = 0

    for agent_id in agent_ids:
        # Sample goal utility once for each agent
        main_agent_utility = goal_utility()

        # Pre-simulations for the main agent using the same sampled goal utility
        rule_exceptional_utility_simulated, _, actual_simulated_agents_exceptional = pre_simulation(
            rule_following=False,
            max_grass_capacity=max_grass_capacity,
            grass_dying_cost=grass_dying_cost,
            main_agent_utility=main_agent_utility,
            initial_x=initial_x,
            initial_y=initial_y,
            goal_x=goal_x,
            goal_y=goal_y,
            max_iterations=MAX_ITERATIONS
        )
        rule_following_utility_simulated, _, actual_simulated_agents_following = pre_simulation(
            rule_following=True,
            max_grass_capacity=max_grass_capacity,
            grass_dying_cost=grass_dying_cost,
            main_agent_utility=main_agent_utility,
            initial_x=initial_x,
            initial_y=initial_y,
            goal_x=goal_x,
            goal_y=goal_y,
            max_iterations=MAX_ITERATIONS
        )

        # Ensure the same number of simulated agents for both rule-following and rule-exceptional
        actual_simulated_agents = max(actual_simulated_agents_exceptional, actual_simulated_agents_following)
        simulated_agents_per_main_agent.append(actual_simulated_agents)

        # Use softmax to calculate the probabilities
        utilities = np.array([rule_following_utility_simulated, rule_exceptional_utility_simulated])
        probabilities = softmax(utilities)

        # The probability that the main agent follows the rule-following strategy
        rule_following_probability = probabilities[0]
        is_rule_following = np.random.random() < rule_following_probability

        policy_trajectory = optimal_policy_with_trajectory(
            initial_x,
            initial_y,
            MAX_ITERATIONS,
            goal_x,
            goal_y,
            is_rule_following
        )

        # Collect agent-specific data
        agent_data = {
            "agent_id": agent_id,
            "policy_trajectory": policy_trajectory,
            "is_rule_following": is_rule_following,
            "main_agent_utility": main_agent_utility
        }
        agents_data.append(agent_data)

        utility_main = 0
        agent_grass_traffic = 0

        for step in policy_trajectory:
            action = step[0]
            location = step[1]
            if location == 'G':
                total_grass_traffic += 1
                agent_grass_traffic += 1

            motion_type = 'is_walking_diagonal' if action in ['north-west', 'north-east', 'south-west', 'south-east'] else 'is_walking'
            utility_main += motion_utility(location, motion_type)

        final_location = policy_trajectory[-1][1]
        if final_location == 'F':
            utility_main += main_agent_utility

        total_utility += utility_main

        if is_rule_following:
            rule_following_agents += 1
            rule_following_total_utility_main += utility_main
            rule_following_total_utility_simulated += rule_following_utility_simulated
        else:
            rule_exceptional_agents += 1
            rule_exceptional_total_utility_main += utility_main
            rule_exceptional_total_utility_simulated += rule_exceptional_utility_simulated

        print(f"Agent {agent_id}: Final utility {utility_main}, final location {final_location}, grass traffic added {agent_grass_traffic}, simulated {actual_simulated_agents} agents")
    

    if total_grass_traffic > max_grass_capacity:
        total_utility -= grass_dying_cost
        print(f"Grass traffic exceeded max capacity. Applying penalty of {grass_dying_cost} to total utility.")

    # Create simulation parameters dictionary
    simulation_parameters = {
        "initial_x": initial_x,
        "initial_y": initial_y,
        "goal_x": goal_x,
        "goal_y": goal_y,
        "max_iterations": MAX_ITERATIONS,
        "num_agents": num_agents,
        "max_grass_capacity": max_grass_capacity,
        "grass_dying_cost": grass_dying_cost,
        "seed": seed,
    }
    # Save all collected data to JSON
    save_trajectories_with_simulation_data(agents_data, gridworld, simulation_parameters)

    rule_following_avg_utility_main = rule_following_total_utility_main / rule_following_agents if rule_following_agents > 0 else 0
    rule_exceptional_avg_utility_main = rule_exceptional_total_utility_main / rule_exceptional_agents if rule_exceptional_agents > 0 else 0
    rule_following_avg_utility_simulated = rule_following_total_utility_simulated / sum(simulated_agents_per_main_agent) if sum(simulated_agents_per_main_agent) > 0 else 0
    rule_exceptional_avg_utility_simulated = rule_exceptional_total_utility_simulated / sum(simulated_agents_per_main_agent) if sum(simulated_agents_per_main_agent) > 0 else 0

    with open("simplified_simulation_results.txt", "a") as file:
        file.write(f"Random Seed: {seed}\n" if seed is not None else "Random Seed: None\n")
        file.write(f"Max Iterations: {MAX_ITERATIONS}\n")
        file.write("Simulation Results:\n")
        file.write(f"Number of agents: {num_agents}\n")
        file.write(f"Starting location of each agent: ({initial_x}, {initial_y})\n")
        file.write(f"Goal location: ({goal_x}, {goal_y})\n")
        file.write(f"Total Grass Traffic: {total_grass_traffic}\n")
        file.write(f"Max Grass Capacity: {max_grass_capacity}\n")
        file.write(f"Grass Dying Cost: {grass_dying_cost}\n")
        file.write(f"Total Utility: {total_utility}\n")
        file.write(f"Average Utility per Agent: {total_utility / num_agents}\n")
        file.write(f"Number of Rule-Following Agents: {rule_following_agents}\n")
        file.write(f"Rule-Following Total Utility (Main Agents): {rule_following_total_utility_main}\n")
        file.write(f"Rule-Following Average Utility per Main Agent: {rule_following_avg_utility_main}\n")
        file.write(f"Rule-Following Total Utility (Simulated Agents): {rule_following_total_utility_simulated}\n")
        file.write(f"Rule-Following Average Utility per Simulated Agent: {rule_following_avg_utility_simulated}\n")
        file.write(f"Number of Rule-Exceptional Agents: {rule_exceptional_agents}\n")
        file.write(f"Rule-Exceptional Total Utility (Main Agents): {rule_exceptional_total_utility_main}\n")
        file.write(f"Rule-Exceptional Average Utility per Main Agent: {rule_exceptional_avg_utility_main}\n")
        file.write(f"Rule-Exceptional Total Utility (Simulated Agents): {rule_exceptional_total_utility_simulated}\n")
        file.write(f"Rule-Exceptional Average Utility per Simulated Agent: {rule_exceptional_avg_utility_simulated}\n")
        file.write(f"Simulated Agents per Main Agent: {simulated_agents_per_main_agent}\n")
        file.write("-" * 40 + "\n\n")

    return {
        'totalGrassTraffic': total_grass_traffic,
        'totalUtility': total_utility,
        'averageUtilityPerAgent': total_utility / num_agents,
        'ruleFollowingAgents': rule_following_agents,
        'ruleExceptionalAgents': rule_exceptional_agents,
        'simulatedAgentsPerMainAgent': simulated_agents_per_main_agent,
        'ruleFollowingUtilityMain': rule_following_total_utility_main,
        'ruleFollowingUtilitySimulated': rule_following_total_utility_simulated,
        'ruleExceptionalUtilityMain': rule_exceptional_total_utility_main,
        'ruleExceptionalUtilitySimulated': rule_exceptional_total_utility_simulated
    }

# Run the simulation, set values 
if __name__ == "__main__":
    simulation_result = simulate(
        initial_x=0,
        initial_y=0,
        goal_x=9,
        goal_y=9,
        MAX_ITERATIONS=10,
        num_agents=100,
        max_grass_capacity=300,
        grass_dying_cost=10000,
        seed=42  
    )
    print("Simulation result:", simulation_result)
