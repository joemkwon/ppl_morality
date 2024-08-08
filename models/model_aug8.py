import numpy as np
from functools import lru_cache

# Define the gridworld map with various locations
# gridworld = (
#     ('S', 'S', 'S', 'hospital'),
#     ('S', 'G', 'G', 'S'),
#     ('S', 'S', 'S', 'coffeeShop'),
#     ('S', 'G', 'G', 'S'),
#     ('S', 'S', 'S', 'busStop')
# )

gridworld = (
  ('S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'hospital'),
  ('S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'),
  ('S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'),
  ('S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'),
  ('S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'),
  ('S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'coffeeShop'),
  ('S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'),
  ('S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'),
  ('S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'),
  ('S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'busStop')
)

# Define the list of goals in the gridworld
goals = ['hospital', 'busStop', 'coffeeShop']

# Possible directions the agent can move, including diagonal directions
directions = ['west', 'east', 'north', 'south', 'north-west', 'north-east', 'south-west', 'south-east']

# Define utility constants for different goals
HOSPITAL_MEAN_UTILITY = 1000
HOSPITAL_UTILITY_VARIANCE = 100

BUSSTOP_MEAN_UTILITY = 100
BUSSTOP_UTILITY_VARIANCE = 10

COFFEESHOP_MEAN_UTILITY = 10
COFFEESHOP_UTILITY_VARIANCE = 1

# Utility functions
def goal_utility(agent_id, goal_type):
    if goal_type == 'hospital':
        return np.random.normal(HOSPITAL_MEAN_UTILITY, HOSPITAL_UTILITY_VARIANCE)
    elif goal_type == 'busStop':
        return np.random.normal(BUSSTOP_MEAN_UTILITY, BUSSTOP_UTILITY_VARIANCE)
    elif goal_type == 'coffeeShop':
        return np.random.normal(COFFEESHOP_MEAN_UTILITY, COFFEESHOP_UTILITY_VARIANCE)
    else:
        return 0

def motion_utility(agent_id, location_type, motion_type):
    if location_type == 'G':
        if motion_type == 'is_walking':
            return -1
        if motion_type == 'is_walking_diagonal':
            return -np.sqrt(2)
        if motion_type == 'is_staying':
            return 0
    if location_type == 'S':
        if motion_type == 'is_walking':
            return -1
        if motion_type == 'is_walking_diagonal':
            return -np.sqrt(2)
        if motion_type == 'is_staying':
            return 0
    return 0

def location_utility(agent_id, location_type, goal):
    if location_type == goal:
        return goal_utility(agent_id, goal)
    else:
        return 0

def get_gridworld_at(gridworld, x, y):
    return gridworld[y][x]

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

def utility_function(agent_id, gridworld, state_x, state_y, action, goal):
    location_type = get_gridworld_at(gridworld, state_x, state_y)
    state_location_utility = location_utility(agent_id, location_type, goal)
    motion_type = 'is_walking_diagonal' if action in ['north-west', 'north-east', 'south-west', 'south-east'] else 'is_walking'
    state_motion_utility = motion_utility(agent_id, location_type, motion_type)
    return state_location_utility + state_motion_utility

@lru_cache(maxsize=None)
def value_function(agent_id, curr_iteration, state_x, state_y, goal):
    if curr_iteration == -1:
        return 0
    prev_optimal_action_value = optimal_action_value(agent_id, curr_iteration - 1, state_x, state_y, goal)
    return prev_optimal_action_value['value']

@lru_cache(maxsize=None)
def available_actions_to_values(agent_id, curr_iteration, state_x, state_y, goal):
    action_values = []
    for action in directions:
        utility = utility_function(agent_id, gridworld, state_x, state_y, action, goal)
        next_state = gridworld_transition(gridworld, state_x, state_y, action)
        next_state_value = value_function(agent_id, curr_iteration, next_state['x'], next_state['y'], goal)
        action_values.append({'action': action, 'value': utility + next_state_value})
    return action_values

@lru_cache(maxsize=None)
def optimal_action_value(agent_id, curr_iteration, state_x, state_y, goal):
    actions_to_values = available_actions_to_values(agent_id, curr_iteration, state_x, state_y, goal)
    return max(actions_to_values, key=lambda a: a['value'])

@lru_cache(maxsize=None)
def should_terminate(agent_id, state_x, state_y, goal):
    if value_function(agent_id, MAX_ITERATIONS, initial_x, initial_y, goal) <= 0:
        return True
    location_type = get_gridworld_at(gridworld, state_x, state_y)
    state_location_utility = location_utility(agent_id, location_type, goal)
    return state_location_utility > 0

def optimal_policy_from_initial_state(agent_id, state_x, state_y, curr_iteration, max_iterations, goal):
    if should_terminate(agent_id, state_x, state_y, goal):
        return []
    curr_optimal_action_value = optimal_action_value(agent_id, curr_iteration, state_x, state_y, goal)
    curr_optimal_action = curr_optimal_action_value['action']
    next_state = gridworld_transition(gridworld, state_x, state_y, curr_optimal_action)
    remaining_policy = optimal_policy_from_initial_state(agent_id, next_state['x'], next_state['y'], curr_iteration + 1, max_iterations, goal)
    return [curr_optimal_action] + remaining_policy

def trajectory_from_initial_state(agent_id, state_x, state_y, curr_iteration, max_iterations, goal):
    if should_terminate(agent_id, state_x, state_y, goal):
        return []
    curr_optimal_action_value = optimal_action_value(agent_id, curr_iteration, state_x, state_y, goal)
    curr_optimal_action = curr_optimal_action_value['action']
    next_state = gridworld_transition(gridworld, state_x, state_y, curr_optimal_action)
    remaining_trajectory = trajectory_from_initial_state(agent_id, next_state['x'], next_state['y'], curr_iteration + 1, max_iterations, goal)
    return [next_state['location']] + remaining_trajectory

def optimal_policy(agent_id, initial_state_x, initial_state_y, max_iterations, goal):
    return [('start', 'start')] + optimal_policy_from_initial_state(agent_id, initial_state_x, initial_state_y, 0, max_iterations, goal)

def optimal_trajectory(agent_id, initial_state_x, initial_state_y, max_iterations, goal):
    return [get_gridworld_at(gridworld, initial_state_x, initial_state_y)] + trajectory_from_initial_state(agent_id, initial_state_x, initial_state_y, 0, max_iterations, goal)

def optimal_policy_with_trajectory(agent_id, initial_state_x, initial_state_y, max_iterations, goal):
    policy = optimal_policy(agent_id, initial_state_x, initial_state_y, max_iterations, goal)
    trajectory = optimal_trajectory(agent_id, initial_state_x, initial_state_y, max_iterations, goal)
    return list(zip(policy, trajectory))

def simulate():
    total_grass_traffic = 0
    total_utility = 0
    
    num_agents = 100
    print(f"Number of agents: {num_agents}")
    agent_ids = range(num_agents)
  
    goal_distribution = ['hospital', 'busStop', 'coffeeShop']
    goal_probabilities = [0.05, 0.45, 0.5]

    goal_counters = {goal: 0 for goal in goals}
    success_counters = {goal: 0 for goal in goals}
    grass_traffic_counters = {goal: 0 for goal in goals}
    
    for agent_id in agent_ids:
        goal = np.random.choice(goal_distribution, p=goal_probabilities)
        goal_counters[goal] += 1
        initial_utility = goal_utility(agent_id, goal)

        print(f"Agent {agent_id}: Goal is {goal} with initial utility {initial_utility}")
        
        policy_trajectory = optimal_policy_with_trajectory(agent_id, initial_x, initial_y, MAX_ITERATIONS, goal)
        
        utility = 0
        agent_grass_traffic = 0
        
        for step in policy_trajectory:
            action = step[0]
            location = step[1]
            if location == 'G':
                total_grass_traffic += 1
                agent_grass_traffic += 1
                grass_traffic_counters[goal] += 1
            
            motion_type = 'is_walking_diagonal' if action in ['north-west', 'north-east', 'south-west', 'south-east'] else 'is_walking'
            utility += motion_utility(agent_id, location, motion_type)
        
        final_location = policy_trajectory[-1][1]
        if final_location == goal:
            utility += initial_utility
            success_counters[goal] += 1
        
        total_utility += utility
        print(f"Agent {agent_id}: Final utility {utility}, final location {final_location}, grass traffic added {agent_grass_traffic}")

    with open("simulation_results.txt", "a") as file:
        file.write(f"Max Iterations: {MAX_ITERATIONS}\n")
        file.write("Simulation Results:\n")
        file.write(f"Number of agents: {num_agents}\n")
        file.write(f"Total Grass Traffic: {total_grass_traffic}\n")
        file.write(f"Total Utility: {total_utility}\n")
        for goal in goals:
            avg_grass_traffic = grass_traffic_counters[goal] / goal_counters[goal] if goal_counters[goal] > 0 else 0
            file.write(f"Goal '{goal}': {goal_counters[goal]} agents, {success_counters[goal]} succeeded, "
                       f"Grass Traffic (Sum: {grass_traffic_counters[goal]}, Avg: {avg_grass_traffic:.2f})\n")
        file.write("-" * 40 + "\n\n")

    return {'totalGrassTraffic': total_grass_traffic, 'totalUtility': total_utility}


# Simulation parameters
MAX_ITERATIONS = 10
initial_x = 0
initial_y = 0

# Run the simulation
simulation_result = simulate()
print("Simulation result:", simulation_result)
