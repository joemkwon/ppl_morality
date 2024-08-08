import numpy as np
import random
from scipy.stats import norm

# Define the gridworld map with various locations
gridworld = [
    ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'hospital'],
    ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'coffeeShop'],
    ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'busStop']
]

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

def location_utility(agent_id, location_type):
    if location_type in ['G', 'S']:
        return 0
    else:
        return goal_utility(agent_id, location_type)

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

def utility_function(agent_id, gridworld, state_x, state_y, action):
    location_type = get_gridworld_at(gridworld, state_x, state_y)
    state_location_utility = location_utility(agent_id, location_type)
    motion_type = 'is_walking_diagonal' if action in ['north-west', 'north-east', 'south-west', 'south-east'] else 'is_walking'
    state_motion_utility = motion_utility(agent_id, location_type, motion_type)
    return state_location_utility + state_motion_utility

def value_function(agent_id, curr_iteration, gridworld, state_x, state_y, max_iterations):
    if curr_iteration == -1:
        return 0
    prev_optimal_action_value = optimal_action_value(agent_id, curr_iteration - 1, gridworld, state_x, state_y, max_iterations)
    return prev_optimal_action_value['value']

def available_actions_to_values(agent_id, curr_iteration, gridworld, state_x, state_y, max_iterations):
    return [{'action': action, 'value': utility_function(agent_id, gridworld, state_x, state_y, action) + value_function(agent_id, curr_iteration, gridworld, gridworld_transition(gridworld, state_x, state_y, action)['x'], gridworld_transition(gridworld, state_x, state_y, action)['y'], max_iterations)} for action in directions]

def optimal_action_value(agent_id, curr_iteration, gridworld, state_x, state_y, max_iterations):
    actions_to_values = available_actions_to_values(agent_id, curr_iteration, gridworld, state_x, state_y, max_iterations)
    return max(actions_to_values, key=lambda a: a['value'])

def should_terminate(agent_id, gridworld, state_x, state_y, initial_x, initial_y, max_iterations):
    if value_function(agent_id, max_iterations, gridworld, initial_x, initial_y, max_iterations) <= 0:
        return True
    location_type = get_gridworld_at(gridworld, state_x, state_y)
    state_location_utility = location_utility(agent_id, location_type)
    return state_location_utility > 0

def optimal_policy_from_initial_state(agent_id, gridworld, state_x, state_y, max_iterations):
    if should_terminate(agent_id, gridworld, state_x, state_y, state_x, state_y, max_iterations):
        return []
    curr_optimal_action_value = optimal_action_value(agent_id, max_iterations, gridworld, state_x, state_y, max_iterations)
    curr_optimal_action = curr_optimal_action_value['action']
    next_state = gridworld_transition(gridworld, state_x, state_y, curr_optimal_action)
    remaining_policy = optimal_policy_from_initial_state(agent_id, gridworld, next_state['x'], next_state['y'], max_iterations)
    return [curr_optimal_action] + remaining_policy

def trajectory_from_initial_state(agent_id, gridworld, state_x, state_y, max_iterations):
    if should_terminate(agent_id, gridworld, state_x, state_y, state_x, state_y, max_iterations):
        return []
    curr_optimal_action_value = optimal_action_value(agent_id, max_iterations, gridworld, state_x, state_y, max_iterations)
    curr_optimal_action = curr_optimal_action_value['action']
    next_state = gridworld_transition(gridworld, state_x, state_y, curr_optimal_action)
    remaining_trajectory = trajectory_from_initial_state(agent_id, gridworld, next_state['x'], next_state['y'], max_iterations)
    return [next_state['location']] + remaining_trajectory

def optimal_policy(agent_id, gridworld, initial_state_x, initial_state_y, max_iterations):
    return [['start', 'start']] + optimal_policy_from_initial_state(agent_id, gridworld, initial_state_x, initial_state_y, max_iterations)

def optimal_trajectory(agent_id, gridworld, initial_state_x, initial_state_y, max_iterations):
    return [get_gridworld_at(gridworld, initial_state_x, initial_state_y)] + trajectory_from_initial_state(agent_id, gridworld, initial_state_x, initial_state_y, max_iterations)

def optimal_policy_with_trajectory(agent_id, gridworld, initial_state_x, initial_state_y, max_iterations):
    policy = optimal_policy(agent_id, gridworld, initial_state_x, initial_state_y, max_iterations)
    trajectory = optimal_trajectory(agent_id, gridworld, initial_state_x, initial_state_y, max_iterations)
    return list(zip(policy, trajectory))

def get_terminal_goal_state(agent_id, gridworld, initial_state_x, initial_state_y, max_iterations):
    return optimal_trajectory(agent_id, gridworld, initial_state_x, initial_state_y, max_iterations)[-1]

def trajectory_has_location_type(agent_id, location_type, gridworld, initial_state_x, initial_state_y, max_iterations):
    return location_type in optimal_trajectory(agent_id, gridworld, initial_state_x, initial_state_y, max_iterations)

def policy_has_motion_type(agent_id, motion_type, gridworld, initial_state_x, initial_state_y, max_iterations):
    policy_motions = [action[0] for action in optimal_policy(agent_id, gridworld, initial_state_x, initial_state_y, max_iterations)]
    return motion_type in policy_motions

def policy_and_trajectory_has_motion_at_location(agent_id, motion_type, location_type, gridworld, initial_state_x, initial_state_y, max_iterations):
    policy_motions = [action[0] for action in optimal_policy(agent_id, gridworld, initial_state_x, initial_state_y, max_iterations)]
    trajectory = optimal_trajectory(agent_id, gridworld, initial_state_x, initial_state_y, max_iterations)
    motions_at_locations = list(zip(policy_motions, trajectory))
    return (motion_type, location_type) in motions_at_locations

def motion_at_location(agent_id, motion_type, location_type, gridworld, initial_state_x, initial_state_y, max_iterations):
    policy_motions = [action[0] for action in optimal_policy(agent_id, gridworld, initial_state_x, initial_state_y, max_iterations)]
    trajectory = optimal_trajectory(agent_id, gridworld, initial_state_x, initial_state_y, max_iterations)
    return list(zip(policy_motions, trajectory))

# Derived predicates and action definition
def gensym(prefix):
    counter = 0
    def generate():
        nonlocal counter
        counter += 1
        return f"{prefix}{counter}"
    return generate

action_id_gensym = gensym("action-")

def is_going_to_actions(agent_id, gridworld, initial_x, initial_y, max_iterations):
    action_states = optimal_policy_with_trajectory(agent_id, gridworld, initial_x, initial_y, max_iterations)
    final_location = action_states[-1][1]
    return [[
        ['action_id', action_id_gensym()],
        ['action_subject', agent_id],
        ['action_predicates', ['is_going', ['to', final_location]]],
        ['action_preposition', 'to'],
        ['action_location', final_location]
    ]]

def is_going_on_actions(agent_id, gridworld, initial_x, initial_y, max_iterations):
    action_states = optimal_policy_with_trajectory(agent_id, gridworld, initial_x, initial_y, max_iterations)
    return [
        [
            ['action_id', action_id_gensym()],
            ['action_subject', agent_id],
            ['action_predicates', ['is_going', action_state[0][0], action_state[0][1], ['on', action_state[1]]]],
            ['action_preposition', 'on'],
            ['action_location', action_state[1]]
        ]
        for action_state in action_states
    ]

def actions_in_scene(agent_id, gridworld, initial_x, initial_y, max_iterations):
    return is_going_to_actions(agent_id, gridworld, initial_x, initial_y, max_iterations) + is_going_on_actions(agent_id, gridworld, initial_x, initial_y, max_iterations)

def is_action(action, action_predicate):
    return action_predicate in action['action_predicates']

def is_subject_of_action(action, entity):
    return action['action_subject'] == entity

def is_preposition_of_action(action, preposition):
    return action['action_preposition'] == preposition

def is_location_of_action(action, location):
    return action['action_location'] == location

def get_location(action):
    return action['action_location']

def exists_action(agent_id, predicate, gridworld, initial_x, initial_y, max_iterations):
    return any(predicate(action) for action in actions_in_scene(agent_id, gridworld, initial_x, initial_y, max_iterations))

def get_actions(agent_id, predicate, gridworld, initial_x, initial_y, max_iterations):
    return [action for action in actions_in_scene(agent_id, gridworld, initial_x, initial_y, max_iterations) if predicate(action)]

# Simulation parameters
MAX_ITERATIONS = 100
initial_x = 0
initial_y = gridworld_max_y(gridworld) - 1

def simulate():
    total_grass_traffic = 0
    total_utility = 0
    
    num_agents = int(np.random.normal(200, 25))
    agent_ids = range(num_agents)
  
    goal_distribution = ['hospital', 'busStop', 'coffeeShop']
    goal_probabilities = [0.05, 0.45, 0.5]
    
    for agent_id in agent_ids:
        goal = np.random.choice(goal_distribution, p=goal_probabilities)
        initial_utility = goal_utility(agent_id, goal)
        
        policy_trajectory = optimal_policy_with_trajectory(agent_id, gridworld, initial_x, initial_y, MAX_ITERATIONS)
        
        utility = 0
        
        for step in policy_trajectory:
            action = step[0]
            location = step[1]
            if location == 'G':
                total_grass_traffic += 1
            
            motion_type = 'is_walking_diagonal' if action in ['north-west', 'north-east', 'south-west', 'south-east'] else 'is_walking'
            utility += motion_utility(agent_id, location, motion_type)
        
        final_location = policy_trajectory[-1][1]
        if final_location == goal:
            utility += initial_utility
        
        total_utility += utility
    
    return {'totalGrassTraffic': total_grass_traffic, 'totalUtility': total_utility}

# Run the simulation
simulation_result = simulate()
print(simulation_result)
