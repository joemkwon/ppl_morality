import numpy as np
from functools import lru_cache

# Define the gridworld map with various locations
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

def rule_follower_motion_utility(agent_id, location_type, motion_type):
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

def utility_function(agent_id, gridworld, state_x, state_y, action, goal, rule_following=False):
    location_type = get_gridworld_at(gridworld, state_x, state_y)
    state_location_utility = location_utility(agent_id, location_type, goal)
    motion_type = 'is_walking_diagonal' if action in ['north-west', 'north-east', 'south-west', 'south-east'] else 'is_walking'
    if rule_following:
        state_motion_utility = rule_follower_motion_utility(agent_id, location_type, motion_type)
    else:
        state_motion_utility = motion_utility(agent_id, location_type, motion_type)
    return state_location_utility + state_motion_utility

@lru_cache(maxsize=None)
def value_function(agent_id, curr_iteration, state_x, state_y, goal, rule_following=False):
    if curr_iteration == -1:
        return 0
    prev_optimal_action_value = optimal_action_value(agent_id, curr_iteration - 1, state_x, state_y, goal, rule_following)
    return prev_optimal_action_value['value']

@lru_cache(maxsize=None)
def available_actions_to_values(agent_id, curr_iteration, state_x, state_y, goal, rule_following=False):
    action_values = []
    for action in directions:
        utility = utility_function(agent_id, gridworld, state_x, state_y, action, goal, rule_following)
        next_state = gridworld_transition(gridworld, state_x, state_y, action)
        next_state_value = value_function(agent_id, curr_iteration, next_state['x'], next_state['y'], goal, rule_following)
        action_values.append({'action': action, 'value': utility + next_state_value})
    return action_values

@lru_cache(maxsize=None)
def optimal_action_value(agent_id, curr_iteration, state_x, state_y, goal, rule_following=False):
    actions_to_values = available_actions_to_values(agent_id, curr_iteration, state_x, state_y, goal, rule_following)
    return max(actions_to_values, key=lambda a: a['value'])

@lru_cache(maxsize=None)
def should_terminate(agent_id, state_x, state_y, goal):
    if value_function(agent_id, MAX_ITERATIONS, initial_x, initial_y, goal) <= 0:
        return True
    location_type = get_gridworld_at(gridworld, state_x, state_y)
    state_location_utility = location_utility(agent_id, location_type, goal)
    return state_location_utility > 0

def optimal_policy_from_initial_state(agent_id, state_x, state_y, curr_iteration, max_iterations, goal, rule_following=False):
    if should_terminate(agent_id, state_x, state_y, goal):
        return []
    curr_optimal_action_value = optimal_action_value(agent_id, curr_iteration, state_x, state_y, goal, rule_following)
    curr_optimal_action = curr_optimal_action_value['action']
    next_state = gridworld_transition(gridworld, state_x, state_y, curr_optimal_action)
    remaining_policy = optimal_policy_from_initial_state(agent_id, next_state['x'], next_state['y'], curr_iteration + 1, max_iterations, goal, rule_following)
    return [curr_optimal_action] + remaining_policy

def trajectory_from_initial_state(agent_id, state_x, state_y, curr_iteration, max_iterations, goal, rule_following=False):
    if should_terminate(agent_id, state_x, state_y, goal):
        return []
    curr_optimal_action_value = optimal_action_value(agent_id, curr_iteration, state_x, state_y, goal, rule_following)
    curr_optimal_action = curr_optimal_action_value['action']
    next_state = gridworld_transition(gridworld, state_x, state_y, curr_optimal_action)
    remaining_trajectory = trajectory_from_initial_state(agent_id, next_state['x'], next_state['y'], curr_iteration + 1, max_iterations, goal, rule_following)
    return [next_state['location']] + remaining_trajectory

def optimal_policy(agent_id, initial_state_x, initial_state_y, max_iterations, goal, rule_following=False):
    return [('start', 'start')] + optimal_policy_from_initial_state(agent_id, initial_state_x, initial_state_y, 0, max_iterations, goal, rule_following)

def optimal_trajectory(agent_id, initial_state_x, initial_state_y, max_iterations, goal, rule_following=False):
    return [get_gridworld_at(gridworld, initial_state_x, initial_state_y)] + trajectory_from_initial_state(agent_id, initial_state_x, initial_state_y, 0, max_iterations, goal, rule_following)

def optimal_policy_with_trajectory(agent_id, initial_state_x, initial_state_y, max_iterations, goal, rule_following=False):
    policy = optimal_policy(agent_id, initial_state_x, initial_state_y, max_iterations, goal, rule_following)
    trajectory = optimal_trajectory(agent_id, initial_state_x, initial_state_y, max_iterations, goal, rule_following)
    return list(zip(policy, trajectory))

def pre_simulation(agent_id, rule_following, max_grass_capacity, grass_dying_cost, main_agent_utility):
    total_utility = 0
    total_grass_traffic = 0
    
    num_agents = 100
    agent_ids = range(num_agents)
  
    goal_distribution = ['hospital', 'busStop', 'coffeeShop']
    goal_probabilities = [0.05, 0.45, 0.5]
    
    for agent_id in agent_ids:
        goal = np.random.choice(goal_distribution, p=goal_probabilities)
        initial_utility = goal_utility(agent_id, goal)
        
        # Skip agents with utility less than the main agent's utility
        if initial_utility < main_agent_utility:
            continue
        
        policy_trajectory = optimal_policy_with_trajectory(agent_id, initial_x, initial_y, MAX_ITERATIONS, goal, rule_following)
        
        utility = 0
        agent_grass_traffic = 0
        
        for step in policy_trajectory:
            action = step[0]
            location = step[1]
            if location == 'G':
                total_grass_traffic += 1
                agent_grass_traffic += 1
            
            motion_type = 'is_walking_diagonal' if action in ['north-west', 'north-east', 'south-west', 'south-east'] else 'is_walking'
            utility += motion_utility(agent_id, location, motion_type)
        
        final_location = policy_trajectory[-1][1]
        if final_location == goal:
            utility += initial_utility
        
        total_utility += utility
    
    if total_grass_traffic > max_grass_capacity:
        total_utility -= grass_dying_cost
    
    return total_utility, total_grass_traffic

def simulate(initial_x=0, initial_y=0):
    total_grass_traffic = 0
    total_utility = 0
    
    max_grass_capacity = 300
    grass_dying_cost = 10000
    
    num_agents = 100
    print(f"Number of agents: {num_agents}")
    print(f"Starting location of each agent: ({initial_x}, {initial_y})")
    agent_ids = range(num_agents)
  
    goal_distribution = ['hospital', 'busStop', 'coffeeShop']
    goal_probabilities = [0.05, 0.45, 0.5]

    goal_counters = {goal: 0 for goal in goals}
    success_counters = {goal: 0 for goal in goals}
    grass_traffic_counters = {goal: 0 for goal in goals}
    
    rule_following_utilities = []
    rule_exceptional_utilities = []
    
    rule_following_agents = 0
    rule_exceptional_agents = 0
    
    for agent_id in agent_ids:
        # Pre-simulations for the main agent
        main_agent_goal = np.random.choice(goal_distribution, p=goal_probabilities)
        main_agent_utility = goal_utility(agent_id, main_agent_goal)
        
        rule_exceptional_utility, _ = pre_simulation(agent_id, rule_following=False, max_grass_capacity=max_grass_capacity, grass_dying_cost=grass_dying_cost, main_agent_utility=main_agent_utility)
        rule_following_utility, _ = pre_simulation(agent_id, rule_following=True, max_grass_capacity=max_grass_capacity, grass_dying_cost=grass_dying_cost, main_agent_utility=main_agent_utility)
        
        rule_following_utilities.append(rule_following_utility)
        rule_exceptional_utilities.append(rule_exceptional_utility)
        
        # Determine agent type
        total_utility_sum = rule_following_utility + rule_exceptional_utility
        rule_following_probability = rule_following_utility / total_utility_sum if total_utility_sum > 0 else 0
        is_rule_following = np.random.random() < rule_following_probability
        
        if is_rule_following:
            rule_following_agents += 1
        else:
            rule_exceptional_agents += 1
        
        goal = np.random.choice(goal_distribution, p=goal_probabilities)
        goal_counters[goal] += 1
        initial_utility = goal_utility(agent_id, goal)
        
        policy_trajectory = optimal_policy_with_trajectory(agent_id, initial_x, initial_y, MAX_ITERATIONS, goal, is_rule_following)
        
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
        
        if is_rule_following:
            rule_following_utilities[-1] += utility
        else:
            rule_exceptional_utilities[-1] += utility
        
        print(f"Agent {agent_id}: Final utility {utility}, final location {final_location}, grass traffic added {agent_grass_traffic}")

    if total_grass_traffic > max_grass_capacity:
        total_utility -= grass_dying_cost
        print(f"Grass traffic exceeded max capacity. Applying penalty of {grass_dying_cost} to total utility.")

    with open("sophisticated_simulation_results.txt", "a") as file:
        file.write(f"Max Iterations: {MAX_ITERATIONS}\n")
        file.write("Simulation Results:\n")
        file.write(f"Number of agents: {num_agents}\n")
        file.write(f"Starting location of each agent: ({initial_x}, {initial_y})\n")
        file.write(f"Total Grass Traffic: {total_grass_traffic}\n")
        file.write(f"Max Grass Capacity: {max_grass_capacity}\n")
        file.write(f"Grass Dying Cost: {grass_dying_cost}\n")
        file.write(f"Total Utility: {total_utility}\n")
        file.write(f"Average Utility per Agent: {total_utility / num_agents}\n")
        file.write(f"Number of Rule-Following Agents: {rule_following_agents}\n")
        file.write(f"Rule-Following Total Utility: {sum(rule_following_utilities)}\n")
        file.write(f"Rule-Following Average Utility per Agent: {sum(rule_following_utilities) / rule_following_agents if rule_following_agents > 0 else 0}\n")
        file.write(f"Number of Rule-Exceptional Agents: {rule_exceptional_agents}\n")
        file.write(f"Rule-Exceptional Total Utility: {sum(rule_exceptional_utilities)}\n")
        file.write(f"Rule-Exceptional Average Utility per Agent: {sum(rule_exceptional_utilities) / rule_exceptional_agents if rule_exceptional_agents > 0 else 0}\n")
        for goal in goals:
            avg_grass_traffic = grass_traffic_counters[goal] / goal_counters[goal] if goal_counters[goal] > 0 else 0
            file.write(f"Goal '{goal}': {goal_counters[goal]} agents, {success_counters[goal]} succeeded, "
                       f"Grass Traffic (Sum: {grass_traffic_counters[goal]}, Avg: {avg_grass_traffic:.2f})\n")
        file.write("-" * 40 + "\n\n")

    return {'totalGrassTraffic': total_grass_traffic, 'totalUtility': total_utility, 'averageUtilityPerAgent': total_utility / num_agents, 'ruleFollowingAgents': rule_following_agents, 'ruleExceptionalAgents': rule_exceptional_agents}

# Simulation parameters
MAX_ITERATIONS = 10
initial_x = 0
initial_y = 0

# Run the simulation
simulation_result = simulate()
print("Simulation result:", simulation_result)
