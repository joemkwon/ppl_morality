import numpy as np

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

# Memoization for parameters
myopia_parameters = {}
grass_fragility_parameters = {}

def get_myopia_parameter(agent_id):
    if agent_id not in myopia_parameters:
        myopia_parameters[agent_id] = abs(np.random.normal(0.5, 0.1))
    return myopia_parameters[agent_id]

def get_grass_fragility_parameter(community_id):
    if community_id not in grass_fragility_parameters:
        grass_fragility_parameters[community_id] = np.random.uniform(0, 1)
    return grass_fragility_parameters[community_id]

def score_plan(gridworld, trajectory, goal_utility, myopia_parameter):
    short_term_plan_u = short_term_plan_utility(trajectory, goal_utility)
    grass_shortcutiness = len(trajectory) / len(full_rule_following_trajectory(gridworld, trajectory[0], trajectory[-1]))
    universalized_cost_for_grass_damage = universalized_plan_utility(gridworld, trajectory, goal_utility, grass_shortcutiness)
    overall_trajectory_utility = short_term_plan_u + myopia_parameter * universalized_cost_for_grass_damage
    return overall_trajectory_utility

def short_term_plan_utility(trajectory, goal_utility):
    reward_for_goal = goal_utility if trajectory[-1] == 'F' else 0
    cost_for_path_length = -len(trajectory)
    return reward_for_goal + cost_for_path_length

def universalized_plan_utility(gridworld, trajectory, my_goal_utility, my_grass_shortcutiness):
    total_grass_steps = 0
    for agent_id in range(100):  # Simulate for 100 agents
        starting_location = sample_starting_location(gridworld)
        goal_location = sample_goal_location(gridworld)
        goal_utility = goal_utility()
        length_strict_rule_following = len(full_rule_following_trajectory(gridworld, starting_location, goal_location))
        if goal_utility >= my_goal_utility:
            grass_steps = my_grass_shortcutiness * length_strict_rule_following
            total_grass_steps += grass_steps
    
    standard_grass_capacity = 200 # Example value
    grass_community_utility = 10000  # Example value
    grass_fragility = get_grass_fragility_parameter(1)  # Example community ID
    grass_capacity = standard_grass_capacity * grass_fragility
    
    # if total grass steps is greater than grass capacity, then we incur the grass_community_utility penalty
    return 0 if total_grass_steps <= grass_capacity else -grass_community_utility

def full_rule_following_trajectory(gridworld, start, goal):
    from collections import deque

    def get_neighbors(x, y):
        directions = [
            ('west', -1, 0), ('east', 1, 0), 
            ('north', 0, -1), ('south', 0, 1),
            ('north-west', -1, -1), ('north-east', 1, -1),
            ('south-west', -1, 1), ('south-east', 1, 1)
        ]
        neighbors = []
        for direction, dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(gridworld[0]) and 0 <= ny < len(gridworld) and gridworld[ny][nx] == 'S':
                neighbors.append((nx, ny, direction))
        return neighbors

    start_x, start_y = start
    goal_x, goal_y = goal
    queue = deque([(start_x, start_y, [])])
    visited = set((start_x, start_y))

    while queue:
        x, y, path = queue.popleft()
        if (x, y) == (goal_x, goal_y):
            return path

        for nx, ny, direction in get_neighbors(x, y):
            if (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny, path + [direction]))

    return []  # Return an empty path if no valid path is found

def sample_starting_location(gridworld):
    # Implement the logic to sample a starting location
    pass

def sample_goal_location(gridworld):
    # Implement the logic to sample a goal location
    pass
