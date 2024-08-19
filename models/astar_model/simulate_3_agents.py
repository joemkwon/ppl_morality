import numpy as np
import random
import heapq
import json

# Utility functions for grid and motion handling
def save_trajectories_with_simulation_data(agents_data, gridworld, simulation_parameters, agent_type, file_name="trajectories_policies.jsonl"):
    gridworld_serializable = [[str(cell) for cell in row] for row in gridworld]
    
    simulation_parameters_serializable = {}
    for k, v in simulation_parameters.items():
        if isinstance(v, np.ndarray):
            simulation_parameters_serializable[k] = v.tolist()
        elif isinstance(v, bool):
            simulation_parameters_serializable[k] = int(v)
        else:
            simulation_parameters_serializable[k] = v
    
    for agent in agents_data:
        if 'is_rule_following' in agent:
            agent['is_rule_following'] = int(agent['is_rule_following'])

    data = {
        "gridworld": gridworld_serializable,
        "simulation_parameters": simulation_parameters_serializable,
        "agents": agents_data,
        "agent_type": agent_type
    }
    
    with open(file_name, "a") as f: 
        json.dump(data, f)
        f.write("\n")

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

directions = ['west', 'east', 'north', 'south', 'north-west', 'north-east', 'south-west', 'south-east']

GOAL_MEAN_UTILITY = 100
GOAL_UTILITY_VARIANCE = 10

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

# A* Functions
def heuristic(state_x, state_y, goal_x, goal_y):
    return abs(state_x - goal_x) + abs(state_y - goal_y)

def a_star_search(initial_x, initial_y, goal_x, goal_y, gridworld, rule_following=False):
    open_list = []
    heapq.heappush(open_list, (0 + heuristic(initial_x, initial_y, goal_x, goal_y), 0, initial_x, initial_y, []))
    visited = set()

    while open_list:
        estimated_cost, cost_so_far, current_x, current_y, path = heapq.heappop(open_list)

        if (current_x, current_y) == (goal_x, goal_y):
            return path

        if (current_x, current_y) in visited:
            continue

        visited.add((current_x, current_y))

        for action in directions:
            next_x = current_x + x_increment(action)
            next_y = current_y + y_increment(action)

            if 0 <= next_x < gridworld_max_x(gridworld) and 0 <= next_y < gridworld_max_y(gridworld):
                next_state = get_gridworld_at(gridworld, next_x, next_y)
                motion_type = 'is_walking_diagonal' if action in ['north-west', 'north-east', 'south-west', 'south-east'] else 'is_walking'
                
                # Properly apply the rule-following penalty for stepping on grass
                if rule_following:
                    next_cost = rule_follower_motion_utility(next_state, motion_type)
                else:
                    next_cost = motion_utility(next_state, motion_type)

                # Ensure grass penalty is correctly calculated for rule-following agents
                new_cost = cost_so_far + next_cost

                # Avoid stepping on grass if the penalty is too high
                if rule_following and next_cost == -1000:
                    continue  # Skip this path if the agent would step on grass

                new_path = path + [(action, next_state)]
                heapq.heappush(open_list, (new_cost + heuristic(next_x, next_y, goal_x, goal_y), new_cost, next_x, next_y, new_path))
    
    return []

def optimal_policy_with_a_star(initial_x, initial_y, goal_x, goal_y, rule_following=False):
    return [('start', 'start')] + a_star_search(initial_x, initial_y, goal_x, goal_y, gridworld, rule_following)

def optimal_policy_with_trajectory_a_star(initial_x, initial_y, goal_x, goal_y, rule_following=False):
    policy = optimal_policy_with_a_star(initial_x, initial_y, goal_x, goal_y, rule_following)
    trajectory = [get_gridworld_at(gridworld, initial_x, initial_y)] + [step[1] for step in policy[1:]]
    return list(zip(policy, trajectory))

def softmax(x):
    e_x = np.exp(x - np.max(x))  
    return e_x / e_x.sum()

def pre_simulation(rule_following, max_grass_capacity, grass_dying_cost, main_agent_utility, initial_x, initial_y, goal_x, goal_y):
    total_utility = 0
    total_grass_traffic = 0
    simulated_agents = 0
    
    num_agents = 100
    agent_ids = range(num_agents)
    actual_simulated_agents = 0

    for agent_id in agent_ids:
        if simulated_agents >= 100:
            break

        sampled_goal_utility = goal_utility()
        simulated_agents += 1

        if sampled_goal_utility < main_agent_utility:
            continue

        actual_simulated_agents += 1
        policy_trajectory = optimal_policy_with_trajectory_a_star(initial_x, initial_y, goal_x, goal_y, rule_following)
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
    
    return total_utility, total_grass_traffic, actual_simulated_agents

def simulate_with_a_star(
    initial_x=0,
    initial_y=0,
    goal_x=9,
    goal_y=9,
    num_agents=100,
    max_grass_capacity=300,
    grass_dying_cost=10000,
    agent_type="main",
    seed=None
):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        print(f"Random seed set to: {seed}")

    set_gridworld_at(gridworld, goal_x, goal_y, 'F')

    total_grass_traffic = 0
    total_utility = 0

    print(f"Number of agents: {num_agents}")
    print(f"Starting location of each agent: ({initial_x}, {initial_y})")
    print(f"Goal location: ({goal_x}, {goal_y})")
    agent_ids = range(num_agents)

    agents_data = []

    rule_following_agents = 0
    rule_exceptional_agents = 0

    for agent_id in agent_ids:
        main_agent_utility = goal_utility()

        if agent_type == "rule-following":
            is_rule_following = True
        elif agent_type == "rule-exceptional":
            is_rule_following = False
        else:
            rule_exceptional_utility_simulated, _, _ = pre_simulation(
                rule_following=False,
                max_grass_capacity=max_grass_capacity,
                grass_dying_cost=grass_dying_cost,
                main_agent_utility=main_agent_utility,
                initial_x=initial_x,
                initial_y=initial_y,
                goal_x=goal_x,
                goal_y=goal_y
            )
            rule_following_utility_simulated, _, _ = pre_simulation(
                rule_following=True,
                max_grass_capacity=max_grass_capacity,
                grass_dying_cost=grass_dying_cost,
                main_agent_utility=main_agent_utility,
                initial_x=initial_x,
                initial_y=initial_y,
                goal_x=goal_x,
                goal_y=goal_y
            )

            utilities = np.array([rule_following_utility_simulated, rule_exceptional_utility_simulated])
            probabilities = softmax(utilities)
            rule_following_probability = probabilities[0]
            is_rule_following = np.random.random() < rule_following_probability

        policy_trajectory = optimal_policy_with_trajectory_a_star(
            initial_x,
            initial_y,
            goal_x,
            goal_y,
            is_rule_following
        )

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
        else:
            rule_exceptional_agents += 1

        print(f"Agent {agent_id}: Final utility {utility_main}, final location {final_location}, grass traffic added {agent_grass_traffic}")
    
    if total_grass_traffic > max_grass_capacity:
        total_utility -= grass_dying_cost
        print(f"Grass traffic exceeded max capacity. Applying penalty of {grass_dying_cost} to total utility.")

    # Save results to combined_simulation_results.txt
    with open("combined_simulation_results.txt", "a") as file:
        file.write(f"Random Seed: {seed}\n" if seed is not None else "Random Seed: None\n")
        file.write(f"Simulation Results for Agent Type: {agent_type}\n")
        file.write(f"Total Grass Traffic: {total_grass_traffic}\n")
        file.write(f"Max Grass Capacity: {max_grass_capacity}\n")
        file.write(f"Grass Dying Cost: {grass_dying_cost}\n")
        file.write(f"Total Utility: {total_utility}\n")
        file.write(f"Average Utility per Agent: {total_utility / num_agents}\n")
        file.write(f"Number of Rule-Following Agents: {rule_following_agents}\n")
        file.write(f"Number of Rule-Exceptional Agents: {rule_exceptional_agents}\n")
        file.write("~" * 40 + "\n\n")  # Delimiter between agent types

    simulation_parameters = {
        "initial_x": initial_x,
        "initial_y": initial_y,
        "goal_x": goal_x,
        "goal_y": goal_y,
        "num_agents": num_agents,
        "max_grass_capacity": max_grass_capacity,
        "grass_dying_cost": grass_dying_cost,
        "seed": seed,
    }
    
    save_trajectories_with_simulation_data(agents_data, gridworld, simulation_parameters, agent_type)

def run_simulation_for_all_agents(
    initial_x=0,
    initial_y=0,
    goal_x=9,
    goal_y=9,
    num_agents=100,
    max_grass_capacity=300,
    grass_dying_cost=10000,
    seed=None
):
    agent_types = ["main", "rule-following", "rule-exceptional"]

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        print(f"Random seed set to: {seed}")

    set_gridworld_at(gridworld, goal_x, goal_y, 'F')

    for agent_type in agent_types:
        simulate_with_a_star(
            initial_x=initial_x,
            initial_y=initial_y,
            goal_x=goal_x,
            goal_y=goal_y,
            num_agents=num_agents,
            max_grass_capacity=max_grass_capacity,
            grass_dying_cost=grass_dying_cost,
            agent_type=agent_type,
            seed=seed
        )

if __name__ == "__main__":
    run_simulation_for_all_agents(
        initial_x=0,
        initial_y=0,
        goal_x=9,
        goal_y=9,
        num_agents=100,
        max_grass_capacity=300,
        grass_dying_cost=10000,
        seed=42
    )
