import numpy as np
import random
from functools import lru_cache
import json

def save_trajectories_with_simulation_data(agents_data, gridworld, simulation_parameters, agent_type, file_name="noisy_trajectories_policies.jsonl"):
    # Ensure gridworld is serializable
    gridworld_serializable = [[str(cell) for cell in row] for row in gridworld]
    
    simulation_parameters_serializable = {}
    for k, v in simulation_parameters.items():
        if isinstance(v, np.ndarray):
            simulation_parameters_serializable[k] = v.tolist()  # Convert NumPy arrays to lists
        elif isinstance(v, bool):
            simulation_parameters_serializable[k] = int(v)  # Convert booleans to integers
        else:
            simulation_parameters_serializable[k] = v  # Keep everything else as is
    
    for agent in agents_data:
        if 'is_rule_following' in agent:
            agent['is_rule_following'] = int(agent['is_rule_following'])  # Convert boolean to integer

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


def utility_function(gridworld, state_x, state_y, action, goal_x, goal_y, rule_following=False):
    location_type = get_gridworld_at(gridworld, state_x, state_y)
    state_location_utility = location_utility(state_x, state_y, goal_x, goal_y)
    motion_type = 'is_walking_diagonal' if action in ['north-west', 'north-east', 'south-west', 'south-east'] else 'is_walking'
    if rule_following:
        state_motion_utility = rule_follower_motion_utility(location_type, motion_type)
    else:
        state_motion_utility = motion_utility(location_type, motion_type)
    return state_location_utility + state_motion_utility

@lru_cache(maxsize=None)
def value_function(curr_iteration, state_x, state_y, goal_x, goal_y, max_iterations, initial_x, initial_y, rule_following=False):
    if curr_iteration == -1:
        return -abs(state_x - goal_x) - abs(state_y - goal_y)
    prev_optimal_action_value = optimal_action_value(curr_iteration - 1, state_x, state_y, goal_x, goal_y, max_iterations, initial_x, initial_y, rule_following)
    return prev_optimal_action_value['value']

@lru_cache(maxsize=None)
def available_actions_to_values(curr_iteration, state_x, state_y, goal_x, goal_y, max_iterations, initial_x, initial_y, rule_following=False):
    action_values = []
    for action in directions:
        next_x = state_x + x_increment(action)
        next_y = state_y + y_increment(action)
        
        if next_x < 0 or next_x >= gridworld_max_x(gridworld) or next_y < 0 or next_y >= gridworld_max_y(gridworld):
            continue
        
        utility = utility_function(gridworld, state_x, state_y, action, goal_x, goal_y, rule_following)
        next_state = gridworld_transition(gridworld, state_x, state_y, action)
        next_state_value = value_function(curr_iteration, next_state['x'], next_state['y'], goal_x, goal_y, max_iterations, initial_x, initial_y, rule_following)
        action_values.append({'action': action, 'value': utility + next_state_value})
    
    return action_values

def softmax(values, temperature=1.0):
    values = np.array([v['value'] for v in values])
    values = values / temperature
    max_value = np.max(values)
    exp_values = np.exp(values - max_value)
    probabilities = exp_values / np.sum(exp_values)
    return probabilities

def softmax_utilities(values):
    values = np.array(values)
    exp_values = np.exp(values - np.max(values))  # Numerical stability trick
    probabilities = exp_values / np.sum(exp_values)
    return probabilities


@lru_cache(maxsize=None)
def optimal_action_value(curr_iteration, state_x, state_y, goal_x, goal_y, max_iterations, initial_x, initial_y, rule_following=False, temperature=1.0):
    actions_to_values = available_actions_to_values(curr_iteration, state_x, state_y, goal_x, goal_y, max_iterations, initial_x, initial_y, rule_following)
    
    probabilities = softmax(actions_to_values, temperature)
    
    chosen_index = np.random.choice(len(actions_to_values), p=probabilities)
    return actions_to_values[chosen_index]

@lru_cache(maxsize=None)
def should_terminate(state_x, state_y, goal_x, goal_y, max_iterations, initial_x, initial_y):
    if value_function(max_iterations, initial_x, initial_y, goal_x, goal_y, max_iterations, initial_x, initial_y) <= 0:
        return True
    return (state_x, state_y) == (goal_x, goal_y)

def optimal_policy_from_initial_state(state_x, state_y, curr_iteration, max_iterations, goal_x, goal_y, initial_x, initial_y, rule_following=False, temperature=1.0):
    if should_terminate(state_x, state_y, goal_x, goal_y, max_iterations, initial_x, initial_y):
        return []
    curr_optimal_action_value = optimal_action_value(curr_iteration, state_x, state_y, goal_x, goal_y, max_iterations, initial_x, initial_y, rule_following, temperature)
    curr_optimal_action = curr_optimal_action_value['action']
    next_state = gridworld_transition(gridworld, state_x, state_y, curr_optimal_action)
    remaining_policy = optimal_policy_from_initial_state(next_state['x'], next_state['y'], curr_iteration + 1, max_iterations, goal_x, goal_y, initial_x, initial_y, rule_following, temperature)
    return [curr_optimal_action] + remaining_policy

def trajectory_from_initial_state(state_x, state_y, curr_iteration, max_iterations, goal_x, goal_y, initial_x, initial_y, rule_following=False, temperature=1.0):
    if should_terminate(state_x, state_y, goal_x, goal_y, max_iterations, initial_x, initial_y):
        return []
    curr_optimal_action_value = optimal_action_value(curr_iteration, state_x, state_y, goal_x, goal_y, max_iterations, initial_x, initial_y, rule_following, temperature)
    curr_optimal_action = curr_optimal_action_value['action']
    next_state = gridworld_transition(gridworld, state_x, state_y, curr_optimal_action)
    remaining_trajectory = trajectory_from_initial_state(next_state['x'], next_state['y'], curr_iteration + 1, max_iterations, goal_x, goal_y, initial_x, initial_y, rule_following, temperature)
    return [next_state['location']] + remaining_trajectory

def optimal_policy(initial_state_x, initial_state_y, max_iterations, goal_x, goal_y, rule_following=False, temperature=1.0):
    return [('start', 'start')] + optimal_policy_from_initial_state(initial_state_x, initial_state_y, 0, max_iterations, goal_x, goal_y, initial_state_x, initial_state_y, rule_following, temperature)

def optimal_trajectory(initial_state_x, initial_state_y, max_iterations, goal_x, goal_y, rule_following=False, temperature=1.0):
    return [get_gridworld_at(gridworld, initial_state_x, initial_state_y)] + trajectory_from_initial_state(initial_state_x, initial_state_y, 0, max_iterations, goal_x, goal_y, initial_state_x, initial_state_y, rule_following, temperature)

def optimal_policy_with_trajectory(initial_state_x, initial_state_y, max_iterations, goal_x, goal_y, rule_following=False, temperature=1.0):
    policy = optimal_policy(initial_state_x, initial_state_y, max_iterations, goal_x, goal_y, rule_following, temperature)
    trajectory = optimal_trajectory(initial_state_x, initial_state_y, max_iterations, goal_x, goal_y, rule_following, temperature)
    return list(zip(policy, trajectory))

def pre_simulation(rule_following, max_grass_capacity, grass_dying_cost, main_agent_utility, initial_x, initial_y, goal_x, goal_y, max_iterations):
    total_utility = 0
    total_grass_traffic = 0
    simulated_agents = 0
    
    num_agents = 100  # Simulate up to 100 agents
    agent_ids = range(num_agents)

    actual_simulated_agents = 0  # Track the number of actual simulated agents

    for agent_id in agent_ids:
        # Stop once 100 agents have been simulated
        if simulated_agents >= 100:
            break

        # Sample goal utility for pre-simulated agent
        sampled_goal_utility = goal_utility()

        # Increment the count of simulated agents
        simulated_agents += 1

        # Skip agents with utility less than the main agent's utility
        if sampled_goal_utility < main_agent_utility:
            continue

        # Increment the actual simulated agent count
        actual_simulated_agents += 1

        # Get the policy trajectory for this agent
        policy_trajectory = optimal_policy_with_trajectory(initial_x, initial_y, max_iterations, goal_x, goal_y, rule_following)
        
        utility = 0
        agent_grass_traffic = 0

        # Calculate the utility and grass traffic for this agent
        for step in policy_trajectory:
            action = step[0]
            location = step[1]
            if location == 'G':
                total_grass_traffic += 1
                agent_grass_traffic += 1
            
            motion_type = 'is_walking_diagonal' if action in ['north-west', 'north-east', 'south-west', 'south-east'] else 'is_walking'
            utility += motion_utility(location, motion_type)

        # Add the goal utility if the agent reaches the final location
        final_location = policy_trajectory[-1][1]
        if final_location == 'F':
            utility += sampled_goal_utility
        
        # Accumulate the utility for all agents
        total_utility += utility

    # Apply penalty if the total grass traffic exceeds the maximum grass capacity
    if total_grass_traffic > max_grass_capacity:
        total_utility -= grass_dying_cost
    
    # Return the total utility, total grass traffic, and the number of actual simulated agents
    return total_utility, total_grass_traffic, actual_simulated_agents


def simulate(
    initial_x=0,
    initial_y=0,
    goal_x=9,
    goal_y=9,
    MAX_ITERATIONS=10,
    num_agents=100,
    max_grass_capacity=300,
    grass_dying_cost=10000,
    agent_type="main",  
    seed=None,
    temperature=1.0  
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
                goal_y=goal_y,
                max_iterations=MAX_ITERATIONS
            )
            rule_following_utility_simulated, _, _ = pre_simulation(
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

            utilities = np.array([rule_following_utility_simulated, rule_exceptional_utility_simulated])
            probabilities = softmax_utilities(utilities)
            rule_following_probability = probabilities[0]
            is_rule_following = np.random.random() < rule_following_probability

        policy_trajectory = optimal_policy_with_trajectory(
            initial_x,
            initial_y,
            MAX_ITERATIONS,
            goal_x,
            goal_y,
            is_rule_following,
            temperature
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
    
    save_trajectories_with_simulation_data(agents_data, gridworld, simulation_parameters, agent_type)

def run_simulation_for_all_agents(
    initial_x=0,
    initial_y=0,
    goal_x=9,
    goal_y=9,
    MAX_ITERATIONS=10,
    num_agents=100,
    max_grass_capacity=300,
    grass_dying_cost=10000,
    seed=None,
    temperature=1.0  
):
    agent_types = ["main", "rule-following", "rule-exceptional"]

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        print(f"Random seed set to: {seed}")

    set_gridworld_at(gridworld, goal_x, goal_y, 'F')

    all_simulation_results = []

    for agent_type in agent_types:
        total_grass_traffic = 0
        total_utility = 0
        rule_following_agents = 0
        rule_exceptional_agents = 0

        agents_data = []  

        for agent_id in range(num_agents):
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
                    goal_y=goal_y,
                    max_iterations=MAX_ITERATIONS
                )
                rule_following_utility_simulated, _, _ = pre_simulation(
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

                utilities = np.array([rule_following_utility_simulated, rule_exceptional_utility_simulated])
                print(f"Rule-following utility: {rule_following_utility_simulated}, Rule-exceptional utility: {rule_exceptional_utility_simulated}")
                probabilities = softmax_utilities(utilities)
                rule_following_probability = probabilities[0]
                is_rule_following = np.random.random() < rule_following_probability

            policy_trajectory = optimal_policy_with_trajectory(
                initial_x,
                initial_y,
                MAX_ITERATIONS,
                goal_x,
                goal_y,
                is_rule_following,
                temperature
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

        if total_grass_traffic > max_grass_capacity:
            total_utility -= grass_dying_cost

        simulation_result = {
            'agent_type': agent_type,
            'totalGrassTraffic': total_grass_traffic,
            'totalUtility': total_utility,
            'averageUtilityPerAgent': total_utility / num_agents,
            'ruleFollowingAgents': rule_following_agents,
            'ruleExceptionalAgents': rule_exceptional_agents
        }

        all_simulation_results.append(simulation_result)

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
        save_trajectories_with_simulation_data(agents_data, gridworld, simulation_parameters, agent_type)

    with open("noisy_combined_simulation_results.txt", "a") as file:
        file.write(f"Random Seed: {seed}\n" if seed is not None else "Random Seed: None\n")
        file.write(f"Max Iterations: {MAX_ITERATIONS}\n")
        file.write(f"Number of agents: {num_agents}\n")
        file.write(f"Starting location of each agent: ({initial_x}, {initial_y})\n")
        file.write(f"Goal location: ({goal_x}, {goal_y})\n")

        for result in all_simulation_results:
            file.write("Simulation Results:\n")
            file.write(f"Agent Type: {result['agent_type']}\n")
            file.write(f"Total Grass Traffic: {result['totalGrassTraffic']}\n")
            file.write(f"Max Grass Capacity: {max_grass_capacity}\n")
            file.write(f"Grass Dying Cost: {grass_dying_cost}\n")
            file.write(f"Total Utility: {result['totalUtility']}\n")
            file.write(f"Average Utility per Agent: {result['averageUtilityPerAgent']}\n")
            file.write(f"Number of Rule-Following Agents: {result['ruleFollowingAgents']}\n")
            file.write(f"Number of Rule-Exceptional Agents: {result['ruleExceptionalAgents']}\n")
            file.write("~" * 40 + "\n\n")

        file.write("-" * 40 + "\n\n")


if __name__ == "__main__":
    run_simulation_for_all_agents(
        initial_x=0,
        initial_y=0,
        goal_x=9,
        goal_y=9,
        MAX_ITERATIONS=50,
        num_agents=100,
        max_grass_capacity=300,
        grass_dying_cost=10000,
        seed=42,
        temperature=0.1
    )
