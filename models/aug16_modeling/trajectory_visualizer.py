import matplotlib.pyplot as plt
import json
import os
import numpy as np
from matplotlib import cm

# Function to draw the gridworld
def draw_grid(ax, grid):
    colors = {'S': 'lightgrey', 'G': 'darkgreen', 'F': 'yellow'}
    n = len(grid)
    m = len(grid[0])

    for i in range(n):
        for j in range(m):
            rect = plt.Rectangle((j, n - i - 1), 1, 1, facecolor=colors.get(grid[i][j], 'white'))
            ax.add_patch(rect)

    ax.set_xlim(0, m)
    ax.set_ylim(0, n)
    ax.set_xticks(np.arange(m + 1))
    ax.set_yticks(np.arange(n + 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True)

# Function to visualize trajectories on the grid
def plot_trajectories(jsonl_file, output_folder='visualizations'):
    # Create the visualizations folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Load JSONL data and create a visualization for each line
    with open(jsonl_file, 'r') as f:
        lines = f.readlines()
    
    cmap = plt.colormaps.get_cmap('coolwarm')

    for line_idx, line in enumerate(lines):
        data = json.loads(line)
        
        # Get the gridworld and agents data from the jsonl line
        gridworld = data['gridworld']
        agents = data['agents']
        simulation_parameters = data['simulation_parameters']
        
        # Print the initial coordinates
        print(f"Initial X: {simulation_parameters['initial_x']}, Initial Y: {simulation_parameters['initial_y']}")
        
        # Create a new plot for each line of data
        fig, ax = plt.subplots(figsize=(10, 10))
        draw_grid(ax, gridworld)

        # Plot the initial position only for debugging
        ax.plot(simulation_parameters['initial_x'] + 0.5, len(gridworld) - simulation_parameters['initial_y'] - 1 + 0.5, 'ro', markersize=10, label='Initial Position')
        
        # Plot each agent's trajectory
        for agent in agents:
            # Extract the trajectory as a list of location types
            trajectory = [t[1] for t in agent['policy_trajectory']]
            print(f"Agent {agent['agent_id']} Trajectory: {trajectory}")
            print(f"Agent {agent['agent_id']} Actions: {[t[0] for t in agent['policy_trajectory']]}")
            
            # Initialize with starting position
            x_coords = [simulation_parameters['initial_x']]
            y_coords = [simulation_parameters['initial_y']]

            # Iterate through trajectory actions and simulate movements
            for step in agent['policy_trajectory'][1:]:
                action = step[0]

                # Update x, y coordinates based on the action
                if action == 'west':
                    x_coords.append(x_coords[-1] - 1)
                elif action == 'east':
                    x_coords.append(x_coords[-1] + 1)
                elif action == 'north':
                    y_coords.append(y_coords[-1] + 1)
                elif action == 'south':
                    y_coords.append(y_coords[-1] - 1)
                elif action == 'north-west':
                    x_coords.append(x_coords[-1] - 1)
                    y_coords.append(y_coords[-1] + 1)
                elif action == 'north-east':
                    x_coords.append(x_coords[-1] + 1)
                    y_coords.append(y_coords[-1] + 1)
                elif action == 'south-west':
                    x_coords.append(x_coords[-1] - 1)
                    y_coords.append(y_coords[-1] - 1)
                elif action == 'south-east':
                    x_coords.append(x_coords[-1] + 1)
                    y_coords.append(y_coords[-1] - 1)

                # Ensure we maintain the same number of x and y points
                if len(x_coords) > len(y_coords):
                    y_coords.append(y_coords[-1])
                elif len(y_coords) > len(x_coords):
                    x_coords.append(x_coords[-1])

            # Adjust y-coordinates for grid orientation
            y_coords = [len(gridworld) - y - 1 for y in y_coords]

            # Debugging: Print first few coordinates for verification
            print(f"Agent {agent['agent_id']} x_coords: {x_coords[:5]}")
            print(f"Agent {agent['agent_id']} y_coords: {y_coords[:5]}")
            
            # Color grading along the trajectory
            num_steps = len(x_coords)
            colors = [cmap(i / num_steps) for i in range(num_steps)]
            
            # Plot each segment of the trajectory with a gradient
            for i in range(num_steps - 1):
                ax.plot(x_coords[i:i+2], y_coords[i:i+2], color=colors[i], linewidth=2, alpha=0.7)

        # Add simulation parameters as text outside the grid
        param_text = '\n'.join([f"{k}: {v}" for k, v in simulation_parameters.items()])
        
        # Move text below the grid
        plt.text(0.5, -0.1, param_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.5))

        # Save each visualization separately
        output_file = os.path.join(output_folder, f'trajectories_visualization_{line_idx + 1}.png')
        plt.savefig(output_file)
        plt.close()

if __name__ == "__main__":
    plot_trajectories('trajectories_policies.jsonl')
