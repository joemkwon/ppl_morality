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
            rect = plt.Rectangle((j, i), 1, 1, facecolor=colors.get(grid[i][j], 'white'))
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
        
        # Create a new plot for each line of data
        fig, ax = plt.subplots(figsize=(10, 10))
        draw_grid(ax, gridworld)

        # Invert the y-axis to make (0,0) at the top-left corner
        ax.invert_yaxis()

        # Plot the initial position
        ax.plot(simulation_parameters['initial_x'] + 0.5, simulation_parameters['initial_y'] + 0.5, 'ro', markersize=10, label='Initial Position')
        
        # Plot each agent's trajectory
        for agent in agents:
            # Initialize with starting position (with a 0.5 offset to center the lines)
            x_coords = [simulation_parameters['initial_x'] + 0.5]
            y_coords = [simulation_parameters['initial_y'] + 0.5]

            # Iterate through trajectory actions and simulate movements
            for step in agent['policy_trajectory'][1:]:
                action = step[0]
                
                # Update x, y coordinates based on the action
                if action == 'west':
                    x_coords.append(x_coords[-1] - 1)
                    y_coords.append(y_coords[-1])
                elif action == 'east':
                    x_coords.append(x_coords[-1] + 1)
                    y_coords.append(y_coords[-1])
                elif action == 'north':
                    x_coords.append(x_coords[-1])
                    y_coords.append(y_coords[-1] - 1)
                elif action == 'south':
                    x_coords.append(x_coords[-1])
                    y_coords.append(y_coords[-1] + 1)
                elif action == 'north-west':
                    x_coords.append(x_coords[-1] - 1)
                    y_coords.append(y_coords[-1] - 1)
                elif action == 'north-east':
                    x_coords.append(x_coords[-1] + 1)
                    y_coords.append(y_coords[-1] - 1)
                elif action == 'south-west':
                    x_coords.append(x_coords[-1] - 1)
                    y_coords.append(y_coords[-1] + 1)
                elif action == 'south-east':
                    x_coords.append(x_coords[-1] + 1)
                    y_coords.append(y_coords[-1] + 1)

            # Color grading along the trajectory
            num_steps = len(x_coords)
            colors = [cmap(i / num_steps) for i in range(num_steps)]
            
            # Plot each segment of the trajectory with a gradient
            for i in range(num_steps - 1):
                ax.plot(x_coords[i:i+2], y_coords[i:i+2], color=colors[i], linewidth=2, alpha=0.7)

        # Add a shorter title
        ax.set_title("Agent Trajectories", fontsize=14)

        # Display simulation parameters below the grid as a text box
        param_text = '\n'.join([f"{k}: {v}" for k, v in simulation_parameters.items()])
        plt.figtext(0.5, -0.05, param_text, wrap=True, horizontalalignment='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

        # Save each visualization separately
        output_file = os.path.join(output_folder, f'trajectories_visualization_{line_idx + 1}.png')
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    plot_trajectories('trajectories_policies.jsonl')
