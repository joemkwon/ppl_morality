import matplotlib.pyplot as plt
import json
import os
import numpy as np

# Function to draw the gridworld
def draw_grid(ax, grid):
    colors = {'S': 'white', 'G': 'green', 'F': 'yellow'}
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

# Function to visualize the trajectory on the grid
def plot_trajectory(json_file, output_folder='trajectory_visualizations'):
    # Create the visualizations folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Load JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    gridworld = data['gridworld']
    trajectory = data['trajectory']
    
    # Create a new plot
    fig, ax = plt.subplots(figsize=(10, 10))
    draw_grid(ax, gridworld)

    # Invert the y-axis to make (0,0) at the top-left corner
    ax.invert_yaxis()

    # Plot the trajectory
    x_coords = [step['coordinate'][0] + 0.5 for step in trajectory]
    y_coords = [len(gridworld) - step['coordinate'][1] - 0.5 for step in trajectory]  # Invert y-coordinates

    # Color grading along the trajectory
    num_steps = len(x_coords)
    cmap = plt.get_cmap('coolwarm')
    colors = [cmap(i / num_steps) for i in range(num_steps)]
    
    # Plot each segment of the trajectory with a gradient
    for i in range(num_steps - 1):
        ax.plot(x_coords[i:i+2], y_coords[i:i+2], color=colors[i], linewidth=2, alpha=0.7)
    
    # Add a title
    ax.set_title("Trajectory Visualization", fontsize=14)

    # Save the visualization
    file_name = os.path.splitext(os.path.basename(json_file))[0]
    output_file = os.path.join(output_folder, f'{file_name}_visualization.png')
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    trajectories_folder = 'trajectories'
    for json_file in os.listdir(trajectories_folder):
        if json_file.endswith('.json'):
            plot_trajectory(os.path.join(trajectories_folder, json_file))