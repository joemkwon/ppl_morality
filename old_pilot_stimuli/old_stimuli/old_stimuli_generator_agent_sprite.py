import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def draw_grid(grid, path, agent_image_path, filename='grid_plot.png', show_coords=False):
    # Define the colors and symbols for different tiles
    colors = {'S': 'lightgrey', 'G': 'darkgreen', 'H': 'white', 'C': 'white'}
    symbols = {'S': '', 'G': '', 'H': 'H', 'C': 'C'}
    
    # Determine the final position of the agent from the path
    agent_position = path[0] if path else (0, 0)
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    n = len(grid)
    m = len(grid[0])

    # Draw the grid
    for i in range(n):
        for j in range(m):
            rect = plt.Rectangle((j, n-i-1), 1, 1, facecolor=colors[grid[i][j]])
            ax.add_patch(rect)
            # Add symbols for hospital and coffee shop
            ax.text(j+0.5, n-i-0.5, symbols[grid[i][j]], weight='bold', 
                    ha='center', va='center', color='black')
            # Optionally display the coordinates
            if show_coords:
                ax.text(j+0.1, n-i-0.1, f'({i},{j})', ha='left', va='top', color='red', fontsize=8)

    # Draw the path using arrows
    for (start, end) in zip(path[:-1], path[1:]):
        ax.annotate('', xy=(end[1] + 0.5, n - end[0] - 0.5),
                    xytext=(start[1] + 0.5, n - start[0] - 0.5),
                    arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    
    # Load and place the agent image
    agent_img = plt.imread(agent_image_path)
    imagebox = OffsetImage(agent_img, zoom=0.7)  # Adjust zoom and transparency
    ab = AnnotationBbox(imagebox, (agent_position[1] + 0.5, n - agent_position[0] - 0.5), frameon=False)
    ax.add_artist(ab)

    # Set limits and grid properties
    ax.set_xlim(0, m)
    ax.set_ylim(0, n)
    ax.set_xticks(np.arange(m + 1))
    ax.set_yticks(np.arange(n + 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True)

    # Save the figure
    plt.savefig(filename)
    plt.close()  # Close the plot to free up memory

# Example grid, path, and agent image path
grid = [
    ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S'],
    ['S', 'G', 'G', 'G', 'S', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'G', 'G', 'G', 'S', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'G', 'G', 'G', 'S', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'G', 'G', 'G', 'S', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'G', 'G', 'G', 'S', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'G', 'G', 'G', 'S', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'G', 'G', 'G', 'S', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'G', 'G', 'G', 'S', 'G', 'G', 'G', 'G', 'S'],
    ['C', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'H']
]

path = [(0,0), (0,1), (0,2), (0,3), (0,4), (1,4), (2,4), (3,4), (4,4), (5,4), (6,4), (7,4), (8,4), (9,4), (9,5), (9,6), (9,7), (9,8), (9,9)]

agent_image_path = '/Users/joekwon/Desktop/yale/overcookedresearch/moral_lines/ToM-gameplaying-POMDP/overcooked_server/new_assets/Characters/1/idle/D.png'

# Draw the grid and save as an image with coordinates shown
draw_grid(grid, path, agent_image_path, filename='instruction.png', show_coords=False)
