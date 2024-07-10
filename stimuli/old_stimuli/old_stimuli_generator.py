import matplotlib.pyplot as plt
import numpy as np

def draw_grid(grid, path, agent_position, filename='grid_plot.png'):
    # Define the colors and symbols for different tiles
    colors = {'S': 'lightgrey', 'G': 'darkgreen', 'H': 'white', 'C': 'white'}
    symbols = {'S': '', 'G': '', 'H': 'H', 'C': 'C'}
    
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
    
    # Draw the path using arrows
    for (start, end) in zip(path[:-1], path[1:]):
        ax.annotate('', xy=(end[1] + 0.5, n - end[0] - 0.5),
                    xytext=(start[1] + 0.5, n - start[0] - 0.5),
                    arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    
    # Place the agent at the specified position
    ax.text(agent_position[1] + 0.5, n - agent_position[0] - 0.5, 'A', weight='bold',
            ha='center', va='center', color='blue', fontsize=12)

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

# Example grid, path, and agent position
grid = [
    ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'H'],
    ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G'],
    ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G'],
    ['S', 'G', 'G', 'G', 'G', 'S', 'G', 'G', 'G', 'G'],
    ['S', 'G', 'G', 'G', 'S', 'S', 'G', 'G', 'G', 'G'],
    ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'C'],
    ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S']
]
path = [(9, 0), (8, 0), (7, 1), (6, 2), (5, 3), (4, 4), (3, 5), (2, 6), (1, 7), (0, 9)]
agent_position = (0, 9)

# Draw the grid and save as an image
draw_grid(grid, path, agent_position, filename='my_custom_grid.png')
