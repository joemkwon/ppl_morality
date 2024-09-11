import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def draw_grid(grid, sprite_info, filename='grid_plot.png', show_coords=False):
    # Define the colors for different tiles
    colors = {'S': 'lightgrey', 'G': 'darkgreen', 'H': 'white', 'C': 'white', 'B': 'white'}
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    n = len(grid)
    m = len(grid[0])

    # Draw the grid
    for i in range(n):
        for j in range(m):
            rect = plt.Rectangle((j, n-i-1), 1, 1, facecolor=colors[grid[i][j]])
            ax.add_patch(rect)
            # Load and add sprite images for specified grid elements
            cell_type = grid[i][j]
            if cell_type in sprite_info and sprite_info[cell_type]['path']:
                img = plt.imread(sprite_info[cell_type]['path'])
                zoom_level = sprite_info[cell_type]['zoom']
                imagebox = OffsetImage(img, zoom=zoom_level)  # Use specific zoom level
                ab = AnnotationBbox(imagebox, (j+0.5, n-i-0.5), frameon=False)
                ax.add_artist(ab)

            # Optionally display the coordinates
            if show_coords:
                ax.text(j+0.5, n-i-0.5, f'({j},{i})', ha='center', va='center', color='black', fontsize=12)

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
    plt.close()

# Sprite paths and zoom factors for H, C, and B types
sprite_info = {
    'H': {'path': '/Users/joekwon/Desktop/neurosymbolic_rule_breaking/stimuli/sprites/hospital.png', 'zoom': 0.1},
    'C': {'path': '/Users/joekwon/Desktop/neurosymbolic_rule_breaking/stimuli/sprites/coffee.png', 'zoom': 0.05},
    'B': {'path': '/Users/joekwon/Desktop/neurosymbolic_rule_breaking/stimuli/sprites/bus.png', 'zoom': 0.1}
}

# Example grid
grid = [
    ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'H'],
    ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'C'],
    ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'B']
]

# Call the function to draw the grid
draw_grid(grid, sprite_info, filename='map1.png', show_coords=False)
