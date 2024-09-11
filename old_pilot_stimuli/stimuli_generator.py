import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def draw_grid(grid, path, agent_image_path, sprite_info, filename='grid_plot.png', show_coords=False):
    # Define the colors for different tiles
    colors = {'S': 'lightgrey', 'G': 'darkgreen', 'H': 'white', 'C': 'white', 'B': 'white'}
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

    # Draw the path using arrows
    for (start, end) in zip(path[:-1], path[1:]):
        ax.annotate('', xy=(end[0] + 0.5, n - end[1] - 0.5),
                    xytext=(start[0] + 0.5, n - start[1] - 0.5),
                    arrowprops=dict(arrowstyle='->', lw=2, color='red'))

    # Load and place the agent image
    agent_img = plt.imread(agent_image_path)
    agent_zoom = 0.7  # You can also make this dynamic if needed
    imagebox = OffsetImage(agent_img, zoom=agent_zoom)
    ab = AnnotationBbox(imagebox, (agent_position[0] + 0.5, n - agent_position[1] - 0.5), frameon=False)
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
    plt.close()



# Sprite paths and zoom factors for H, C, and B types
sprite_info = {
    'H': {'path': '/Users/joekwon/Desktop/neurosymbolic_rule_breaking/stimuli/sprites/hospital.png', 'zoom': 0.1},
    'C': {'path': '/Users/joekwon/Desktop/neurosymbolic_rule_breaking/stimuli/sprites/coffee.png', 'zoom': 0.05},
    'B': {'path': '/Users/joekwon/Desktop/neurosymbolic_rule_breaking/stimuli/sprites/bus.png', 'zoom': 0.1}
}

# Example grid, path, and agent image path
grid = [
    ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'C'],
    ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'G', 'S', 'S'],
    ['S', 'G', 'G', 'G', 'G', 'G', 'G', 'S', 'S', 'S'],
    ['S', 'G', 'G', 'G', 'G', 'G', 'S', 'S', 'G', 'S'],
    ['S', 'G', 'G', 'G', 'G', 'S', 'S', 'G', 'G', 'S'],
    ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S'],
    ['S', 'G', 'G', 'S', 'S', 'G', 'H', 'G', 'G', 'S'],
    ['S', 'G', 'S', 'S', 'G', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'S', 'S', 'G', 'G', 'G', 'G', 'G', 'G', 'B'],
    ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S']
]


path = [
    (0,9), (1,9), (2,8),(3,7),(4,6),(5,5),(6,6)
    ]

agent_image_path = '/Users/joekwon/Desktop/yale/overcookedresearch/moral_lines/ToM-gameplaying-POMDP/overcooked_server/new_assets/Characters/1/idle/D.png'

# Draw the grid and save as an image with coordinates shown
draw_grid(grid, path, agent_image_path, sprite_info, filename='map3_hospital_sidewalk_shortest.png', show_coords=False)
