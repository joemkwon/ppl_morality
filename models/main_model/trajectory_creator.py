import matplotlib.pyplot as plt
import numpy as np
import json
import os

# Define the gridworld
gridworld = [
    ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S'],
    ['S', 'G', 'G', 'G', 'S', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'G', 'G', 'G', 'S', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'G', 'G', 'G', 'S', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'G', 'G', 'G', 'S', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'G', 'G', 'G', 'S', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'G', 'G', 'G', 'S', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'G', 'G', 'G', 'S', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'G', 'G', 'G', 'S', 'G', 'G', 'G', 'G', 'S'],
    ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S']
]

# Initialize the trajectory list
trajectory = []

# Define the colors for the grid
colors = {'S': 'white', 'G': 'green'}

# Define the directions
directions = {
    (-1, 0): 'north', (1, 0): 'south', (0, -1): 'west', (0, 1): 'east',
    (-1, -1): 'north-west', (-1, 1): 'north-east', (1, -1): 'south-west', (1, 1): 'south-east'
}

# Function to draw the gridworld
def draw_gridworld(gridworld):
    fig, ax = plt.subplots()
    n = len(gridworld)
    m = len(gridworld[0])
    for i in range(n):
        for j in range(m):
            rect = plt.Rectangle((j, i), 1, 1, facecolor=colors[gridworld[i][j]], edgecolor='black')
            ax.add_patch(rect)
    ax.set_xlim(0, m)
    ax.set_ylim(0, n)
    ax.set_xticks(np.arange(m + 1))
    ax.set_yticks(np.arange(n + 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True)
    return fig, ax

# Function to handle mouse clicks
def on_click(event):
    if event.inaxes:
        x, y = int(event.xdata), int(event.ydata)
        if 0 <= x < len(gridworld[0]) and 0 <= y < len(gridworld):
            if len(trajectory) > 0:
                last_x, last_y = trajectory[-1]['coordinate']
                if (y - last_y, x - last_x) not in directions:
                    print("Invalid move. You can only move to adjacent or diagonal squares.")
                    return
                action = directions[(y - last_y, x - last_x)]
                trajectory[-1]['action'] = action
            square_type = gridworld[y][x]
            trajectory.append({'coordinate': (x, y), 'type': square_type, 'action': None})
            ax.plot(x + 0.5, y + 0.5, 'rx')
            fig.canvas.draw()

def finish_trajectory(event):
    if len(trajectory) > 0:
        trajectory[-1]['type'] = 'F'
        data = {
            'gridworld': gridworld,
            'trajectory': trajectory
        }
        file_name = input("Enter the name for the trajectory file (without extension): ")
        file_path = os.path.join("trajectories", f"{file_name}.json")
        
        # Ensure the directory exists
        os.makedirs("trajectories", exist_ok=True)
        
        print(json.dumps(data, indent=2))
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        plt.close()

# Draw the gridworld
fig, ax = draw_gridworld(gridworld)

# Connect the click event
fig.canvas.mpl_connect('button_press_event', on_click)

# Add a button to finish the trajectory
finish_button_ax = plt.axes([0.81, 0.01, 0.1, 0.075])
finish_button = plt.Button(finish_button_ax, 'Finish')
finish_button.on_clicked(finish_trajectory)

plt.show()