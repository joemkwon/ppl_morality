import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def create_plot(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot points and lines for each goal
    for goal in df['Goal'].unique():
        goal_data = df[df['Goal'] == goal].sort_values('Steps on grass')
        plt.plot('Steps on grass', 'moral judgment', data=goal_data, marker='o', linestyle='-', label=goal)

    # Customize the plot
    plt.title('Moral Judgment vs Steps on Grass by Goal')
    plt.xlabel('Steps on Grass')
    plt.ylabel('Moral Judgment')

    # Set y-axis limit to 100
    plt.ylim(0, 100)

    # Adjust legend
    plt.legend(title='Goal', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust layout and save
    plt.tight_layout()
    output_file = csv_file.replace('.csv', '_plot.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_file}")

def process_directory(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            csv_file = os.path.join(directory_path, filename)
            create_plot(csv_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create plots from CSV data in a directory.')
    parser.add_argument('--directory', type=str, help='Path to the directory containing CSV files')
    args = parser.parse_args()

    process_directory(args.directory)