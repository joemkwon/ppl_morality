import pandas as pd
import numpy as np

# Load the data
file_path = 'experiment-2a-norule.csv'
data = pd.read_csv(file_path)

# Filter the rows which contain a value in the stimulus_name column
filtered_data = data.dropna(subset=['stimulus_name'])

# Ensure 'response' is numeric, converting if necessary
filtered_data.loc[:, 'response'] = pd.to_numeric(filtered_data['response'], errors='coerce')

# Group the responses by each stimulus_name and calculate mean and variance
grouped_data = filtered_data.groupby('stimulus_name')['response'].apply(list)
filtered_grouped_data = grouped_data[grouped_data.apply(lambda x: len(x) == 10)].reset_index()

# Calculate mean and variance
filtered_grouped_data['mean'] = filtered_grouped_data['response'].apply(np.mean)
filtered_grouped_data['variance'] = filtered_grouped_data['response'].apply(np.var)

# Save the filtered, grouped, and extended data to a new CSV file
output_file_path = '2a-norule_filtered_grouped_data.csv'
filtered_grouped_data.to_csv(output_file_path, index=False)