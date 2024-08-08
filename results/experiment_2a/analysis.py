import pandas as pd
import numpy as np

# Load the data
file_path = 'experiment-2a.csv'
data = pd.read_csv(file_path)

# Load the second dataset
file_path_2 = 'experiment-2a-norule.csv'
data_2 = pd.read_csv(file_path_2)

# Filter the rows which contain a value in the stimulus_name column
filtered_data = data.dropna(subset=['stimulus_name'])
filtered_data_2 = data_2.dropna(subset=(['stimulus_name']))

# Ensure 'response' is numeric, converting if necessary
filtered_data.loc[:, 'response'] = pd.to_numeric(filtered_data['response'], errors='coerce')
filtered_data_2.loc[:, 'response'] = pd.to_numeric(filtered_data_2['response'], errors='coerce')

# Group the responses by each stimulus_name
grouped_data = filtered_data.groupby('stimulus_name')['response'].apply(list)
filtered_grouped_data = grouped_data[grouped_data.apply(lambda x: len(x) == 10)].reset_index()

grouped_data_2 = filtered_data_2.groupby('stimulus_name')['response'].apply(list)
filtered_grouped_data_2 = grouped_data_2[grouped_data_2.apply(lambda x: len(x) == 10)].reset_index()

# Calculate mean and standard deviation
filtered_grouped_data['mean'] = filtered_grouped_data['response'].apply(np.mean)
filtered_grouped_data['std_dev'] = filtered_grouped_data['response'].apply(np.std)

filtered_grouped_data_2['mean'] = filtered_grouped_data_2['response'].apply(np.mean)
filtered_grouped_data_2['std_dev'] = filtered_grouped_data_2['response'].apply(np.std)

# Merge the two datasets on stimulus_name
merged_data = pd.merge(filtered_grouped_data, filtered_grouped_data_2, on='stimulus_name', suffixes=('_rule', '_norule'))

# Calculate the difference in mean and standard deviation between the two datasets
merged_data['mean_difference'] = merged_data['mean_rule'] - merged_data['mean_norule']
merged_data['std_dev_difference'] = merged_data['std_dev_rule'] - merged_data['std_dev_norule']

# Save the merged data to a CSV file
output_file_path = 'comparison_data.csv'
merged_data.to_csv(output_file_path, index=False)

output_file_path
