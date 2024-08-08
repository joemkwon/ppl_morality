import pandas as pd

# Load the CSV files
df1 = pd.read_csv('/Users/joekwon/Desktop/neurosymbolic_rule_breaking/results/experiment_2a/experiment-2a-norule-full.csv')
df2 = pd.read_csv('/Users/joekwon/Desktop/neurosymbolic_rule_breaking/results/experiment_2a/experiment-2a-norule.csv')

# Combine the CSV files
combined_df = pd.concat([df1, df2])

# Save the combined dataframe to a new CSV file
combined_df.to_csv('/Users/joekwon/Desktop/neurosymbolic_rule_breaking/results/experiment_2a/experiment-2a-norule-combined.csv', index=False)