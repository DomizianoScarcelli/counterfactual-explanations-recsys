import pandas as pd

df = pd.read_csv("results/raw/model_sensitivity_category.csv")

# Columns to transform (assuming the column containing sequences is named 'sequence')
column_to_transform = 'sequence'

# Apply transformation to the column
df[column_to_transform] = df[column_to_transform].str.replace(r'\[|\]', '', regex=True)
df[column_to_transform] = df[column_to_transform].str.replace(" ", "", regex=False)

# Save the modified DataFrame back to a file (if needed)
output_file_path = "output_file.csv"
df.to_csv(output_file_path, index=False)
