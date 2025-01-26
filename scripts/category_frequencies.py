import pandas as pd

# Read the data from the item file
df = pd.read_csv(
    "dataset/ml-100k/ml-100k.item",
    sep="\t",
    names=["item_id", "movie_title", "release_year", "class"],
)

# Split the class column and explode to get individual classes
df_exploded = df["class"].str.split(expand=True).stack()

# Compute and print class frequencies
class_frequencies = df_exploded.value_counts()
print("Class Frequencies:")
print(class_frequencies)

# Optional: Compute class percentages
class_percentages = class_frequencies / len(df) * 100
print("\nClass Percentages:")
print(class_percentages)

# Read item and interaction files
items_df = pd.read_csv(
    "dataset/ml-100k/ml-100k.item",
    sep="\t",
    names=["item_id", "movie_title", "release_year", "class"],
)
inter_df = pd.read_csv(
    "dataset/ml-100k/ml-100k.inter",
    sep="\t",
    names=["user_id", "item_id", "rating", "timestamp"],
)

# Merge interaction data with item classes
merged_df = inter_df.merge(items_df[["item_id", "class"]], on="item_id")

# Split and explode classes
merged_df_exploded = merged_df["class"].str.split(expand=True).stack()

# Compute class frequencies at interaction level
interaction_class_frequencies = merged_df_exploded.value_counts()
print("Interaction-Level Class Frequencies:")
print(interaction_class_frequencies)

# Optional: Compute interaction-level class percentages
interaction_class_percentages = (
    interaction_class_frequencies / len(merged_df_exploded) * 100
)
print("\nInteraction-Level Class Percentages:")
print(interaction_class_percentages)

# Class Frequencies:
# Drama              725
# Comedy             505
# Action             251
# Thriller           251
# Romance            247
# Adventure          135
# Children's         122
# Crime              109
# Sci-Fi             101
# Horror              92
# War                 71
# Mystery             61
# Musical             56
# Documentary         50
# Animation           42
# Western             27
# Film-Noir           24
# Fantasy             22
# unknown              2
# class:token_seq      1
# Name: count, dtype: int64

# Class Percentages:
# Drama              43.077837
# Comedy             30.005942
# Action             14.913844
# Thriller           14.913844
# Romance            14.676173
# Adventure           8.021390
# Children's          7.248960
# Crime               6.476530
# Sci-Fi              6.001188
# Horror              5.466429
# War                 4.218657
# Mystery             3.624480
# Musical             3.327392
# Documentary         2.970885
# Animation           2.495544
# Western             1.604278
# Film-Noir           1.426025
# Fantasy             1.307190
# unknown             0.118835
# class:token_seq     0.059418

# Name: count, dtype: float64
# Interaction-Level Class Frequencies:
# Drama              39895
# Comedy             29832
# Action             25589
# Thriller           21872
# Romance            19461
# Adventure          13753
# Sci-Fi             12730
# War                 9398
# Crime               8055
# Children's          7182
# Horror              5317
# Mystery             5245
# Musical             4954
# Animation           3605
# Western             1854
# Film-Noir           1733
# Fantasy             1352
# Documentary          758
# unknown               10
# class:token_seq        1
# Name: count, dtype: int64

# Interaction-Level Class Percentages:
# Drama              18.765640
# Comedy             14.032249
# Action             12.036445
# Thriller           10.288058
# Romance             9.153982
# Adventure           6.469077
# Sci-Fi              5.987883
# War                 4.420591
# Crime               3.788877
# Children's          3.378239
# Horror              2.500988
# Mystery             2.467121
# Musical             2.330241
# Animation           1.695705
# Western             0.872077
# Film-Noir           0.815161
# Fantasy             0.635948
# Documentary         0.356545
# unknown             0.004704
# class:token_seq     0.000470
# Name: count, dtype: float64
