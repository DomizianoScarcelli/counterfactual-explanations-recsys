import pandas as pd
import fire


# Read the data from the item file
def main(dataset: str, categorized: bool):
    item_df = pd.read_csv(
        f"dataset/{dataset}/{dataset}.item",
        sep="\t",
        names=["item_id", "movie_title", "release_year", "class"],
    )
    inter_df = pd.read_csv(
        f"dataset/{dataset}/{dataset}.inter",
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
    )

    if categorized:
        df_exploded = item_df["class"].str.split(expand=True).stack()
    else:
        df_exploded = inter_df["item_id"].str.split(expand=True).stack()

    class_frequencies = df_exploded.value_counts()
    if categorized:
        class_frequencies = class_frequencies.sort(on="item_id")
    print("Class Frequencies:")
    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None, "display.width", None
    ):
        print(class_frequencies)
    if not categorized:
        return

    class_percentages = class_frequencies / len(item_df) * 100
    print("\nClass Percentages:")
    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None, "display.width", None
    ):
        print(class_percentages)

    merged_df = inter_df.merge(item_df[["item_id", "class"]], on="item_id")
    merged_df_exploded = merged_df["class"].str.split(expand=True).stack()

    interaction_class_frequencies = merged_df_exploded.value_counts()
    print("Interaction-Level Class Frequencies:")
    print(interaction_class_frequencies)

    interaction_class_percentages = (
        interaction_class_frequencies / len(merged_df_exploded) * 100
    )
    print("\nInteraction-Level Class Percentages:")
    print(interaction_class_percentages)


if __name__ == "__main__":
    fire.Fire(main)
