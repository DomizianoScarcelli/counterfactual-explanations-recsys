import fire
import matplotlib.pyplot as plt
import pandas as pd


def main(csv_path: str, save_path: str, title: str):
    df = pd.read_csv(csv_path)
    # Define score columns
    score_columns = ["score@1", "score@5", "score@10", "score@20"]

    # Group by 'pos_from_end' and calculate the mean of the scores
    mean_scores = df.groupby("pos_from_end")[score_columns].mean()

    # Plot the results
    plt.figure(figsize=(10, 6))
    for score in score_columns:
        plt.plot(mean_scores.index, mean_scores[score], marker="o", label=score)

    plt.xlabel("Position from End")
    plt.ylabel("Mean Score")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(save_path)

if __name__ == "__main__":
    fire.Fire(main)
