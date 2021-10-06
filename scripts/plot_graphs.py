import matplotlib
import pandas as pd
from matplotlib import pyplot as plt

matplotlib.rcParams.update({'font.size': 14})
CSV_FILE = "../data/steps-5s.csv"
STEPS = 5


def get_color(name: str):
    if "+ CD" in name:
        return "b"
    return "r"


def get_style(name: str):
    if "RGBIR" in name:
        return "--"
    return "-"


if __name__ == "__main__":
    df = pd.read_csv(CSV_FILE)
    print(df)
    plt.figure(figsize=(8, 5), dpi=200)
    for row_id, row in df.iterrows():
        name = row["method"]
        values = [row[f"step{i}"] for i in range(STEPS)]
        xx = list(range(len(values)))
        print(name)
        print(values)
        plt.plot(xx, values, marker='o', label=name, c=get_color(name), linestyle=get_style(name))

    plt.xticks(list(range(STEPS)))
    plt.xlabel("Steps")
    plt.ylabel("F1 Score (micro avg.)")
    plt.grid(which="major", axis="both", linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot.png")
