import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from macros import *

sns.set()


if __name__ == '__main__':
    filename = "results_classification_baseline_selection"
    filepath = os.path.join(os.getcwd(), "results", filename + ".parquet")
    final_df = pd.read_parquet(filepath)
    avg_df = final_df.copy()
    avg_df[DATASET] = "Average"
    final_df = pd.concat([avg_df, final_df], ignore_index=True)
    final_df = final_df.loc[final_df[DATASET] != "HTTP"]
    final_df = final_df.loc[final_df[DATASET] != "CreditCard"]

    agg_df = (final_df.drop(REPETITION, axis=1)
              .groupby([DATASET, APPROACH, METRIC])[SCORE]
              .aggregate(["mean", "std"])
              .round(2)
              .reset_index())

    agg_df["Mean (Std)"] = agg_df["mean"].astype(str) + " (" + agg_df["std"].astype(str) + ")"
    agg_df = agg_df.drop(["mean", "std"], axis=1)
    print(agg_df.to_markdown())

    g = sns.catplot(kind="bar", data=final_df, x=DATASET, y=SCORE, hue=APPROACH,
                    errwidth=1, linewidth=.5)
    sns.move_legend(g, "upper center", ncol=3, title="")
    # plt.yscale("log")
    plt.xlabel("")
    plt.ylabel(final_df[METRIC].iloc[0])
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout(pad=.5)
    plt.subplots_adjust(top=.83)
    plt.ylim(bottom=0.3)
    plt.savefig(os.path.join(os.getcwd(), "figures", filename + ".pdf"))