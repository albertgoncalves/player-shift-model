#!/usr/bin/env python3

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    samples = pd.read_csv(os.path.join("out", "samples.csv"), comment="#")

    results = {
        "offense": [],
        "defense": [],
    }

    for column in samples.columns:
        if not (column.startswith("offense") or column.startswith("defense")):
            continue

        (key, player_id) = column.split(".")

        values = np.quantile(samples[column], [0.25, 0.5, 0.75])

        results[key].append(
            {
                "id": int(player_id),
                f"model_{key}_025": values[0],
                f"model_{key}_05": values[1],
                f"model_{key}_075": values[2],
            },
        )

    players = pd.read_csv(os.path.join("out", "players.csv"))

    players = players.merge(
        pd.DataFrame(results["offense"]),
        on=["id"],
        how="outer",
        validate="1:1",
    ).merge(
        pd.DataFrame(results["defense"]),
        on=["id"],
        how="outer",
        validate="1:1",
    )

    # ---

    n_cols = 4
    n_rows = 3

    (fig, axs) = plt.subplots(n_rows, n_cols, figsize=(18, 10))

    for i, column in enumerate(list(samples.columns)[: n_rows * n_cols]):
        axs[i // n_cols, i % n_cols].set_title(column)
        axs[i // n_cols, i % n_cols].plot(samples[column])

    plt.tight_layout()
    plt.show()
    plt.close()

    # ---

    line = [
        min(
            players.offense.min(),
            players.defense.min(),
            players.model_offense_025.min(),
            players.model_defense_025.min(),
        ),
        max(
            players.offense.max(),
            players.defense.max(),
            players.model_offense_075.max(),
            players.model_defense_075.max(),
        ),
    ]

    (fig, axs) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(13, 7))

    colors = ["tab:blue", "tab:orange"]

    for i, column in enumerate(["offense", "defense"]):
        axs[i].set_title(column)

        axs[i].plot(line, line, color="k", alpha=0.25)
        for team in players.team.unique():
            subset = players.loc[players.team == team]
            axs[i].scatter(
                subset[column],
                subset[f"model_{column}_05"],
                color=colors[team],
                ec="w",
                label=team,
            )

        for row in players.itertuples():
            axs[i].plot(
                [getattr(row, column), getattr(row, column)],
                [getattr(row, f"model_{column}_025"), getattr(row, f"model_{column}_075")],
                color=colors[row.team],
                alpha=0.5,
            )

        axs[i].legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)

        axs[i].set_aspect("equal")

        axs[i].set_xlabel("actual")

    axs[0].set_ylabel("model")

    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
