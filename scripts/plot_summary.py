#!/usr/bin/env python3

import json
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

        (key, value) = column.split(".")

        (a, b, c) = np.quantile(samples[column], [0.25, 0.5, 0.75])

        results[key].append(
            {
                "index": int(value),
                f"{key}_025": a,
                f"{key}_05": b,
                f"{key}_075": c,
            },
        )

    with open(os.path.join("out", "player_metadata.json"), "r") as file:
        player_metadata = pd.DataFrame(
            [
                {**{"playerId": int(key)}, **value}
                for key, value in json.load(file).items()
                if value.get("index") is not None
            ],
        )

    players = player_metadata.merge(
        pd.DataFrame(results["offense"]),
        on=["index"],
        how="inner",
        validate="1:1",
    ).merge(
        pd.DataFrame(results["defense"]),
        on=["index"],
        how="inner",
        validate="1:1",
    )

    # ---

    n_cols = 4
    n_rows = 3

    (fig, axs) = plt.subplots(n_rows, n_cols, figsize=(18, 10))

    for i, column in enumerate(list(samples.columns)[: n_rows * n_cols]):
        ij = (i // n_cols, i % n_cols)
        axs[*ij].set_title(column)
        axs[*ij].plot(samples[column])

    plt.tight_layout()
    plt.show()
    plt.close()

    # ---

    (fig, ax) = plt.subplots(figsize=(8, 8))

    ax.set_xlabel("defense")
    ax.set_ylabel("offense")

    ax.scatter(players.defense_05, players.offense_05, ec="w")

    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
