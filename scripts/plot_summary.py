#!/usr/bin/env python3

import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    with open(os.path.join("out", "data.json"), "r") as file:
        data = json.load(file)

    samples = pd.read_csv(os.path.join("out", "samples.csv"), comment="#")

    # ---

    results = {
        "offense": [],
        "defense": [],
    }

    for column in samples.columns:
        if not (column.startswith("offense") or column.startswith("defense")):
            continue

        (key, value) = column.split(".")
        results[key].append({"index": int(value), key: samples[column].mean()})

    # ---

    check = pd.DataFrame(
        {
            key: samples[
                [column for column in samples.columns if column.startswith(f"{key}_shots_check")]
            ].sum(axis=1)
            for key in ["home", "away"]
        },
    )

    # ---

    for key in ["home", "away"]:
        data[f"{key}_shots_check"] = [
            samples[column].mean()
            for column in samples.columns
            if column.startswith(f"{key}_shots_check")
        ]

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

    columns = ["firstName", "lastName"]
    print(players.sort_values(["offense"], ascending=False)[columns + ["offense"]].head())
    print(players.sort_values(["defense"], ascending=True)[columns + ["defense"]].head())

    # ---

    n_cols = 4
    n_rows = 3

    (fig, axs) = plt.subplots(n_rows, n_cols, figsize=(18, 10))

    for i, column in enumerate(list(samples.columns)[: n_rows * n_cols]):
        ij = (i // n_cols, i % n_cols)
        axs[*ij].set_title(column)
        axs[*ij].plot(samples[column])

    plt.tight_layout()
    plt.savefig(os.path.join("out", "summary.png"))
    plt.close()

    # ---

    (fig, axs) = plt.subplots(1, 3, figsize=(16, 8))

    axs[0].scatter(players.defense, players.offense, ec="w")
    axs[0].set_xlabel("defense")
    axs[0].set_ylabel("offense")

    for i, key in enumerate(["home", "away"]):
        sns.histplot(check[key], discrete=True, kde=True, ec="w", ax=axs[i + 1])
        axs[i + 1].axvline(sum(data[f"{key}_shots"]), color="tomato")
        axs[i + 1].set_ylabel("shots")

    plt.tight_layout()
    plt.savefig(os.path.join("out", "samples.png"))
    plt.close()


if __name__ == "__main__":
    main()
