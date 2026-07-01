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

    with open(os.path.join("out", "player_metadata.json"), "r") as file:
        player_metadata = pd.DataFrame(
            [
                {**{"playerId": int(key)}, **value}
                for key, value in json.load(file).items()
                if value.get("index") is not None
            ],
        )

    assert data["n_players"] == len(player_metadata)

    kwargs = {
        "on": ["index"],
        "how": "outer",
        "validate": "1:1",
    }

    results = {}

    for key in ["offense", "defense"]:
        columns = [column for column in samples.columns if column.startswith(key)]
        assert len(columns) == data["n_players"]

        results[key] = []

        for i in range(data["n_players"]):
            column = columns[i]
            assert f"{i + 1}" in column, (i, column)
            results[key].append(
                {
                    "index": i + 1,
                    f"{key}_mu": samples[column].mean(),
                    f"{key}_std": samples[column].std(),
                },
            )

    players = player_metadata.merge(pd.DataFrame(results["offense"]), **kwargs).merge(
        pd.DataFrame(results["defense"]),
        **kwargs,
    )

    for column in players.columns:
        assert players[column].notnull().all(), column

    for key in ["offense", "defense"]:
        for ascending in [False, True]:
            print(key, ascending)
            print(
                players.sort_values(
                    [f"{key}_mu"],
                    ascending=ascending,
                )[
                    ["firstName", "lastName", f"{key}_mu", f"{key}_std"]
                ].head(5),
            )
            print()

    # ---

    train = pd.DataFrame(
        {
            key: samples[
                [column for column in samples.columns if column.startswith(f"{key}_shots_train")]
            ].sum(axis=1)
            for key in ["home", "away"]
        },
    )

    test = pd.DataFrame(
        {
            key: samples[
                [column for column in samples.columns if column.startswith(f"{key}_shots_test")]
            ].sum(axis=1)
            for key in ["home", "away"]
        },
    )

    # ---

    figsize = (19.125, 11.25)

    n_cols = 4
    n_rows = 3

    (fig, axs) = plt.subplots(n_rows, n_cols, figsize=figsize)

    for i, column in enumerate(list(samples.columns)[: n_rows * n_cols]):
        ij = (i // n_cols, i % n_cols)
        axs[*ij].set_title(column)
        axs[*ij].plot(samples[column])

    plt.tight_layout()
    plt.savefig(os.path.join("out", "summary.png"))
    plt.close()

    # ---

    (fig, axs) = plt.subplots(1, 5, figsize=figsize)

    axs[0].scatter(players.defense_mu, players.offense_mu, ec="w")
    axs[0].set_title("players")
    axs[0].set_xlabel("defense_mu")
    axs[0].set_ylabel("offense_mu")

    for i, key in enumerate(["home", "away"]):
        sns.histplot(train[key], discrete=True, kde=True, ec="w", ax=axs[i + 1])
        axs[i + 1].axvline(sum(data[f"{key}_shots"][: data["n_train"]]), color="tomato")

        axs[i + 1].set_title("train")
        axs[i + 1].set_ylabel("shots")

    for i, key in enumerate(["home", "away"]):
        sns.histplot(test[key], discrete=True, kde=True, ec="w", ax=axs[i + 3])
        axs[i + 3].axvline(sum(data[f"{key}_shots"][data["n_train"] :]), color="tomato")

        axs[i + 3].set_title("test")
        axs[i + 3].set_ylabel("shots")

    plt.tight_layout()
    plt.savefig(os.path.join("out", "samples.png"))
    plt.close()


if __name__ == "__main__":
    main()
