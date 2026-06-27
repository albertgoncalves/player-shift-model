#!/usr/bin/env python3

import json
import os

import numpy as np
import pandas as pd


def inv_logit(p):
    return 1 / (1 + np.exp(-p))


def main():
    rng = np.random.default_rng(123456789)

    player_mu = 0
    player_std = 0.1

    n_players = 38
    n_shifts = 100

    home_advantage = 1

    players = pd.DataFrame(
        {
            "team": ([0] * (n_players // 2)) + ([1] * (n_players // 2)),
            "offense": rng.normal(player_mu, player_std, n_players),
            "defense": -rng.normal(player_mu, player_std, n_players),
        },
    )

    players.reset_index(drop=False, inplace=True)
    players.rename(columns={"index": "id"}, inplace=True)
    players.id += 1

    players.to_csv(os.path.join("out", "players.csv"), index=False)

    # ---

    data = {
        "n_players": n_players,
        "n_shifts": n_shifts,
        "time": [],
        "home_player_1": [],
        "home_player_2": [],
        "home_player_3": [],
        "home_player_4": [],
        "home_player_5": [],
        "away_player_1": [],
        "away_player_2": [],
        "away_player_3": [],
        "away_player_4": [],
        "away_player_5": [],
        "home_shots": [],
        "away_shots": [],
    }

    times = rng.negative_binomial(10, 0.125, n_shifts)

    for i in range(n_shifts // 2):
        for j, (home_rows, away_rows) in enumerate(
            [
                (players.team == 0, players.team == 1),
                (players.team == 1, players.team == 0),
            ],
        ):
            selected = {
                "home": rng.choice(players.loc[home_rows, "id"], 5, replace=False),
                "away": rng.choice(players.loc[away_rows, "id"], 5, replace=False),
            }

            for key in ["home", "away"]:
                for i, player_id in enumerate(selected[key]):
                    data[f"{key}_player_{i + 1}"].append(int(player_id))

            time = times[(i * 2) + j]

            data["time"].append(int(time))
            data["home_shots"].append(
                int(
                    rng.binomial(
                        time,
                        inv_logit(
                            players.loc[players.id.isin(selected["home"]), "offense"].sum()
                            + players.loc[players.id.isin(selected["away"]), "defense"].sum()
                            + home_advantage,
                        ),
                        1,
                    )[0],
                )
            )
            data["away_shots"].append(
                int(
                    rng.binomial(
                        time,
                        inv_logit(
                            players.loc[players.id.isin(selected["away"]), "offense"].sum()
                            + players.loc[players.id.isin(selected["home"]), "defense"].sum(),
                        ),
                        1,
                    )[0],
                )
            )

    for key in data.keys():
        if key in ["n_players", "n_shifts"]:
            continue
        assert n_shifts == len(data[key]), key

    with open(os.path.join("out", "data.json"), "w") as file:
        json.dump(data, file)


if __name__ == "__main__":
    main()
