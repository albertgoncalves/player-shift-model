#!/usr/bin/env python3

from statsmodels.regression.linear_model import OLS

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def inv_logit(p):
    return 1 / (1 + np.exp(-p))


def main():
    rng = np.random.default_rng(12345678)

    player_mu = 0
    player_std = 0.25

    n_players = 38
    n_shifts = 100

    alpha_std = 1

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

    # ---

    shifts = []
    for i in range(n_shifts):
        for home_rows, away_rows in [
            (players.team == 0, players.team == 1),
            (players.team == 1, players.team == 0),
        ]:
            home_players = rng.choice(players.loc[home_rows, "id"], 5, replace=False)
            away_players = rng.choice(players.loc[away_rows, "id"], 5, replace=False)

            home_player_ids = {f"for_{j}": 0 for j in players.id}
            away_player_ids = {f"against_{j}": 0 for j in players.id}

            for j in home_players:
                home_player_ids[f"for_{j}"] = 1

            for j in away_players:
                away_player_ids[f"against_{j}"] = 1

            shifts.append(
                {
                    **{
                        "alpha": rng.normal(
                            players.loc[players.id.isin(home_players), "offense"].sum()
                            + players.loc[players.id.isin(away_players), "defense"].sum()
                            + home_advantage,
                            alpha_std,
                            1,
                        )[0],
                        "home": 1,
                    },
                    **home_player_ids,
                    **away_player_ids,
                },
            )

            # ---

            home_player_ids = {f"against_{j}": 0 for j in players.id}
            away_player_ids = {f"for_{j}": 0 for j in players.id}

            for j in home_players:
                home_player_ids[f"against_{j}"] = 1

            for j in away_players:
                away_player_ids[f"for_{j}"] = 1

            shifts.append(
                {
                    **{
                        "alpha": rng.normal(
                            players.loc[players.id.isin(away_players), "offense"].sum()
                            + players.loc[players.id.isin(home_players), "defense"].sum(),
                            alpha_std,
                            1,
                        )[0],
                        "home": 0,
                    },
                    **away_player_ids,
                    **home_player_ids,
                },
            )

    shifts = pd.DataFrame(shifts)

    for column in shifts.columns:
        if column == "alpha":
            continue

        assert 0 < shifts[column].sum(), column

    # ---

    model = OLS(
        shifts.alpha,
        shifts[[column for column in shifts.columns if column != "alpha"]],
    ).fit()

    offense_results = []
    defense_results = []

    for key, value in zip(model.params.index, model.params):
        if key == "home":
            print(home_advantage, value)
            continue

        (for_against, player_id) = key.split("_")

        if for_against == "for":
            offense_results.append({"id": int(player_id), "model_offense": value})

        else:
            assert for_against == "against", for_against
            defense_results.append({"id": int(player_id), "model_defense": value})

    players = players.merge(
        pd.DataFrame(offense_results),
        on=["id"],
        how="outer",
        validate="1:1",
    ).merge(pd.DataFrame(defense_results), on=["id"], how="outer", validate="1:1")

    # ---

    line = [
        min(
            players.offense.min(),
            players.defense.min(),
            players.model_offense.min(),
            players.model_defense.min(),
        ),
        max(
            players.offense.max(),
            players.defense.max(),
            players.model_offense.max(),
            players.model_defense.max(),
        ),
    ]

    (fig, axs) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(13, 7))

    for i, column in enumerate(["offense", "defense"]):
        axs[i].set_title(column)

        axs[i].plot(line, line, color="k", alpha=0.25)
        for team in players.team.unique():
            subset = players.loc[players.team == team]
            axs[i].scatter(subset[column], subset[f"model_{column}"], ec="w", label=team)

        axs[i].legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)

        axs[i].set_aspect("equal")

        axs[i].set_xlabel("actual")

    axs[0].set_ylabel("model")

    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
