#!/usr/bin/env python3

import datetime
import json
import os

import numpy as np
import pandas as pd


def parse_time(string):
    time = datetime.time.strptime(string, "%M:%S")
    return (time.minute * 60) + time.second


def wrangle_shifts(shifts):
    assert 0 < shifts["total"]

    shifts = pd.DataFrame(shifts["data"])

    shifts.loc[shifts.duration.isnull(), "duration"] = "00:00"
    for column in ["startTime", "endTime", "duration"]:
        shifts[column] = shifts[column].map(parse_time)

    assert ((shifts.endTime - shifts.startTime) == shifts.duration).all()

    shifts = shifts.loc[0 < shifts.duration].copy()

    assert (shifts.detailCode == 0).all()
    assert (shifts.typeCode == 517).all()
    assert shifts.gameId.nunique() == 1

    shifts = shifts[["teamId", "playerId", "period", "startTime", "endTime", "duration"]].copy()
    shifts.sort_values(
        ["period", "startTime", "endTime", "teamId", "playerId"],
        ignore_index=True,
        inplace=True,
    )

    splits = []

    for period in shifts.period.unique():
        subset0 = shifts.loc[shifts.period == period]

        edges = np.unique(np.concatenate([subset0.startTime.values, subset0.endTime.values]))
        for start, end in zip(edges[:-1], edges[1:]):
            subset1 = subset0.loc[(subset0.startTime <= start) & (start < subset0.endTime)].copy()
            subset1["startTime"] = start
            subset1["endTime"] = end
            subset1["duration"] = end - start
            splits.append(subset1)

    splits = pd.concat(splits, ignore_index=True)

    combined = (
        shifts.groupby(["playerId"], as_index=False)
        .agg(before=("duration", "sum"))
        .merge(
            splits.groupby(["playerId"], as_index=False).agg(after=("duration", "sum")),
            on=["playerId"],
            how="outer",
            validate="1:1",
        )
    )

    for column in ["before", "after"]:
        assert combined[column].notnull().all()
    assert (combined.before == combined.after).all()

    assert splits[["period", "startTime", "endTime", "playerId"]].duplicated().sum() == 0

    return splits


def wrangle_players(play_by_play):
    players = pd.DataFrame(
        [
            {
                "playerId": player["playerId"],
                "positionCode": player["positionCode"],
                "firstName": player["firstName"]["default"],
                "lastName": player["lastName"]["default"],
            }
            for player in play_by_play["rosterSpots"]
        ],
    )

    assert players.positionCode.isin(["C", "L", "R", "D", "G"]).all()

    return players


def wrangle_shots(play_by_play):
    shots = []
    for event in play_by_play["plays"]:
        if event["typeDescKey"] not in ["shot-on-goal", "blocked-shot", "missed-shot", "goal"]:
            for key in ["shootingPlayerId", "scoringPlayerId"]:
                assert event.get("details", {}).get(key, None) is None, event
            continue

        shots.append(
            {
                "period": event["periodDescriptor"]["number"],
                "timeInPeriod": parse_time(event["timeInPeriod"]),
                "teamId": event["details"]["eventOwnerTeamId"],
                "situationCode": event["situationCode"],
            },
        )
    return pd.DataFrame(shots)


def append_data(home_team_id, shots, splits, data):
    for row in splits[["period", "startTime", "endTime"]].drop_duplicates().itertuples():
        subset_splits = splits.loc[
            (splits.period == row.period)
            & (splits.startTime == row.startTime)
            & (splits.endTime == row.endTime),
        ]

        home_rows = subset_splits.teamId == home_team_id
        assert home_rows.any() and (~home_rows).any()

        goalie_rows = subset_splits.positionCode == "G"
        assert goalie_rows.sum() <= 2

        home_skaters = subset_splits.loc[home_rows & (~goalie_rows), "playerId"].tolist()
        away_skaters = subset_splits.loc[(~home_rows) & (~goalie_rows), "playerId"].tolist()

        home_goalies = (home_rows & goalie_rows).sum()
        away_goalies = ((~home_rows) & goalie_rows).sum()

        if not (
            (len(home_skaters) == 5)
            and (len(away_skaters) == 5)
            and (home_goalies == 1)
            and (away_goalies == 1)
        ):
            continue

        subset_shots = shots.loc[
            (shots.period == row.period)
            & (row.startTime <= shots.timeInPeriod)
            & (shots.timeInPeriod < row.endTime)
            # TODO: Is this necessary?
            & (shots.situationCode == "1551"),
        ]

        home_rows = subset_shots.teamId == home_team_id

        data["durations"].append(row.endTime - row.startTime)
        data["home_players"].append(home_skaters)
        data["away_players"].append(away_skaters)
        data["home_shots"].append(int(home_rows.sum()))
        data["away_shots"].append(int((~home_rows).sum()))


def main():
    game_ids = [
        2025020001,
        2025020050,
        2025021014,
    ]

    data = {
        "durations": [],
        "home_players": [],
        "away_players": [],
        "home_shots": [],
        "away_shots": [],
    }

    player_metadata = {}

    for game_id in game_ids:
        with open(os.path.join("cache", f"play-by-play-{game_id}.json"), "r") as file:
            play_by_play = json.load(file)

        with open(os.path.join("cache", f"shifts-{game_id}.json"), "r") as file:
            shifts = json.load(file)

        players = wrangle_players(play_by_play)

        for row in players.itertuples():
            player_metadata[row.playerId] = {
                "firstName": row.firstName,
                "lastName": row.lastName,
            }

        append_data(
            play_by_play["homeTeam"]["id"],
            wrangle_shots(play_by_play),
            wrangle_shifts(shifts).merge(
                players[["playerId", "positionCode"]],
                on=["playerId"],
                how="inner",
                validate="m:1",
            ),
            data,
        )

    data["n_shifts"] = len(data["durations"])
    for key in ["home_players", "away_players", "home_shots", "away_shots"]:
        assert data["n_shifts"] == len(data[key]), key

    k = 1
    for i in range(data["n_shifts"]):
        for key in ["home", "away"]:
            for j, player_id in enumerate(data[f"{key}_players"][i]):
                index = player_metadata.get(player_id, {}).get("index")
                if index is None:
                    index = k
                    k += 1
                    player_metadata[player_id]["index"] = index
                data[f"{key}_players"][i][j] = index

    data["n_players"] = len(player_metadata)

    with open(os.path.join("out", "player_metadata.json"), "w") as file:
        json.dump(player_metadata, file)

    with open(os.path.join("out", "data.json"), "w") as file:
        json.dump(data, file)


if __name__ == "__main__":
    main()
