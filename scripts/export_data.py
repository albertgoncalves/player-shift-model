#!/usr/bin/env python3

# NOTE: See `https://github.com/Zmalski/NHL-API-Reference`.

from zoneinfo import ZoneInfo

import datetime
import json
import logging
import os

import numpy as np
import pandas as pd
import requests

TIMEZONE = ZoneInfo("America/New_York")


def last_modified(path):
    return (
        datetime.datetime.fromtimestamp(os.path.getmtime(path), ZoneInfo("UTC"))
        .replace(tzinfo=ZoneInfo("UTC"))
        .astimezone(TIMEZONE)
    )


def cache(f, path, now):
    if os.path.exists(path) and ((now is None) or (last_modified(path).date() == now.date())):
        with open(path, "r") as file:
            return json.load(file)

    x = f()
    with open(path, "w") as file:
        json.dump(x, file)

    return x


def get_and_cache(url, path, now):
    def f():
        response = requests.get(url, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        return response.json()

    return cache(f, path, now)


def get_game_ids(year):
    season = f"{year - 1}{year}"
    now = datetime.datetime.now().astimezone(TIMEZONE)

    teams = pd.DataFrame.from_dict(
        get_and_cache(
            "https://api.nhle.com/stats/rest/en/team",
            os.path.join("cache", "team.json"),
            now,
        )["data"],
    )
    assert (teams.leagueId == 133).all()
    assert (teams.triCode == teams.rawTricode).all()

    game_ids = set()
    for team_abbrev in teams.triCode:
        schedule = get_and_cache(
            f"https://api-web.nhle.com/v1/club-schedule-season/{team_abbrev}/{season}",
            os.path.join("cache", f"club-schedule-season-{team_abbrev}-{season}.json"),
            now,
        )

        if len(schedule["games"]) == 0:
            continue

        for game in schedule["games"]:
            if (game["gameType"] != 2) or (game["gameState"] != "OFF"):
                continue

            game_ids.add(game["id"])

    return sorted(game_ids)


def parse_time(string):
    time = datetime.time.strptime(string, "%M:%S")
    return (time.minute * 60) + time.second


def wrangle_shifts(shifts):
    assert 0 < shifts["total"]

    shifts = pd.DataFrame(shifts["data"])
    shifts.drop_duplicates(
        ["playerId", "period", "startTime", "endTime"],
        ignore_index=True,
        inplace=True,
    )

    assert shifts[["period", "startTime", "endTime", "playerId"]].duplicated().sum() == 0

    for column in ["startTime", "endTime"]:
        shifts[column] = shifts[column].map(parse_time)
    shifts.duration = shifts.endTime - shifts.startTime

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

    splits.drop_duplicates(
        ["playerId", "period", "startTime", "endTime"],
        ignore_index=True,
        inplace=True,
    )

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

        home_skaters = sorted(subset_splits.loc[home_rows & (~goalie_rows), "playerId"])
        away_skaters = sorted(subset_splits.loc[(~home_rows) & (~goalie_rows), "playerId"])

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

        duration = row.endTime - row.startTime

        home_rows = subset_shots.teamId == home_team_id

        home_shots = int(home_rows.sum())
        away_shots = int((~home_rows).sum())

        assert home_shots <= duration
        assert away_shots <= duration

        data["duration"].append(duration)
        data["home_players"].append(home_skaters)
        data["away_players"].append(away_skaters)
        data["home_shots"].append(home_shots)
        data["away_shots"].append(away_shots)


def main():
    logging.basicConfig()
    logging.getLogger("urllib3").setLevel(logging.DEBUG)

    # ---

    data = {
        "duration": [],
        "home_players": [],
        "away_players": [],
        "home_shots": [],
        "away_shots": [],
    }

    player_metadata = {}

    for game_id in get_game_ids(2026)[:125]:
        play_by_play_path = os.path.join("cache", f"play-by-play-{game_id}.json")
        play_by_play = get_and_cache(
            f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play",
            play_by_play_path,
            None,
        )

        shifts_path = os.path.join("cache", f"shifts-{game_id}.json")
        shifts = get_and_cache(
            f"https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={game_id}",
            shifts_path,
            None,
        )

        if shifts["total"] == 0:
            os.remove(play_by_play_path)
            os.remove(shifts_path)
            continue

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

    # ---

    data = pd.DataFrame(data)
    columns = ["home_players", "away_players"]

    for column in columns:
        data[column] = data[column].map(json.dumps)

    data = data.groupby(columns, as_index=False).agg(
        {column: "sum" for column in ["duration", "home_shots", "away_shots"]},
    )
    data = data.sample(frac=1, replace=False, ignore_index=True, random_state=123456789).copy()

    for column in columns:
        data[column] = data[column].map(json.loads)

    data = data.to_dict(orient="list")

    # ---

    data["n_shifts"] = len(data["duration"])
    assert 0 < data["n_shifts"]

    data["n_train"] = int(data["n_shifts"] * 0.5)

    for key in ["home_players", "away_players", "home_shots", "away_shots"]:
        assert data["n_shifts"] == len(data[key]), key

    k = 1
    for i in range(data["n_shifts"]):
        for key in ["home", "away"]:
            for j, player_id in enumerate(data[f"{key}_players"][i]):
                index = player_metadata[player_id].get("index")
                if index is None:
                    index = k
                    player_metadata[player_id]["index"] = index
                    k += 1
                data[f"{key}_players"][i][j] = index

    data["n_players"] = k - 1
    assert 0 < data["n_players"]

    # ---

    with open(os.path.join("out", "player_metadata.json"), "w") as file:
        json.dump(player_metadata, file)

    with open(os.path.join("out", "data.json"), "w") as file:
        json.dump(data, file)


if __name__ == "__main__":
    main()
