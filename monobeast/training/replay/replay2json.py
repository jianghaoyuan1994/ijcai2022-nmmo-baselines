from ijcai2022nmmo import CompetitionConfig, scripted, submission, RollOut

import argparse
import json
import pickle
import shutil
from pathlib import Path

import lz4.block
from tqdm import tqdm

# generate replay

# config = CompetitionConfig()
# config.SAVE_REPLAY = "XXX"
# my_team = submission.get_team_from_submission(
#     submission_path="../../my-submission/",
#     team_id="MyTeam",
#     env_config=config,
# )
# # Or initialize the team directly
# # my_team = MyTeam("Myteam", config, ...)
#
# teams = []
# teams.extend([scripted.CombatTeam(f"Combat-{i}", config) for i in range(3)])
# teams.extend([scripted.ForageTeam(f"Forage-{i}", config) for i in range(5)])
# teams.extend([scripted.RandomTeam(f"Random-{i}", config) for i in range(7)])
# teams.append(my_team)
#
# ro = RollOut(config, teams, parallel=True, show_progress=True)
# ro.run(n_episode=1,render=True)


def replace(data: dict, dictionary: dict):
    """submission_id --> team_name"""
    if dictionary is None:
        return data
    for packet in data["packets"]:
        if "player" not in packet:
            continue
        for p in packet["player"].values():
            if "base" in p and "name" in p["base"]:
                name: str = p["base"]["name"]
                subm_id, eid = name.split("_")
                p["base"]["name"] = dictionary[subm_id] + "_" + eid
    return data


def replay2json(filename: str):
    dictionary = None
    if filename.endswith(".zip"):
        extract_dir = filename.replace(".zip", "")
        shutil.unpack_archive(filename, extract_dir=extract_dir)
        replays = []
        for x in Path(extract_dir).iterdir():
            if x.suffix == ".replay":
                replays.append(x)
            elif x.name == "submissions.json":
                with open(x, "r") as fp:
                    dictionary = json.load(fp)
            else:
                pass
    else:
        assert filename.endswith(".replay")
        replays = [Path(filename)]

    for rep in tqdm(replays):
        with open(rep, "rb") as fp:
            data = fp.read()
        data = pickle.loads(lz4.block.decompress(data))
        data = replace(data, dictionary)
        jsonfile = str(rep.resolve()).replace(".replay", ".json")
        with open(jsonfile, "w") as fp:
            json.dump(data, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="Replay file")
    args = parser.parse_args()
    replay2json(args.file)