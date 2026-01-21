#!/usr/bin/env python3

import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
from habitat.datasets import make_dataset
from VLN_CE.vlnce_baselines.config.default import get_config
from habitat import Env
from tqdm import trange


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="Path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "--exp-save",
        type=str,
        required=True,
        help="Results types required to be saved",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name of evaluation model",
    )
    parser.add_argument(
        "--split-num",
        type=int,
        required=True,
        help="Number of evaluation chunks",
    )
    parser.add_argument(
        "--split-id",
        type=int,
        required=True,
        help="Chunk ID of evaluation",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Location of model weights",
    )
    parser.add_argument(
        "--result-path",
        type=str,
        required=True,
        help="Location to save results",
    )
    args = parser.parse_args()
    run_exp(**vars(args))


def run_exp(
    exp_config: str,
    split_num: int,
    split_id: int,
    model_path: str,
    result_path: str,
    model_name: str,
    exp_save: str,
    opts: Optional[List[str]] = None,
) -> None:
    """
    Runs experiment given mode and config.
    Args:
        exp_config: Path to config file.
        split_num: Number of splits.
        split_id: Split index.
        model_path: Path to model weights.
        result_path: Directory to save results.
        model_name: Name of the evaluation model.
        exp_save: Result types to save.
        opts: Additional config options.
    """
    config = get_config(exp_config, opts)
    dataset = make_dataset(id_dataset=config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET)
    dataset.episodes.sort(key=lambda ep: ep.episode_id)
    np.random.seed(42)
    dataset_split = dataset.get_splits(split_num)[split_id]
    evaluate_agent(config, split_id, dataset_split, model_path, result_path, model_name, exp_save)


def evaluate_agent(
    config: Any,
    split_id: int,
    dataset: Any,
    model_path: str,
    result_path: str,
    model_name: str,
    exp_save: str,
) -> None:
    """
    Evaluate the agent on the given dataset split.
    Args:
        config: Experiment config.
        split_id: Split index.
        dataset: Dataset split.
        model_path: Path to model weights.
        result_path: Directory to save results.
        model_name: Name of the evaluation model.
        exp_save: Result types to save.
    """
    log_dir = os.path.join(result_path, "log")
    os.makedirs(log_dir, exist_ok=True)
    # Get finished episode IDs
    finished_ids = set()
    existing_files = glob.glob(os.path.join(log_dir, "stats_*.json"))
    for f in existing_files:
        filename = os.path.basename(f)
        ep_id = filename.replace("stats_", "").replace(".json", "")
        finished_ids.add(ep_id)
    # Filter out finished episodes
    original_count = len(dataset.episodes)
    dataset.episodes = [ep for ep in dataset.episodes if str(ep.episode_id) not in finished_ids]
    skipped_count = original_count - len(dataset.episodes)
    if skipped_count > 0:
        print(f"Found existing results. Resuming... Skipped {skipped_count} episodes. Remaining: {len(dataset.episodes)}")
    if len(dataset.episodes) == 0:
        print("All episodes in this split are already finished.")
        return
    env = Env(config.TASK_CONFIG, dataset)
    if model_name == "WayPointVLN":
        from Agent_WayPointVLN import WayPointVLN_Agent

        agent = WayPointVLN_Agent(model_path, result_path, exp_save)
    num_episodes = len(env.episodes)
    EARLY_STOP_ROTATION = config.EVAL.EARLY_STOP_ROTATION
    EARLY_STOP_STEPS = config.EVAL.EARLY_STOP_STEPS
    target_key = {"distance_to_goal", "success", "spl", "path_length", "oracle_success"}
    count = 0
    for _ in trange(num_episodes, desc=f"{config.EVAL.IDENTIFICATION}-{split_id}"):
        obs = env.reset()
        iter_step = 0
        agent.reset()
        continuse_rotation_count = 0
        last_dtg = 999
        action_queue = []
        while not env.episode_over:
            info = env.get_metrics()
            if info["distance_to_goal"] != last_dtg:
                last_dtg = info["distance_to_goal"]
                continuse_rotation_count = 0
            else:
                continuse_rotation_count += 1
            if not action_queue:
                action_queue = agent.act(obs, info, env.current_episode.episode_id)
                if not isinstance(action_queue, list):
                    action_queue = [action_queue]
            action = action_queue.pop(0)
            if continuse_rotation_count > EARLY_STOP_ROTATION or iter_step > EARLY_STOP_STEPS:
                action = {"action": 0}
            iter_step += 1
            obs = env.step(action)
        info = env.get_metrics()
        result_dict = {k: info[k] for k in target_key if k in info}
        result_dict["id"] = env.current_episode.episode_id
        count += 1
        if "data" in exp_save:
            with open(os.path.join(result_path, "log", f"stats_{env.current_episode.episode_id}.json"), "w") as f:
                json.dump(result_dict, f, indent=4)


if __name__ == "__main__":
    main()
