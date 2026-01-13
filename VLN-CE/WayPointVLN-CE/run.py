#!/usr/bin/env python3

import argparse
from habitat.datasets import make_dataset
from VLN_CE.vlnce_baselines.config.default import get_config

from habitat import Env
from tqdm import trange
import json
import os

import numpy as np

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    
    parser.add_argument(
        "--exp-save",
        type=str,
        required=True,
        help="results types requried to be saved",
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="names of evaluation model",
    )
    
    parser.add_argument(
        "--split-num",
        type=int,
        required=True,
        help="chunks of evluation"
    )
    
    parser.add_argument(
        "--split-id",
        type=int,
        required=True,
        help="chunks ID of evluation"

    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="location of model weights"

    )

    parser.add_argument(
        "--result-path",
        type=str,
        required=True,
        help="location to save results"

    )

    args = parser.parse_args()
    run_exp(**vars(args))


def run_exp(exp_config: str, split_num: str, split_id: str, model_path: str, result_path: str, model_name: str, exp_save: str, opts=None) -> None:
    """Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.
    """

    config = get_config(exp_config, opts)
            
    dataset = make_dataset(id_dataset=config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET)
    dataset.episodes.sort(key=lambda ep: ep.episode_id)
    np.random.seed(42)
    
    
    dataset_split = dataset.get_splits(split_num)[split_id]
    
    evaluate_agent(config, split_id, dataset_split, model_path, result_path, model_name, exp_save)
  
  
  
  
  
  
def evaluate_agent(config, split_id, dataset, model_path, result_path, model_name, exp_save) -> None:
 
    env = Env(config.TASK_CONFIG, dataset)

    if model_name == "navid":
        from agent_navid import NaVid_Agent
        agent = NaVid_Agent(model_path, result_path, exp_save)
        
    elif model_name == "uni-navid":
        from agent_uninavid import UniNaVid_Agent
        agent = UniNaVid_Agent(model_path, result_path, exp_save)
    
    if model_name == "rvln":
        from agent_rvln import RVLN_Agent
        agent = RVLN_Agent(model_path, result_path, exp_save)


    num_episodes = len(env.episodes)
    
    EARLY_STOP_ROTATION = config.EVAL.EARLY_STOP_ROTATION
    EARLY_STOP_STEPS = config.EVAL.EARLY_STOP_STEPS

    
    target_key = {"distance_to_goal", "success", "spl", "path_length", "oracle_success"}

    count = 0
    
      
    for _ in trange(num_episodes, desc=config.EVAL.IDENTIFICATION+"-{}".format(split_id)):
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
                continuse_rotation_count=0
            else :
                continuse_rotation_count +=1 
        
            if not action_queue:
                action_queue = agent.act(obs, info, env.current_episode.episode_id)
                if not isinstance(action_queue, list):
                    action_queue = [action_queue]
            
            action = action_queue.pop(0)
            if continuse_rotation_count > EARLY_STOP_ROTATION or iter_step>EARLY_STOP_STEPS:
                action = {"action": 0}
            iter_step+=1
            obs = env.step(action)
            
        info = env.get_metrics()
        result_dict = dict()
        result_dict = {k: info[k] for k in target_key if k in info}
        result_dict["id"] = env.current_episode.episode_id
        count+=1


        if "data" in exp_save:
            with open(os.path.join(os.path.join(result_path, "log"),"stats_{}.json".format(env.current_episode.episode_id)), "w") as f:
                json.dump(result_dict, f, indent=4)



if __name__ == "__main__":
    main()
