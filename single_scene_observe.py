#!/usr/bin/env python3

# 绝对路径配置（请按需修改为你的实际绝对路径）
BASE_DIR = "/home/yang/VLN/navid_ws/NaVid-VLN-CE"
EXP_CONFIG = f"{BASE_DIR}/VLN_CE/vlnce_baselines/config/r2r_baselines/cma.yaml"  # 示例，占位
RESULT_PATH = f"{BASE_DIR}/results"
EPISODE_ID = "6296"  # 优先使用 episode 过滤，若不想用则置为 None
SCENE_ID = None       # 若 EPISODE_ID 为 None 时按 scene 过滤
ACTION = 2           # 单步动作，默认 MOVE_FORWARD=1；可改为 0:STOP, 2:TURN_LEFT, 3:TURN_RIGHT

import os
import json
import numpy as np
from typing import Optional

from habitat.datasets import make_dataset
from habitat import Env
from VLN_CE.vlnce_baselines.config.default import get_config

# 确保结果根目录存在（启动即创建）
os.makedirs(RESULT_PATH, exist_ok=True)
print(f"[INFO] Result root ensured: {os.path.abspath(RESULT_PATH)}")

def _get_instruction_text(ep) -> Optional[str]:
    candidates = [
        getattr(ep, "instruction", None),
        getattr(ep, "goal", None),
        getattr(ep, "instruction_text", None),
    ]
    for c in candidates:
        if c is None:
            continue
        if isinstance(c, str) and c.strip():
            return c
        if isinstance(c, dict):
            for key in ("text", "instruction", "command"):
                if key in c and isinstance(c[key], str) and c[key].strip():
                    return c[key]
    return None


def step(env, action):
    """执行一步动作并更新状态。
    参数:
      env: Habitat 环境实例
      action: 整数或字典动作；若为整数，将封装为 {"action": action}
    返回:
      obs: 新的观测字典
      info: 当前指标（env.get_metrics()）
      done: 是否结束（env.episode_over）
      episode_id: 当前 episode id
    """
    # 统一为 habitat 的动作字典格式
    act = action if isinstance(action, dict) else {"action": int(action)}
    obs = env.step(act)
    info = env.get_metrics()
    done = env.episode_over
    episode_id = env.current_episode.episode_id
    return obs, info, done, episode_id


def save_observation(env, obs, result_path, step_idx=0):
    """将当前观测保存为 npy 与可视化 PNG，并记录元信息。
    参数:
      env: Habitat 环境实例（用于获取 episode/scene 信息）
      obs: 当前观测字典（包含 rgb/depth 等）
      result_path: 输出根目录
      step_idx: 当前保存步骤索引（用于创建形如 scene_id_step0 的子目录）
    返回:
      out_meta: 保存的元信息字典
    """
    # 提取并标准化通道
    rgb = obs.get("rgb", None)
    if rgb is None:
        rgb = obs.get("rgb_sensor", None)
    depth = obs.get("depth", None)
    if depth is None:
        depth = obs.get("depth_sensor", None)
    if rgb is not None:
        rgb = np.asarray(rgb)
    if depth is not None:
        depth = np.asarray(depth)
    

    # 以 results/<scene_name>/{image,depth}/step_{n}.(png,npy) 组织
    raw_scene_id = getattr(env.current_episode, "scene_id", "unknown_scene")
    scene_id_str = str(raw_scene_id)
    scene_basename = os.path.basename(scene_id_str)
    scene_name = os.path.splitext(scene_basename)[0] if ("." in scene_basename) else scene_basename

    scene_root = os.path.join(result_path, scene_name)
    image_dir = os.path.join(scene_root, "image")
    depth_dir = os.path.join(scene_root, "depth")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    print(f"[INFO] Scene root: {os.path.abspath(scene_root)}")

    # 文件路径（按 step_n 命名）
    rgb_npy_path = os.path.join(image_dir, f"step_{int(step_idx)}.npy")
    rgb_png_path = os.path.join(image_dir, f"step_{int(step_idx)}.png")
    depth_npy_path = os.path.join(depth_dir, f"step_{int(step_idx)}.npy")
    depth_png_path = os.path.join(depth_dir, f"step_{int(step_idx)}.png")

    # 同步一个 meta（覆盖写，记录最新 step 信息；如需所有步的 meta，可改为 step_n.json）
    meta_path = os.path.join(scene_root, "meta.json")

    out_meta = {
        "episode_id": env.current_episode.episode_id,
        "scene_id": scene_id_str,
        "step_idx": int(step_idx),
        "instruction": _get_instruction_text(env.current_episode),
        "rgb_shape": None,
        "depth_shape": None,
        "paths": {
            "rgb_npy": rgb_npy_path,
            "rgb_png": rgb_png_path,
            "depth_npy": depth_npy_path,
            "depth_png": depth_png_path,
        },
    }

    # 保存 NPY
    if rgb is not None:
        try:
            np.save(rgb_npy_path, rgb)
            out_meta["rgb_shape"] = list(rgb.shape)
        except Exception as e:
            out_meta["rgb_error"] = str(e)
    if depth is not None:
        try:
            np.save(depth_npy_path, depth)
            out_meta["depth_shape"] = list(depth.shape)
        except Exception as e:
            out_meta["depth_error"] = str(e)

    # 保存 PNG
    try:
        from PIL import Image
        if rgb is not None:
            arr = np.asarray(rgb)
            if arr.dtype != np.uint8:
                if arr.max() <= 1.0:
                    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    arr = arr.clip(0, 255).astype(np.uint8)
            if arr.ndim == 3:
                Image.fromarray(arr[..., :3]).save(rgb_png_path)
            elif arr.ndim == 2:
                Image.fromarray(arr).save(rgb_png_path)
        if depth is not None:
            d = np.asarray(depth).astype(np.float32)
            d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
            if d.ndim == 3:
                if d.shape[2] == 1:
                    d = d[..., 0]
                else:
                    d = d.mean(axis=2)
            d_min, d_max = float(d.min()), float(d.max())
            if d_max > d_min:
                d_norm = (d - d_min) / (d_max - d_min)
            else:
                d_norm = np.zeros_like(d)
            d_vis = (d_norm * 255.0).clip(0, 255).astype(np.uint8)
            Image.fromarray(d_vis).save(depth_png_path)
    except Exception as e:
        out_meta["png_error"] = str(e)

    with open(meta_path, "w") as f:
        json.dump(out_meta, f, ensure_ascii=False, indent=2)

    print(
        "[INFO] Saved step {} to:\n  - {}\n  - {}\n  - {}\n  - {}".format(
            int(step_idx), rgb_npy_path, rgb_png_path, depth_npy_path, depth_png_path
        )
    )
    return out_meta


class InteractiveSession:
    """支持增量式操作的交互会话：初始化、保存当前观测、按动作步进并保存。"""
    def __init__(self, base_dir, exp_config, result_path, episode_id=None, scene_id=None):
        self.base_dir = base_dir
        self.exp_config = exp_config
        self.result_path = result_path
        self.episode_id = episode_id
        self.scene_id = scene_id
        self.env = None
        self.obs = None
        self.step_idx = 0

    def init(self):
        # 切到 VLN_CE 目录，保证相对路径可用
        vlnce_dir = os.path.join(self.base_dir, "VLN_CE")
        os.chdir(vlnce_dir)
        config = get_config(self.exp_config, opts=None)
        dataset = make_dataset(
            id_dataset=config.TASK_CONFIG.DATASET.TYPE,
            config=config.TASK_CONFIG.DATASET,
        )
        dataset.episodes.sort(key=lambda ep: ep.episode_id)
        filtered = dataset.episodes
        if self.episode_id:
            filtered = [ep for ep in filtered if str(ep.episode_id) == str(self.episode_id)]
        elif self.scene_id:
            filtered = [ep for ep in filtered if str(getattr(ep, "scene_id", "")) == str(self.scene_id)]
        if not filtered:
            raise ValueError("No episode found for given filters (episode_id/scene_id).")
        dataset.episodes = [filtered[0]]
        np.random.seed(42)
        self.env = Env(config.TASK_CONFIG, dataset)
        self.obs = self.env.reset()
        self.step_idx = 0
        # 初始化时也确保结果根目录存在
        os.makedirs(self.result_path, exist_ok=True)
        print(f"[INFO] Result root ensured in init: {os.path.abspath(self.result_path)}")
        return self.obs

    def save_current(self):
        if self.env is None or self.obs is None:
            raise RuntimeError("Session not initialized. Call init() first.")
        meta = save_observation(self.env, self.obs, self.result_path, step_idx=self.step_idx)
        print(f"[INFO] Initial observation saved for step {self.step_idx}.")
        return meta

    def step(self, action):
        if self.env is None:
            raise RuntimeError("Session not initialized. Call init() first.")
        self.obs, info, done, episode_id = step(self.env, action)
        self.step_idx += 1
        meta = save_observation(self.env, self.obs, self.result_path, step_idx=self.step_idx)
        print(f"[INFO] Step {self.step_idx} saved.")
        return {
            "meta": meta,
            "info": info,
            "done": done,
            "episode_id": episode_id,
        }

    def close(self):
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
            self.env = None


def run_single_scene_observe(exp_config: str, result_path: str, episode_id: Optional[str], scene_id: Optional[str]) -> None:
    # 增量式：仅初始化并保存当前观测，不自动执行动作；外部可多次调用 session.step(action)
    session = InteractiveSession(BASE_DIR, exp_config, result_path, episode_id, scene_id)
    obs = session.init()
    out_meta_init = session.save_current()
    print(json.dumps({"init": out_meta_init}, ensure_ascii=False, indent=2))

    session.step(ACTION)
    out_meta_step = session.save_current()
    print(json.dumps({"step": out_meta_step}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run_single_scene_observe(EXP_CONFIG, RESULT_PATH, EPISODE_ID, SCENE_ID)
