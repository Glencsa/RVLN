"""
WayPointVLN agent implementation used with Habitat environments.
Cleaned up: removed non-English comments, added type hints and clearer docstrings,
and improved formatting for readability.
"""

import os
import re
import sys
import random
import json
from typing import Any, Dict, List, Optional, Tuple

import imageio
import numpy as np
import torch
import cv2
from PIL import Image
from tqdm import trange
from habitat import Env
from habitat.core.agent import Agent
from habitat.utils.visualizations import maps
from transformers import InstructBlipProcessor

from point_project import process_depth_rgb_simple, process_depth_rgb_highlight

# Add project root to Python path
current_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))
sys.path.insert(0, project_root)

from models.WayPointVLN import WayPointVLN
from utils.utils import prepare_inputs_for_generate


def normalize_depth_for_vis(depth: np.ndarray) -> np.ndarray:
    """Normalize a raw depth map to uint8 (0-255) for visualization.

    Args:
        depth: Raw depth image as numpy array.

    Returns:
        depth_vis: uint8 depth image in range [0, 255].
    """
    depth_float = depth.astype(np.float32)
    depth_float = np.nan_to_num(depth_float, nan=0.0, posinf=0.0, neginf=0.0)
    d_min, d_max = float(depth_float.min()), float(depth_float.max())
    if d_max > d_min:
        depth_norm = (depth_float - d_min) / (d_max - d_min)
    else:
        depth_norm = np.zeros_like(depth_float)
    depth_vis = (depth_norm * 255.0).clip(0, 255).astype(np.uint8)
    return depth_vis


class WayPointVLN_Agent(Agent):
    """Agent that uses an WayPointVLN multi-task model to predict navigation routes.

    The agent collects a short history of RGB and depth frames, prepares inputs
    for the model and decodes a predicted route number which is then mapped to
    discrete environment actions.
    """

    def __init__(self, model_path: str, result_path: str, exp_save: str):
        print("Initialize WayPointVLN Agent")

        self.result_path = result_path
        self.require_map = "video" in exp_save
        self.require_data = "data" in exp_save
        self.model_path = model_path

        if self.require_map or self.require_data:
            os.makedirs(self.result_path, exist_ok=True)
        if self.require_data:
            os.makedirs(os.path.join(self.result_path, "log"), exist_ok=True)
        if self.require_map:
            os.makedirs(os.path.join(self.result_path, "video"), exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16

        print(f"Loading WayPointVLN model from: {self.model_path}")
        try:
            self.processor = InstructBlipProcessor.from_pretrained(self.model_path)
            self.tokenizer = self.processor.tokenizer
            self.tokenizer.padding_side = "right"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = WayPointVLN.from_pretrained(self.model_path, torch_dtype=self.dtype).to(self.device)
            if hasattr(self.model, "depth_model"):
                self.model.depth_model.to(dtype=torch.float32)
            self.model.eval()

            print("WayPointVLN Agent Initialization Complete")

        except Exception as e:
            print(f"[ERROR] Failed to load WayPointVLN model: {e}")
            print("Please check model path, config.json, and vision encoder settings")
            raise

        # History buffers
        self.history_rgb_list: List[np.ndarray] = []
        self.history_depth_list: List[np.ndarray] = []
        self.max_history = 5

        # Visualization buffers
        self.rgb_list: List[np.ndarray] = []
        self.depth_list: List[np.ndarray] = []
        self.topdown_map_list: List[np.ndarray] = []

        # State
        self.count_id = 0
        self.pending_action_list: List[int] = []
        self.episode_id: Optional[str] = None

        self.reset()

    def reset(self) -> None:
        """Reset agent state and save previous episode visualizations if required."""
        if self.require_map and self.topdown_map_list:
            output_video_path = os.path.join(self.result_path, "video", f"{self.episode_id}.gif")
            imageio.mimsave(output_video_path, self.topdown_map_list)

        self.history_rgb_list = []
        self.history_depth_list = []
        self.rgb_list = []
        self.depth_list = []
        self.topdown_map_list = []
        self.pending_action_list = []
        self.count_id += 1

    def process_observations(self, rgb: np.ndarray, depth: np.ndarray) -> None:
        """Append a new RGB/depth observation to history buffers.

        Args:
            rgb: RGB observation (H, W, 3) uint8
            depth: Depth observation (H, W) or (H, W, 1)
        """
        self.history_rgb_list.append(rgb)
        self.history_depth_list.append(depth)

        self.rgb_list.append(rgb)
        self.depth_list.append(depth)

        if len(self.history_rgb_list) > self.max_history:
            self.history_rgb_list = self.history_rgb_list[-self.max_history:]
            self.history_depth_list = self.history_depth_list[-self.max_history:]

    def predict_route(self, instruction: str, max_new_tokens: int = 100) -> Tuple[int, str]:
        """Predict a route number given the current history and an instruction.

        Args:
            instruction: Navigation instruction text
            max_new_tokens: Maximum tokens to generate

        Returns:
            (route_number, output_text)
        """
        if not self.history_rgb_list:
            raise RuntimeError("[WayPointVLN Agent] No observations available.")

        inputs = prepare_inputs_for_generate(
            self.history_rgb_list, self.history_depth_list, instruction, self.processor, self.device
        )

        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values=inputs["pixel_values"],
                depth_pixel_values=inputs["depth_pixel_values"],
                qformer_input_ids=inputs["qformer_input_ids"],
                qformer_attention_mask=inputs["qformer_attention_mask"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.0,
            )

        print(f"[WayPointVLN Agent] Model raw output IDs: {outputs}")
        output_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        print(f"[WayPointVLN Agent] Model output: {output_text}")

        route_number = self._parse_route_number(output_text)
        return route_number, output_text

    @staticmethod
    def _parse_route_number(output_text: str) -> int:
        """Parse a route number from model output text.

        Supports patterns like:
        - {'Route': 3}
        - "Route": 3
        - Route: 3
        - plain number: 3
        """
        match = re.search(r"['\"]?Route['\"]?\s*[:=]\s*(-?\d+)", output_text, re.IGNORECASE)
        if match:
            return int(match.group(1))

        match = re.search(r"(-?\d+)", output_text)
        if match:
            return int(match.group(1))

        return -2

    def route_to_actions(self, route_number: int) -> List[int]:
        """Map a route number to a sequence of discrete environment actions.

        Action mapping (example):
            -1: stop
             0: left x4
             1: left, left, forward
             2: left, forward
             3: left, forward x3
             4: forward x4
             5: right, forward x3
             6: right x2, forward x2
             7: right x3, forward
             8: right x4

        For unknown route numbers the method returns a short random action list.
        """
        if route_number == -1:
            return [0]
        elif route_number == 0:
            return [2, 2, 2, 2]
        elif route_number == 1:
            return [2, 2, 1]
        elif route_number == 2:
            return [2, 1]
        elif route_number == 3:
            return [2, 1, 1, 1]
        elif route_number == 4:
            return [1, 1, 1, 1]
        elif route_number == 5:
            return [3, 1, 1, 1]
        elif route_number == 6:
            return [3, 3, 1, 1]
        elif route_number == 7:
            return [3, 3, 3, 1]
        elif route_number == 8:
            return [3, 3, 3, 3]

        return [random.randint(1, 3) for _ in range(random.randint(1, 3))]

    def addtext(self, image: np.ndarray, instruction: str, navigation: str) -> np.ndarray:
        """Draw instruction and navigation text below an image.

        The function expands the image canvas and writes wrapped text lines.
        """
        h, w = image.shape[:2]
        new_height = h + 150
        new_image = np.full((new_height, w, 3), 255, dtype=np.uint8)
        new_image[:h, :w] = image

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        textsize = cv2.getTextSize(instruction, font, font_scale, thickness)[0]
        textY = h + (50 + textsize[1]) // 2

        y_line = textY
        words = instruction.split(" ")
        x = 10
        line = ""

        for word in words:
            test_line = f"{line} {word}".strip()
            test_size, _ = cv2.getTextSize(test_line, font, font_scale, thickness)
            if test_size[0] > image.shape[1] - x:
                cv2.putText(new_image, line, (x, y_line), font, font_scale, (0, 0, 0), thickness)
                line = word
                y_line += textsize[1] + 5
            else:
                line = test_line

        if line:
            cv2.putText(new_image, line, (x, y_line), font, font_scale, (0, 0, 0), thickness)

        y_line += textsize[1] + 10
        cv2.putText(new_image, navigation, (x, y_line), font, font_scale, (0, 0, 0), thickness)

        return new_image

    def act(self, observations: Dict[str, Any], info: Dict[str, Any], episode_id: str) -> Dict[str, int]:
        """Main action selection interface.

        Args:
            observations: Habitat observations containing 'rgb', 'depth', 'instruction'
            info: Environment info, expected to include 'top_down_map_vlnce'
            episode_id: Current episode identifier

        Returns:
            Dictionary with 'action' key and discrete action int value.
        """
        self.episode_id = episode_id

        rgb = observations["rgb"]
        depth = observations.get("depth")
        if depth is None:
            raise ValueError("[WayPointVLN Agent] Missing depth in observations")

        rgb_point, points_3d, point_ids = process_depth_rgb_simple(depth, rgb)

        depth_vis = normalize_depth_for_vis(depth)
        cv2.imwrite("debug_rgb.png", rgb_point)
        cv2.imwrite("debug_depth.png", depth_vis)

        self.process_observations(rgb_point, depth)

        if self.pending_action_list:
            return {"action": self.pending_action_list.pop(0)}

        instruction = observations["instruction"]["text"]
        print(f"[WayPointVLN Agent] Instruction: {instruction}")

        route_number, output_text = self.predict_route(instruction)
        print(f"[WayPointVLN Agent] Predicted route number: {route_number}")

        navigation_text = f"Route: {route_number} | {output_text}"

        if self.require_map:
            top_down_map = maps.colorize_draw_agent_and_fit_to_height(info["top_down_map_vlnce"], rgb.shape[0])
            rgb_highlight = process_depth_rgb_highlight(depth_vis, rgb, selected_ids=[route_number])
            output_im = np.concatenate((rgb_highlight, top_down_map), axis=1)

            img = self.addtext(output_im, instruction, navigation_text)
            self.topdown_map_list.append(img)

        action_list = self.route_to_actions(route_number)
        if not action_list:
            action_list = [random.randint(1, 3)]

        self.pending_action_list.extend(action_list)
        return {"action": self.pending_action_list.pop(0)}