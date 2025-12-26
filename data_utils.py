import json
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import PreTrainedTokenizer

class InstructBlipLoRADataset(Dataset):
    def __init__(self, data_path, processor, tokenizer, image_root=".", history_len=4, current_len=1, query_tokens=32):
        print(f"正在加载 JSON文件: {data_path}")
        with open(data_path, "r", encoding='utf-8') as f:
            self.data = json.load(f)
        self.processor = processor
        self.tokenizer = tokenizer
        self.image_root = image_root
        
        self.history_len = history_len
        self.current_len = current_len
        # 总共需要的图片数量 (4 + 1 = 5)
        self.total_len = history_len + current_len
        
        self.hist_token_count = history_len * query_tokens
        self.hist_token = "<history>"
        self.curr_token = "<current>"
        self.curr_token_count = current_len * query_tokens

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # --- 核心修复：列表对齐逻辑 ---
        # 目标：无论 JSON 里有几张图，都要凑成 [H, H, H, H, C] 共 5 张
        
        def normalize_paths(path_list):
            # 1. 如果图片太多，取最后 5 张
            if len(path_list) >= self.total_len:
                return path_list[-self.total_len:]
            # 2. 如果图片不够，前面补 None
            else:
                pad_num = self.total_len - len(path_list)
                return [None] * pad_num + path_list

        rgb_paths_fixed = normalize_paths(item["images"])
        depth_paths_fixed = normalize_paths(item["depth_images"])

        # --- 图片加载逻辑 ---
        def load_image(path, type_name):
            # 如果 path 是 None，说明是填充位，直接返回纯黑图
            if path is None:
                return Image.new('RGB', (224, 224), (0, 0, 0)), True
            
            full_path = os.path.join(self.image_root, path)
            try:
                img = Image.open(full_path).convert("RGB")
                return img, True
            except Exception as e:
                # print(f"❌ [加载失败] {type_name}: {full_path}") # 调试时可打开
                return Image.new('RGB', (224, 224), (0, 0, 0)), False

        # 加载 RGB
        rgb_images = []
        for p in rgb_paths_fixed:
            img, _ = load_image(p, "RGB")
            rgb_images.append(img)
            
        # 加载 Depth
        depth_images = []
        for p in depth_paths_fixed:
            img, _ = load_image(p, "Depth")
            depth_images.append(img)

        # Processor 处理 (现在输入肯定是 5 张图了)
        pixel_values_rgb = self.processor(images=rgb_images, return_tensors="pt").pixel_values
        pixel_values_depth = self.processor(images=depth_images, return_tensors="pt").pixel_values

        # 文本处理 (保持不变)
        human_input = item["conversations"][0]["value"]
        gpt_response = item["conversations"][1]["value"]
        
        expanded_human_input = human_input.replace(
            self.hist_token, self.hist_token * self.hist_token_count
        ).replace(
            self.curr_token, self.curr_token * self.curr_token_count
        )
        
        prompt_text = f"USER: {expanded_human_input} ASSISTANT:"
        full_text = f"{prompt_text} {gpt_response}</s>"
        
        return {
            "pixel_values_rgb": pixel_values_rgb,     # Shape: [5, 3, 224, 224]
            "pixel_values_depth": pixel_values_depth, # Shape: [5, 3, 224, 224]
            "qformer_prompt": human_input,
            "llm_prompt": prompt_text,
            "llm_full_text": full_text
        }
    

class DataCollatorForInstructBlip:
    def __init__(self, processor, tokenizer, qformer_tokenizer):
        self.processor = processor
        self.tokenizer = tokenizer
        self.qformer_tokenizer = qformer_tokenizer
        
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.ignore_index = -100

    def __call__(self, batch):
        # 1. 处理图像 (Stack RGB and Depth separately)
        # 结果维度: [Batch_Size, 5, 3, 224, 224]
        pixel_values_rgb = torch.stack([item["pixel_values_rgb"] for item in batch])
        pixel_values_depth = torch.stack([item["pixel_values_depth"] for item in batch])
        
        # 2. 处理 Q-Former 文本输入
        qformer_prompts = [item["qformer_prompt"] for item in batch]
        qformer_inputs = self.qformer_tokenizer(
            qformer_prompts, 
            padding=True, 
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        # 3. 处理 LLM 文本输入
        llm_full_texts = [item["llm_full_text"] for item in batch]
        llm_inputs = self.tokenizer(
            llm_full_texts,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            add_special_tokens=False
        )
        
        input_ids = llm_inputs.input_ids
        attention_mask = llm_inputs.attention_mask
        labels = input_ids.clone()
        
        # 4. Mask User Prompt (计算 Loss 时忽略 Instruction 部分)
        for i, item in enumerate(batch):
            prompt = item["llm_prompt"]
            # 注意：add_special_tokens=False 很重要，避免重复添加 BOS
            prompt_ids = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                add_special_tokens=False
            ).input_ids[0]
            
            prompt_len = len(prompt_ids)
            
            # Mask 掉 prompt 部分
            labels[i, :prompt_len] = self.ignore_index
            
            # Mask 掉 padding 部分
            padding_mask = input_ids[i] == self.pad_token_id
            labels[i, padding_mask] = self.ignore_index

        return {
            "pixel_values_rgb": pixel_values_rgb,       # [B, 5, 3, 224, 224]
            "pixel_values_depth": pixel_values_depth,   # [B, 5, 3, 224, 224]
            "qformer_input_ids": qformer_inputs.input_ids,
            "qformer_attention_mask": qformer_inputs.attention_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }