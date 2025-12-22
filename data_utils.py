import json
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import PreTrainedTokenizer

class InstructBlipLoRADataset(Dataset):
    def __init__(
        self, 
        data_path, 
        processor, 
        tokenizer: PreTrainedTokenizer,
        image_root=".", # 图片路径的前缀，如果是相对路径填 "."
        history_len=4,
        current_len=1,
        query_tokens=32 # Q-Former 每个 query 的 token 数，默认是 32
    ):
        self.data = json.load(open(data_path, "r"))
        self.processor = processor
        self.tokenizer = tokenizer
        self.image_root = image_root
        
        # 计算需要重复的次数
        self.hist_token_count = history_len * query_tokens # 4 * 32 = 128
        self.curr_token_count = current_len * query_tokens # 1 * 32 = 32
        
        # 获取特殊 Token 的 字符串形式
        # 假设你已经在外部 add_special_tokens 了
        self.hist_token = "<history>"
        self.curr_token = "<current>"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. 加载 5 张图片
        # 确保图片按照 [H, H, H, H, C] 的顺序
        image_paths = item["images"]
        images = []
        for img_path in image_paths:
            full_path = os.path.join(self.image_root, img_path)
            image = Image.open(full_path).convert("RGB")
            images.append(image)
            
        # 2. 处理文本
        # 我们需要分别提取 Human 的指令和 GPT 的回答
        human_input = item["conversations"][0]["value"]
        gpt_response = item["conversations"][1]["value"]
        
        # --- 核心逻辑：Token 扩展 ---
        # 将单个 <history> 替换为 128 个 <history> 字符串连接
        # 将单个 <current> 替换为 32 个 <current> 字符串连接
        # 这样 Tokenizer 处理后就会产生对应数量的 ID
        
        expanded_human_input = human_input.replace(
            self.hist_token, self.hist_token * self.hist_token_count
        ).replace(
            self.curr_token, self.curr_token * self.curr_token_count
        )
        
        # 3. 构造 LLM 的完整 Prompt (Vicuna 格式)
        # Vicuna v1.5 template: "USER: <prompt> ASSISTANT: <answer></s>"
        # 注意：这里我们手动拼接，方便控制 loss mask
        prompt_text = f"USER: {expanded_human_input} ASSISTANT:"
        full_text = f"{prompt_text} {gpt_response}</s>"
        
        # 4. 使用 Processor 处理图像
        # InstructBlipProcessor 会返回 pixel_values
        # 注意：Processor 默认只能处理单张图或 Batch，我们需要手动堆叠
        # 这里我们分别处理每一张图，然后 stack
        # 或者直接传 list of images 给 processor (如果支持)
        
        # 官方 processor 接受 images=List[Image] 会返回 [B, C, H, W]
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs.pixel_values # [5, 3, 224, 224]
        
        # 5. 返回原始数据给 DataCollator 处理 (Tokenization 在 Collator 做更灵活)
        return {
            "pixel_values": pixel_values, # Tensor
            "qformer_prompt": human_input, # 原始指令给 Q-Former (不需要扩展 tag)
            "llm_prompt": prompt_text,     # 用于计算 Mask 长度
            "llm_full_text": full_text     # 完整文本用于 Tokenize
        }

class DataCollatorForInstructBlip:
    def __init__(self, processor, tokenizer, qformer_tokenizer): # <--- 新增参数 qformer_tokenizer
        self.processor = processor
        self.tokenizer = tokenizer
        self.qformer_tokenizer = qformer_tokenizer # <--- 保存它
        
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.ignore_index = -100

    def __call__(self, batch):
        # 1. 处理图像
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        
        # 2. 处理 Q-Former 输入
        # 【核心修复】：必须使用 qformer_tokenizer !!!
        qformer_prompts = [item["qformer_prompt"] for item in batch]
        qformer_inputs = self.qformer_tokenizer(  # <--- 修改这里，原来是 self.tokenizer
            qformer_prompts, 
            padding=True, 
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        # 3. 处理 LLM 输入 (保持不变，使用 self.tokenizer)
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
        
        # 4. Mask User Prompt (保持不变)
        for i, item in enumerate(batch):
            prompt = item["llm_prompt"]
            prompt_ids = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                add_special_tokens=False
            ).input_ids[0]
            prompt_len = len(prompt_ids)
            labels[i, :prompt_len] = self.ignore_index
            padding_mask = input_ids[i] == self.pad_token_id
            labels[i, padding_mask] = self.ignore_index

        return {
            "pixel_values": pixel_values,
            "qformer_input_ids": qformer_inputs.input_ids,
            "qformer_attention_mask": qformer_inputs.attention_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }