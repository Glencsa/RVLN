import os
import torch
import numpy as np
from PIL import Image
from transformers import InstructBlipProcessor
try:
    from models.rvln import RvlnMultiTask
except ImportError:
    raise ImportError("请确保 models/rvln.py 存在，并且其中定义了 RvlnMultiTask 类。")

# --- 关键参数 (必须与 Dataset __init__ 保持一致) ---
HISTORY_LEN = 4
CURRENT_LEN = 1
TOTAL_LEN = HISTORY_LEN + CURRENT_LEN  # 5
QUERY_TOKENS = 32                      # Q-Former输出特征数
CHECKPOINT_PATH = "output/rvln_merged_final"  
stage1_checkpoint = "output/stage1_checkpoint/latest_checkpoint.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16  # 3090/4090/A100 必选 bf16

def load_model():
    print(f"Loading model from: {CHECKPOINT_PATH}")
    processor = InstructBlipProcessor.from_pretrained(CHECKPOINT_PATH)
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    hist_id = tokenizer.convert_tokens_to_ids("<history>")
    curr_id = tokenizer.convert_tokens_to_ids("<current>")
    vocab_size = len(tokenizer)
    print(f"   -> Tokenizer IDs: <history>={hist_id}, <current>={curr_id}, Vocab={vocab_size}")
    # Load Model
    model = RvlnMultiTask.from_pretrained(
        CHECKPOINT_PATH,
        torch_dtype=DTYPE,
    ).to(DEVICE)
    model.eval()
    model_emb_size = model.language_model.get_input_embeddings().weight.shape[0]
    # print(f"   -> Model Embedding Size: {model_emb_size}")
    # model.language_model.resize_token_embeddings(len(tokenizer))
    print("Model loaded successfully!")
    return model, processor

def prepare_inputs(rgb_path, depth_path, instruction, processor, device):
    # ================= 1. 构造 5 张图片序列 =================
    def _load_as_rgb(path):
        if path is None:
            return Image.new('RGB', (224, 224), (0, 0, 0))
        try:
            return Image.open(path).convert("RGB")
        except:
            return Image.new('RGB', (224, 224), (0, 0, 0))

    rgb_imgs = [_load_as_rgb(None)] * HISTORY_LEN + [_load_as_rgb(rgb_path)]
    depth_imgs = [_load_as_rgb(None)] * HISTORY_LEN + [_load_as_rgb(depth_path)]
    raw_prompt = f"Imagine you are a robot designed for navigation tasks. Your instruction is {instruction}.\nYou are provided with:\n- Historical observations(four images): <history> \n- Current observation: <current>, there are some routes on the current observation.\n\nYour task is to select the best route number based on these routes, you can also choose the 0 to turn left 30\u00b0 ,choose the 8 to turn right 30\u00b0, or return -1 to Stop. \n The format of the result is {{'Route': number -1~8}}"


    inputs = processor(
        images=rgb_imgs,
        text=[raw_prompt],  
        return_tensors="pt",
        padding="max_length",   # 确保 Q-Former ID 长度一致
        truncation=True
    )
    
    # B. 处理 Depth
    depth_inputs = processor(
        images=depth_imgs,
        return_tensors="pt"
    )
    

    # ================= 4. 处理 Q-Former 输入 =================
    
    # 为了保险，我们取整个 tensor 并增加 batch 维度
    qformer_input_ids = inputs.qformer_input_ids.to(device) # [1, 5, 32]
    qformer_attention_mask = inputs.qformer_attention_mask.to(device) # [1, 5, 32]

    # ================= 5. 文本处理 (LLM Prompt) =================
    hist_token = "<history>"
    curr_token = "<current>"
    
    hist_replacement = hist_token * (HISTORY_LEN * QUERY_TOKENS)
    curr_replacement = curr_token * (CURRENT_LEN * QUERY_TOKENS)
    
    expanded_prompt = raw_prompt.replace(
        hist_token, hist_replacement
    ).replace(
        curr_token, curr_replacement
    )
    
    final_prompt = f"USER: {expanded_prompt} ASSISTANT:"

    text_inputs = processor(
        text=final_prompt,
        return_tensors="pt",
        padding="longest",
        truncation=True
    )

    return {
        "pixel_values": inputs.pixel_values.unsqueeze(0).to(device),
        "depth_pixel_values": depth_inputs.pixel_values.unsqueeze(0).to(device),
        "input_ids": text_inputs.input_ids.to(device),
        "attention_mask": text_inputs.attention_mask.to(device),
        "qformer_input_ids": qformer_input_ids,
        "qformer_attention_mask": qformer_attention_mask
    }


def run_inference(model, processor, rgb_path, depth_path, instruction):
    print(f"\n Image: {rgb_path}")
    
    # 1. 预处理
    inputs = prepare_inputs(rgb_path, depth_path, instruction, processor, model.device)
    # print(inputs["input_ids"])
    input_len = inputs["input_ids"].shape[1]

    print("Generating...")
    with torch.no_grad():
        outputs = model.generate(
            pixel_values=inputs["pixel_values"],
            depth_pixel_values=inputs["depth_pixel_values"],
            qformer_input_ids=inputs["qformer_input_ids"],
            qformer_attention_mask=inputs["qformer_attention_mask"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            # 生成参数
            max_new_tokens=100,
            do_sample=False,       # 95% 准确率直接由 Greedy 决定
            repetition_penalty=1.0 # JSON 格式敏感，不要惩罚重复
        )
        # print("生成出的 Token IDs:", outputs[0][-20:]) 

    # 3. 解码
    output_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    output_text = output_text.strip()
    
    print("-" * 40)
    print(f"Instruction: {instruction}")
    print(f"Prediction:  {output_text}")
    print("-" * 40)

if __name__ == "__main__":
    # 初始化
    model, processor = load_model()
    
    # 测试数据
    test_rgb = "test_data/rgb/step_0_depth_with_points.jpg"
    test_depth = "test_data/depth/step_0_depth.png"
    # 指令
    instruction = 'Go around the right side of the center unit and stop by the right side doorway with the dining table and mirror in it.'
    
    if os.path.exists(test_rgb) and os.path.exists(test_depth):
        run_inference(model, processor, test_rgb, test_depth, instruction)
    else:
        print("找不到测试图片，请检查路径。")