import json
import os
from PIL import Image
from tqdm import tqdm

# ================= 配置区域 =================
INPUT_JSON_PATH = '/home/isvl/r2r_16_images_6_act.json'      # 原始 JSON 路径
OUTPUT_JSON_PATH = './dataset_instructblip.json' # 输出 JSON 路径
PADDING_IMAGE_PATH = './black.jpg' # 填充用的黑图路径

# 【新增】设置 r2r_training_rgb 所在的新的根目录
# 例如：如果你的图片实际在 /data/datasets/r2r_training_rgb/...
# 这里就填 '/data/datasets/'
NEW_IMAGE_ROOT = '/media/isvl/Elements/'  # <--- 请在这里修改你的新路径前缀
# ===========================================

def create_black_padding_image(path):
    """如果填充用的黑图不存在，则创建它"""
    if not os.path.exists(path):
        print(f"正在创建填充用全黑图片: {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        img = Image.new('RGB', (224, 224), color=(0, 0, 0))
        img.save(path)

def update_image_path(path):
    """
    【新增】修改路径逻辑：
    找到 'r2r_training_rgb' 关键字，保留它及之后的部分，
    替换它之前的所有路径为 NEW_IMAGE_ROOT。
    """
    keyword = 'r2r_training_rgb'
    
    if keyword in path:
        # 找到 keyword 开始的索引
        index = path.find(keyword)
        # 截取从 r2r_training_rgb 开始的相对路径
        # 例如: "old/path/r2r_training_rgb/ep1/img.jpg" -> "r2r_training_rgb/ep1/img.jpg"
        relative_path = path[index:]
        
        # 拼接新根目录
        return os.path.join(NEW_IMAGE_ROOT, relative_path)
    else:
        # 如果路径里没有这个关键字，保持原样或者按需处理
        return path

def process_images(image_list, padding_path):
    """将任意长度的图片列表转换为固定 5 张: [H, H, H, H, C]"""
    if not image_list:
        return None 
        
    current_img = image_list[-1]
    raw_history = image_list[:-1]
    
    # 目标历史帧数量
    target_hist_len = 4
    
    final_history = []
    
    if len(raw_history) >= target_hist_len:
        # 截断：取最后4帧
        final_history = raw_history[-target_hist_len:]
    else:
        # 填充：前面补黑图
        num_padding = target_hist_len - len(raw_history)
        final_history = [padding_path] * num_padding + raw_history
        
    return final_history + [current_img]

def process_human_text(text):
    """处理 Human 输入：保留指令，替换任务模板"""
    new_suffix = (
        "You are provided with:\n"
        "- Historical observations(four images): <history> \n"
        "- Current observation: <current>, there are 3 routes on the current observation.\n\n"
        "Your task is to select the best route number based on these routes, or return zero to Stop. \n"
        " The format of the result is {'Route': number 0~3}"
    )

    split_marker = "You are provided with:"
    
    if split_marker in text:
        prefix = text.split(split_marker)[0]
        return prefix + new_suffix
    else:
        return new_suffix

def process_gpt_response(text):
    """
    处理 GPT 输出：根据首字母映射到 Route 格式
    """
    if not text:
        return text

    clean_text = text.strip()
    first_char = clean_text[0].upper()

    mapping = {
        'A': 2,
        'B': 1,
        'C': 3,
        'D': 0
    }

    if first_char in mapping:
        route_num = mapping[first_char]
        return f"{{'Route': {route_num}}}"
    else:
        print(f"Warning: 无法映射的回答开头 '{first_char}'，原文本: {clean_text[:20]}...")
        return text

def main():
    create_black_padding_image(PADDING_IMAGE_PATH)
    
    if not os.path.exists(INPUT_JSON_PATH):
        print(f"错误：找不到输入文件 {INPUT_JSON_PATH}")
        return

    print(f"正在读取 {INPUT_JSON_PATH} ...")
    with open(INPUT_JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed_data = []
    skipped_count = 0

    print("开始批处理...")
    for idx, item in tqdm(enumerate(data), total=len(data)):
        
        # --- 1. 处理图片路径 & 填充逻辑 ---
        raw_images = item.get('images', [])
        
        # 【新增步骤】先批量更新路径前缀
        # 这里的 update_image_path 会把 r2r_training_rgb 前面的路径全换掉
        raw_images = [update_image_path(img) for img in raw_images]
        
        # 然后再进行截断/填充处理
        new_images = process_images(raw_images, PADDING_IMAGE_PATH)
        
        if new_images is None:
            skipped_count += 1
            continue

        # --- 2. 处理对话 (Human 和 GPT) ---
        new_conversations = []
        for turn in item['conversations']:
            new_turn = turn.copy()
            
            if turn['from'] == 'human':
                new_turn['value'] = process_human_text(turn['value'])
            elif turn['from'] == 'gpt':
                new_turn['value'] = process_gpt_response(turn['value'])
            
            new_conversations.append(new_turn)

        # --- 3. 构建新条目 ---
        new_item = {
            "id": item.get('id', f"identity_{idx}"),
            "images": new_images,
            "conversations": new_conversations
        }
        processed_data.append(new_item)

    # 保存结果
    print(f"正在保存结果到 {OUTPUT_JSON_PATH} ...")
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)

    print(f"处理完成！")
    print(f"成功转换: {len(processed_data)} 条数据")
    print(f"跳过无效数据: {skipped_count}")

if __name__ == "__main__":
    main()