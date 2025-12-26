import json
import os

# ================= 配置区域 =================
# 1. 设置输入和输出文件名
input_file = 'datasets/rgb_images_r2r_train.json'      # 原始数据
output_file = 'datasets/filtered_traj_3279.json'       # 结果数据

# 目标轨迹 ID
target_traj = "traj_3279"

# 新的路径前缀 (保持你之前的设置)
rgb_prefix_new = "datasets/test/rgb/ep_4991/traj_3279"
depth_prefix_new = "datasets/test/depth/ep_4991/traj_3279"
# ===========================================

def process_data():
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 {input_file}")
        return

    processed_data = []

    for entry in data:
        # 1. 检查是否属于目标轨迹 traj_3279
        if 'images' not in entry:
            continue
            
        is_target_traj = any(target_traj in img_path for img_path in entry['images'])
        
        if is_target_traj:
            new_rgb_list = []
            new_depth_list = []

            # 2. 处理每个图片路径
            for old_path in entry['images']:
                # 提取原始文件名 (例如: step_0.jpg)
                file_name = os.path.basename(old_path)
                
                # 提取文件名核心部分 (例如: step_0)
                # os.path.splitext('step_0.jpg') -> ('step_0', '.jpg')
                name_part, _ = os.path.splitext(file_name)
                
                # --- RGB 处理 ---
                # 要求: 改成 step_0_depth_with_points.jpg
                new_rgb_name = f"{name_part}_depth_with_points.jpg"
                new_rgb_path = os.path.join(rgb_prefix_new, new_rgb_name)
                new_rgb_list.append(new_rgb_path)

                # --- Depth 处理 ---
                # 要求: 改成 step_0_depth.png
                new_depth_name = f"{name_part}_depth.png"
                new_depth_path = os.path.join(depth_prefix_new, new_depth_name)
                new_depth_list.append(new_depth_path)

            # 3. 更新条目数据
            entry['images'] = new_rgb_list
            entry['depth_images'] = new_depth_list

            processed_data.append(entry)

    # 4. 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)

    print(f"✅ 处理完成！")
    print(f"共筛选并修改了 {len(processed_data)} 条数据。")
    print(f"结果已保存至: {output_file}")
    
    # 打印示例以供检查
    if processed_data:
        print("\n--- 示例数据检查 (第一帧) ---")
        print(f"原始文件名核心: {os.path.splitext(os.path.basename(data[0]['images'][0]))[0]}")
        print(f"New RGB:   {processed_data[0]['images'][0]}")
        print(f"New Depth: {processed_data[0]['depth_images'][0]}")

if __name__ == '__main__':
    process_data()