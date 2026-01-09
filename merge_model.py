import torch
import os
import glob
import gc
from safetensors.torch import load_file
from peft import LoraConfig, get_peft_model
from models.rvln import RvlnMultiTask
from transformers import InstructBlipConfig, InstructBlipProcessor

# ================= 配置区域 =================
sharded_weights_dir = "lora_weight/rvln_sft_llm" 
base_model_path = "instructblip-vicuna-7b"
# 3. 最终输出路径 (合并后，可以直接 from_pretrained 的目录)
output_dir = "./output/rvln_merged_final_1"

# 4. LoRA 配置 (必须手动重建，因为 adapter_config 丢了)
# 请确保这些参数和你训练时完全一致
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 5. 输出分片大小 (控制输出文件的数量)
# Vicuna-7B 约 13GB。设置 4GB 左右大约会生成 3-4 个文件。
SHARD_SIZE = "4GB" 
# ===========================================

def main():
    print(f"🚀 开始合并任务...")
    print(f"   输入目录: {sharded_weights_dir}")
    print(f"   输出目录: {output_dir}")

    # ---------------------------------------------------------
    # Step 1: 搭建带有 LoRA 插槽的模型骨架
    # ---------------------------------------------------------
    print("\n[1/5] 初始化模型骨架...")
    config = InstructBlipConfig.from_pretrained(base_model_path)
    
    # ⚠️ 如果你有自定义 token (如 <history>), 请确保 config 里已更新
    # 最好是从 sharded_weights_dir 里读取 config.json (如果有的话)
    if os.path.exists(os.path.join(sharded_weights_dir, "config.json")):
        print("   -> 发现新 config.json，使用新配置")
        config = InstructBlipConfig.from_pretrained(sharded_weights_dir)
    
    # 使用 CPU 加载以节省显存 (7B 模型约需 14GB RAM)
    model = RvlnMultiTask.from_pretrained(
        base_model_path,
        config=config,
        torch_dtype=torch.float16,
        device_map="cpu"
    )

    # 挂载 LoRA 空壳 (关键：这会创建 lora_A/lora_B 的层，让权重有地方放)
    print("   -> 挂载 LoRA 结构...")
    model.language_model = get_peft_model(model.language_model, lora_config)


    # ---------------------------------------------------------
    # Step 2: 读取所有分片权重到内存
    # ---------------------------------------------------------
    print("\n[2/5] 读取分片权重...")
    shard_files = sorted(glob.glob(os.path.join(sharded_weights_dir, "*.safetensors")))
    if not shard_files:
        raise FileNotFoundError("未找到 .safetensors 文件")

    full_state_dict = {}
    for i, shard in enumerate(shard_files):
        print(f"   -> Loading shard {i+1}/{len(shard_files)}: {os.path.basename(shard)} ...")
        # 强制加载到 CPU
        shard_weights = load_file(shard, device="cpu")
        full_state_dict.update(shard_weights)
        del shard_weights # 释放内存
        gc.collect()

    print(f"   -> 总计加载 Key 数量: {len(full_state_dict)}")


    # ---------------------------------------------------------
    # Step 3: 权重注入
    # ---------------------------------------------------------
    print("\n[3/5] 将权重注入模型...")
    # strict=False 允许忽略一些无关紧要的 buffer
    missing, unexpected = model.load_state_dict(full_state_dict, strict=False)
    
    # 释放巨大的 state_dict 字典，腾出内存给下一步 Merge
    del full_state_dict
    gc.collect()

    if len(unexpected) > 0:
        print(f"   ⚠️ Warning: Unexpected keys (前3个): {unexpected[:3]}")
    else:
        print("   ✅ 权重加载完美匹配。")


    # ---------------------------------------------------------
    # Step 4: 执行合并 (Merge)
    # ---------------------------------------------------------
    print("\n[4/5] 执行 Merge & Unload...")
    # 这一步将 (Base + LoRA) 永久合并为一个普通的 Linear 矩阵
    model.language_model = model.language_model.merge_and_unload()
    
    # 验证模型现在是否还是 PeftModel
    print(f"   -> 当前模型类型: {type(model.language_model)}")
    # 此时应该已经变回了普通的 LlamaForCausalLM 或类似结构


    # ---------------------------------------------------------
    # Step 5: 重新切分并保存
    # ---------------------------------------------------------
    print(f"\n[5/5] 保存最终模型到 {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    # max_shard_size 会自动帮你切分文件
    model.save_pretrained(
        output_dir, 
        max_shard_size=SHARD_SIZE, 
        safe_serialization=True
    )
    
    # 别忘了保存 Tokenizer/Processor，方便后续直接用
    try:
        print("   -> 复制 Processor/Tokenizer...")
        processor = InstructBlipProcessor.from_pretrained(base_model_path)
        # 如果你之前添加了特殊 token，记得在这里 add_special_tokens 并 save
        # ...
        processor.save_pretrained(output_dir)
    except Exception as e:
        print(f"   ⚠️ Processor 保存失败 (可能需要手动复制): {e}")

    print("\n🎉 全部完成！")
    print(f"现在你可以直接使用: model = RvlnMultiTask.from_pretrained('{output_dir}')")

if __name__ == "__main__":
    main()