import os
import torch
from transformers import InstructBlipProcessor, InstructBlipConfig
from peft import PeftModel
from models.WayPointVLN import RvlnMultiTask  # 确保能导入你的自定义模型类

def merge_lora():
    # ================= 配置路径 =================
    base_model_path = "./instructblip-vicuna-7b"
    adapter_path = "./output_116/final_adapter" 
    stage1_weights_path = "output/stage1_checkpoint/latest_checkpoint.pth"
    depth_encoder_path = "./vit-base-patch16-224"

    output_path = "./output/rvln_merged_final_116"
    
    print(f"🚀 开始合并流程...")
    print(f" -> Base: {base_model_path}")
    print(f" -> Adapter: {adapter_path}")
    print(f" -> Output: {output_path}")

    # ================= 1. 加载 Processor 和 Config =================
    print("⏳ Loading Processor & Config...")
    processor = InstructBlipProcessor.from_pretrained(base_model_path)    
    tokenizer = processor.tokenizer
    
    special_tokens_dict = {'additional_special_tokens': ["<history>", "<current>"]}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(f" -> Added {num_added_toks} special tokens. Vocab size: {len(tokenizer)}")

    config = InstructBlipConfig.from_pretrained(base_model_path)
    config.history_token_id = tokenizer.convert_tokens_to_ids("<history>")
    config.current_token_id = tokenizer.convert_tokens_to_ids("<current>")
    config.depth_model_name_or_path = depth_encoder_path
    print("⏳ Loading Base Model (RvlnMultiTask)...")
    model = RvlnMultiTask.from_pretrained(
        base_model_path,
        config=config,
        torch_dtype=torch.float16 # 建议用 float16 节省显存
    )
    model.language_model.resize_token_embeddings(len(tokenizer))
    config.vocab_size = len(tokenizer)
    # ================= 3. [关键] 加载 Stage 1 视觉权重 =================
    # 我们希望最终保存的模型包含：[原始ViT] + [训练好的Fusion/Q-Former] + [融合了LoRA的LLM]
    # 所以在合并 LLM 之前，先把视觉部分更新到最新状态
    if os.path.exists(stage1_weights_path):
        print(f"📥 Loading Stage 1 Visual Weights from: {stage1_weights_path}")
        stage1_state_dict = torch.load(stage1_weights_path, map_location="cpu")
        msg = model.load_state_dict(stage1_state_dict, strict=False)
        print(f"   Load Status: {msg}")
    else:
        print("⚠️ Warning: Stage 1 weights not found! The visual part will remain original/random.")

    # ================= 4. 加载 LoRA Adapter =================
    print("⏳ Loading LoRA Adapter...")
    # 你的 LoRA 是加在 language_model 上的
    # 所以我们要把 adapter 挂载到 model.language_model 上
    
    # 关键：PeftModel.from_pretrained 会自动识别 adapter_config.json
    model.language_model = PeftModel.from_pretrained(
        model.language_model,
        adapter_path,
        torch_dtype=torch.float16
    )

    # ================= 5. 执行合并 (Merge & Unload) =================
    print("⚡ Merging LoRA into Base Model...")
    # 这一步会把 LoRA 的矩阵 A*B 加回到原始权重 W 上，并移除 LoRA 层
    model.language_model = model.language_model.merge_and_unload()
    
    # 验证一下：现在 model.language_model 应该变回了原来的 LlamaForCausalLM (或类似)，不再是 PeftModel

    # ================= 6. 保存最终模型 =================
    print(f"💾 Saving Merged Model to: {output_path} ...")
    os.makedirs(output_path, exist_ok=True)
    
    # 保存模型权重 (这会保存整个 RvlnMultiTask，包含视觉部分和融合后的 LLM)
    model.save_pretrained(output_path)
    
    # 保存 Processor / Tokenizer
    processor.save_pretrained(output_path)
    
    # 如果你有自定义 tokenizer 文件在 adapter 目录里，也可以手动复制过去
    # tokenizer.save_pretrained(output_path)

    print("✅ Merge Complete! You can now use the model directly without loading adapters.")

if __name__ == "__main__":
    merge_lora()