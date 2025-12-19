import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, InstructBlipProcessor
import sys
from data_utils import InstructBlipLoRADataset, DataCollatorForInstructBlip

def test_pipeline():
    # 1. 配置路径
    model_name = "Salesforce/instructblip-vicuna-7b"
    data_path = "dataset_instructblip.json" # 你生成的 JSON
    image_root = "." # 图片根目录
    
    print(">>> 1. Loading Tokenizer & Processor (Lightweight)...")
    try:
        # 只加载预处理工具，不加载模型权重，速度快且不占显存
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        processor = InstructBlipProcessor.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading huggingface files: {e}")
        return

    # 2. 模拟添加特殊 Token (这是关键步骤)
    print(">>> 2. Adding Special Tokens...")
    special_tokens = {"additional_special_tokens": ["<history>", "<current>"]}
    num_added = tokenizer.add_special_tokens(special_tokens)
    print(f"Added {num_added} special tokens.")
    
    # 验证 ID 是否存在
    hist_id = tokenizer.convert_tokens_to_ids("<history>")
    curr_id = tokenizer.convert_tokens_to_ids("<current>")
    print(f"ID Check -> <history>: {hist_id}, <current>: {curr_id}")

    # 3. 初始化 Dataset
    print(">>> 3. Initializing Dataset...")
    dataset = InstructBlipLoRADataset(
        data_path=data_path,
        processor=processor,
        tokenizer=tokenizer,
        image_root=image_root,
        history_len=4,
        current_len=1,
        query_tokens=32
    )
    print(f"Dataset Length: {len(dataset)}")

    # 取一个样本看看 Dataset 输出是否正常
    sample = dataset[0]
    print("\n--- [Dataset Single Sample Check] ---")
    print(f"Pixel Values Shape: {sample['pixel_values'].shape} (Expect: [5, 3, 224, 224])")
    print(f"LLM Full Text Prefix: {sample['llm_full_text'][:100]}...")
    
    # 检查 Token 扩展是否成功
    # 应该包含大量 <history>
    if sample['llm_full_text'].count("<history>") == 128:
        print("✅ Token Expansion Check: <history> count is 128.")
    else:
        print(f"❌ Token Expansion Failed! Count is {sample['llm_full_text'].count('<history>')}")

    # 4. 初始化 DataLoader & Collator
    print("\n>>> 4. Initializing DataLoader...")
    collator = DataCollatorForInstructBlip(processor, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collator)

    # 5. 获取一个 Batch 进行详细检查
    print(">>> 5. Fetching one batch...")
    try:
        batch = next(iter(dataloader))
    except Exception as e:
        print(f"❌ DataLoader Error: {e}")
        import traceback
        traceback.print_exc()
        return

    # 6. 验证 Tensor 形状
    print("\n--- [Batch Tensor Shape Check] ---")
    print(f"Pixel Values: {batch['pixel_values'].shape} (Expect: [2, 5, 3, 224, 224])")
    print(f"Input IDs:    {batch['input_ids'].shape}    (Expect: [2, Seq_Len])")
    print(f"Labels:       {batch['labels'].shape}       (Expect: [2, Seq_Len])")
    print(f"Q-Former IDs: {batch['qformer_input_ids'].shape}")

    # 7. 验证 Masking 逻辑 (最重要的一步)
    print("\n--- [Label Masking Logic Check] ---")
    # 我们把 input_ids 和 labels 解码回文本，看看对不对
    
    input_ids = batch['input_ids'][0]
    labels = batch['labels'][0]
    
    print("Decoding Input IDs (What the model sees):")
    decoded_input = tokenizer.decode(input_ids, skip_special_tokens=False)
    # 为了避免打印太长，只打印开头和结尾
    print(f"Start: {decoded_input[:200]}...")
    print(f"End:   ...{decoded_input[-200:]}")

    print("\nDecoding Labels (What the model predicts):")
    # Labels 里有很多 -100，直接 decode 会报错，需要过滤
    # 我们把 -100 变成 Pad Token ID 用于展示，或者直接忽略
    valid_labels = labels.clone()
    valid_labels[valid_labels == -100] = tokenizer.pad_token_id
    
    decoded_label = tokenizer.decode(valid_labels, skip_special_tokens=True)
    print(f"Decoded Label Content: '{decoded_label.strip()}'")
    
    # 8. 判定逻辑
    print("\n--- [Final Verdict] ---")
    # 检查 Label 是否只包含回答部分
    # 比如输入是 "USER: ... ASSISTANT: {'Route': 2}"
    # Label 应该只剩下 "{'Route': 2}"
    if "USER:" not in decoded_label and "ASSISTANT:" not in decoded_label and "Route" in decoded_label:
         print("✅ Label Masking looks CORRECT. User prompt is masked.")
    else:
         print("⚠️ Label Masking might be WRONG. Check above outputs.")
         
    if batch['pixel_values'].shape[1] == 5:
        print("✅ Image stacking looks CORRECT (5 frames).")
    else:
        print("❌ Image stacking WRONG.")

if __name__ == "__main__":
    test_pipeline()