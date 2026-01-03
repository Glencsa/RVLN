import numpy as np
import torch


def print_trainable_parameters(model):
    """打印可训练参数统计"""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}"
    )


def compute_metrics(eval_pred):
    # 注意：这里的 predictions 已经是 argmax 后的结果了
    predictions, labels = eval_pred
    
    # 如果 labels 里有 -100，需要 mask 掉
    mask = labels != -100
    
    # 计算准确率
    acc = (predictions[mask] == labels[mask]).mean()
    
    return {"accuracy": acc}

def preprocess_logits_for_metrics(logits, labels):
    """
    在计算指标前，先把 Logits 变成预测的 Token ID。
    这能节省 99.9% 的显存占用。
    """
    if isinstance(logits, tuple):
        # 某些模型会返回 (logits, past_key_values)，我们只要 logits
        logits = logits[0]
    # 直接取 argmax，扔掉原始 logits
    return logits.argmax(dim=-1)