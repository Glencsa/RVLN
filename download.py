import os
# 强制使用国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id='zzywhu/VLN_Waypoint',
    repo_type='dataset',  # <--- 关键修改：告诉它这是个数据集
    local_dir='./dataset_waypoint',
    ignore_patterns=['*.h5', '*.ot', '*.bin'],
    resume_download=True,
    max_workers=4
)
