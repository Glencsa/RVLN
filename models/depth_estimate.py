import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import numpy as np
import cv2 

class DepthEstimator:
    def __init__(self, model_id="depth-anything/Depth-Anything-V2-Small-hf", device="cuda"):
        """
        初始化深度估计模型
        Args:
            model_id: Hugging Face 模型 ID
            device: 'cuda' or 'cpu'
        """
        print(f"Loading Depth Anything model: {model_id}...")
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def predict_depth(self, image: Image.Image, return_type="pil", colormap=cv2.COLORMAP_INFERNO):
        """
        输入 RGB 图像，输出估计的深度图
        Args:
            image: PIL.Image 对象 (RGB)
            return_type: 
                - 'pil': 返回可视化的彩色深度图 (PIL Image)
                - 'tensor': 返回原始深度数值 (torch.Tensor)
            colormap: cv2 的颜色映射模式 (例如 cv2.COLORMAP_INFERNO, cv2.COLORMAP_JET)
        
        Returns:
            PIL.Image (彩色) 或 torch.Tensor
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        original_size = image.size[::-1] # (H, W)
        
        prediction = F.interpolate(
            predicted_depth.unsqueeze(1),
            size=original_size,
            mode="bicubic",
            align_corners=False,
        )

        if return_type == "tensor":
            depth_min = prediction.min()
            depth_max = prediction.max()
            normalized_depth = (prediction - depth_min) / (depth_max - depth_min)
            return normalized_depth # [1, 1, H, W]

        elif return_type == "pil":
            depth_numpy = prediction.squeeze().cpu().numpy()
            depth_min = depth_numpy.min()
            depth_max = depth_numpy.max()
            if depth_max - depth_min > 1e-6:
                depth_normalized = (depth_numpy - depth_min) / (depth_max - depth_min)
            else:
                depth_normalized = depth_numpy
                
            depth_uint8 = (depth_normalized * 255).astype(np.uint8)
            depth_color_bgr = cv2.applyColorMap(depth_uint8, colormap)
            depth_color_rgb = cv2.cvtColor(depth_color_bgr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(depth_color_rgb)

if __name__ == "__main__":
    import os
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    estimator = DepthEstimator(model_id="./Depth-Anything-V2-Small-hf", device=device)
    img_path = "step_3.jpg" 
    
    if not os.path.exists(img_path):
        print(f"warning: {img_path}  not exist")
        image = Image.fromarray(np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8))
    else:
        image = Image.open(img_path).convert("RGB")
    
    print("正在推理...")

    # 4. 获取并保存彩色深度图 (使用 JET 彩虹色)
    depth_image_jet = estimator.predict_depth(image, return_type="pil", colormap=cv2.COLORMAP_JET)
    depth_image_jet.save("depth_result_jet.png")
    print("saved successfully: depth_result_jet.png")