import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    InstructBlipForConditionalGeneration, 
    InstructBlipConfig, 
    AutoModelForDepthEstimation
)



class DepthCrossAttentionFusion(nn.Module):
    def __init__(self, rgb_dim, depth_dim, num_heads=8, dropout=0.1):
        super().__init__()
        
        # 1. 为了让 Depth 能和 RGB 计算 Attention，先把 Depth 投影到 RGB 维度
        self.depth_proj = nn.Linear(depth_dim, rgb_dim)
        
        # 2. LayerNorm (Pre-Norm 结构)
        self.norm_rgb = nn.LayerNorm(rgb_dim)
        self.norm_depth = nn.LayerNorm(rgb_dim)
        
        # 3. Cross Attention
        # query = RGB, key/value = Depth
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=rgb_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        
        # 4. Feed Forward Network (FFN)
        self.norm_ffn = nn.LayerNorm(rgb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(rgb_dim, rgb_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rgb_dim * 4, rgb_dim)
        )
        self.dropout = nn.Dropout(dropout)
        self.decay_rate = 0.8
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        
        # 初始化 Trick: 
        # 让 Cross Attention 的输出投影层初始值很小，使得初始阶段 RGB 特征主要保留原始语义
        # 避免初始阶段混乱的 Attention 破坏预训练的 RGB 特征
        nn.init.constant_(self.cross_attn.out_proj.weight, 0)
        nn.init.constant_(self.cross_attn.out_proj.bias, 0)

    def forward(self, rgb_embeds, depth_embeds):
        """
        Input:
            rgb_embeds:   [B, N_rgb, C_rgb]   (Query)
            depth_embeds: [B, N_depth, C_depth] (Key, Value) 
            注意：N_rgb 和 N_depth 可以不相等！Cross-Attention 会处理对齐。
        """
        # 1. 投影并归一化 Depth
        # [B, N_depth, C_depth] -> [B, N_depth, C_rgb]
        depth_feat = self.depth_proj(depth_embeds)
        depth_feat = self.norm_depth(depth_feat)
        
        # 2. 归一化 RGB (作为 Query)
        rgb_feat_norm = self.norm_rgb(rgb_embeds)
        
        # 3. Cross Attention 计算
        # Query=RGB, Key=Depth, Value=Depth
        attn_output, _ = self.cross_attn(
            query=rgb_feat_norm,
            key=depth_feat,
            value=depth_feat
        )
        
        # 4. 残差连接 1 (RGB + Attention)
        rgb_fused = rgb_embeds + self.dropout(attn_output)
        
        # 5. FFN + 残差连接 2
        rgb_fused = rgb_fused + self.dropout(self.ffn(self.norm_ffn(rgb_fused)))
        
        return rgb_fused

    

class InstructBlipMultiTask(InstructBlipForConditionalGeneration):
    def __init__(self, config: InstructBlipConfig):
        super().__init__(config)
        # 检查 Config 中是否包含必要的 token id，如果没有则报错提醒
        if not hasattr(config, 'history_token_id') or config.history_token_id is None:
            raise ValueError("Config must contain 'history_token_id'")
        if not hasattr(config, 'current_token_id') or config.current_token_id is None:
            raise ValueError("Config must contain 'current_token_id'")
        depth_model_name = "./Depth-Anything-V2-Small-hf"
        print(f"Loading Depth Backbone: {depth_model_name}...")
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(depth_model_name)
        self.depth_backbone = self.depth_model.backbone
        
        if hasattr(self.depth_model, "head"):
            del self.depth_model.head
            
        for param in self.depth_backbone.parameters():
            param.requires_grad = False
            
        self.register_buffer("clip_mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer("clip_std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.rgb_hidden_size = config.vision_config.hidden_size 
        self.depth_hidden_size = self.depth_backbone.config.hidden_size
        
        # 注意：InstructBLIP 的 hidden size 较大 (1408)，head 数量选 8 或 16 都可以
        self.visual_fusion = DepthCrossAttentionFusion(
            rgb_dim=self.rgb_hidden_size,    # 1408
            depth_dim=self.depth_hidden_size, # 384
            num_heads=8 # 1408 / 8 = 176 维度每头，整除即可
        )

        self.qformer_hidden_size = config.qformer_config.hidden_size
        self.itm_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.qformer_hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 2) 
        )
        self.decay_rate = 0.8
        self._init_weights(self.itm_head)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        
    def forward_itm(self, pixel_values, input_ids, attention_mask):
            rgb_outputs = self.vision_model(
                pixel_values=pixel_values,
                return_dict=True,
            )
            rgb_embeds = rgb_outputs.last_hidden_state # [B, N, 1408], 类型通常是 bfloat16


            with torch.no_grad():
                self.depth_backbone.eval()
                
                # 这里的转换是为了适配 depth backbone 的输入要求
                images_unnorm = pixel_values * self.clip_std + self.clip_mean
                depth_input = (images_unnorm - self.imagenet_mean) / self.imagenet_std
                
                # 如果 depth backbone 是 float32，这里可能需要把输入转回 float32 防止报错
                # 但通常混合精度下没问题。如果有问题，可以用 .float()
                depth_outputs = self.depth_backbone(depth_input, output_hidden_states=True)
                
                # depth_raw: [B, N_depth, 384]
                depth_raw = depth_outputs.hidden_states[-1] 
                
                if depth_raw.dtype != rgb_embeds.dtype:
                    depth_raw = depth_raw.to(rgb_embeds.dtype)

            # 3. Cross Attention 融合
            # 现在两个输入都是 bfloat16 了
            image_embeds = self.visual_fusion(rgb_embeds, depth_raw)

            # 创建 Visual Mask
            image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

            # 4. Q-Former 交互
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            batch_size = input_ids.shape[0]
            query_attention_mask = torch.ones(
                (batch_size, query_tokens.shape[1]), 
                dtype=torch.long, 
                device=input_ids.device
            )
            qformer_attention_mask = torch.cat([query_attention_mask, attention_mask], dim=1)

            query_outputs = self.qformer(
                input_ids=input_ids,
                attention_mask=qformer_attention_mask,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
                return_dict=True,
            )
            
            qformer_features = query_outputs.last_hidden_state
            pooled_features = torch.mean(qformer_features, dim=1)
            
            head_dtype = self.itm_head[1].weight.dtype 
            pooled_features = pooled_features.to(head_dtype)
            
            itm_logits = self.itm_head(pooled_features)
            
            return itm_logits
    
    def get_fused_visual_features(self, pixel_values, qformer_input_ids, qformer_attention_mask):
        """
        处理图像特征的核心函数：
        输入 pixel_values: [Batch, Num_Images, 3, H, W]
        返回: 拼接后的视觉 Query Tokens [Batch, Num_Images * Num_Queries, Hidden_Dim]
        """
        b, num_images, c, h, w = pixel_values.shape
        
        # 1. 展平 Batch 和 Num_Images 维度，以便并行处理所有图片
        # [B*5, 3, H, W]
        flat_pixel_values = pixel_values.view(b * num_images, c, h, w)
        
        # 2. RGB 特征提取 (Vision Tower)
        rgb_outputs = self.vision_model(
            pixel_values=flat_pixel_values,
            return_dict=True,
        )
        rgb_embeds = rgb_outputs.last_hidden_state # [B*5, N_patches, 1408]

        # 3. Depth 特征提取
        with torch.no_grad():
            self.depth_backbone.eval()
            images_unnorm = flat_pixel_values * self.clip_std + self.clip_mean
            depth_input = (images_unnorm - self.imagenet_mean) / self.imagenet_std
            depth_outputs = self.depth_backbone(depth_input, output_hidden_states=True)
            depth_raw = depth_outputs.hidden_states[-1] # [B*5, N_depth, 384]
            if depth_raw.dtype != rgb_embeds.dtype:
                depth_raw = depth_raw.to(rgb_embeds.dtype)

        # 4. 融合 RGB 和 Depth
        # [B*5, N_patches, 1408]
        image_embeds = self.visual_fusion(rgb_embeds, depth_raw)
        
        # 5. 准备 Q-Former 输入
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
        
        # 扩展 Query Tokens: [1, N_query, Dim] -> [B*5, N_query, Dim]
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        
        # 扩展 Q-Former 的文本输入 (Instruction)
        # 假设 Instruction 对这 5 张图都是一样的，我们需要将其复制 5 倍
        # qformer_input_ids: [B, Seq_Len] -> [B, 1, Seq_Len] -> [B, 5, Seq_Len] -> [B*5, Seq_Len]
        flat_qformer_input_ids = qformer_input_ids.unsqueeze(1).repeat(1, num_images, 1).view(b * num_images, -1)
        flat_qformer_attention_mask = qformer_attention_mask.unsqueeze(1).repeat(1, num_images, 1).view(b * num_images, -1)

        # Q-Former 内部的 Attention Mask 构造
        query_attention_mask = torch.ones(
            (b * num_images, query_tokens.shape[1]),
            dtype=torch.long,
            device=image_embeds.device
        )
        # [B*5, N_query + Seq_Len]
        qformer_attention_mask_full = torch.cat([query_attention_mask, flat_qformer_attention_mask], dim=1)
        
        # 6. Q-Former 前向传播
        query_outputs = self.qformer(
                    input_ids=flat_qformer_input_ids,
                    attention_mask=qformer_attention_mask_full,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_attention_mask,
                    return_dict=True,
                )
        
        # [B*5, Seq_Len_Total, Dim] 
        # Seq_Len_Total = 32 (Query) + Text_Len (Instruction)
        qformer_output = query_outputs.last_hidden_state
        # =======================================================
        # 【核心修复】: 必须只取前 num_query_tokens 个特征！！！
        # 通常是前 32 个。我们用 query_tokens.shape[1] 来动态获取这个值。
        # =======================================================
        num_queries = self.query_tokens.shape[1] # 通常是 32
        qformer_output = qformer_output[:, :num_queries, :] 
        qformer_output = self.language_projection(qformer_output)
        # 此时 qformer_output 形状变为 [B*5, 32, Dim]
        
        # 7. 维度重组与拼接
        # [B*5, 32, Dim] -> [B, 5, 32, Dim]
        qformer_output = qformer_output.view(b, num_images, num_queries, qformer_output.shape[-1])
        
        # ... (后面的 decay 和 split 逻辑保持不变) ...
        _, num_frames, _, _ = qformer_output.shape
        decay_factors = torch.tensor([self.decay_rate ** (num_frames - 1 - i) for i in range(num_frames)])
        decay_factors = decay_factors.view(1, num_frames, 1, 1).to(qformer_output.device)
        qformer_output = qformer_output * decay_factors

        # 这里的切片逻辑也不需要变了，直接用
        history_feats = qformer_output[:, :4, :, :].flatten(1, 2)
        current_feats = qformer_output[:, 4:, :, :].flatten(1, 2)
        
        return history_feats, current_feats
    def _replace_image_tokens(self, inputs_embeds, input_ids, history_feats, current_feats, history_token_id, current_token_id):
        
        # 1. 强制类型转换 (保持上一步的修复)
        history_feats = history_feats.to(inputs_embeds.dtype)
        current_feats = current_feats.to(inputs_embeds.dtype)
        
        # 2. 准备特征
        # [Batch * Total_Hist_Tokens, Dim]
        flat_hist_feats = history_feats.flatten(0, 1)
        flat_curr_feats = current_feats.flatten(0, 1)
        
        # --- 处理 History Token ---
        history_mask = (input_ids == history_token_id) # [B, Seq_Len]
        
        # 获取所有需要填坑的索引位置
        # nonzero() 返回 [N, 2] 的索引矩阵 (dim0=batch_idx, dim1=seq_idx)
        hist_indices = torch.nonzero(history_mask)
        
        # 【核心修复】：安全填入
        # 取 "坑的数量" 和 "萝卜的数量" 的最小值，防止越界
        num_hist_fill = min(hist_indices.shape[0], flat_hist_feats.shape[0])
        
        if num_hist_fill > 0:
            # 只取前 num_hist_fill 个索引进行赋值
            # 这样即使坑比萝卜多，我也只填前几个坑；如果萝卜比坑多，我也只用前几个萝卜
            # 从而完美避免 CUDA 越界
            target_indices = hist_indices[:num_hist_fill]
            source_feats = flat_hist_feats[:num_hist_fill]
            
            # 使用索引赋值代替 masked_scatter_
            inputs_embeds[target_indices[:, 0], target_indices[:, 1]] = source_feats

        # --- 处理 Current Token ---
        current_mask = (input_ids == current_token_id)
        curr_indices = torch.nonzero(current_mask)
        
        num_curr_fill = min(curr_indices.shape[0], flat_curr_feats.shape[0])
        
        if num_curr_fill > 0:
            target_indices = curr_indices[:num_curr_fill]
            source_feats = flat_curr_feats[:num_curr_fill]
            
            inputs_embeds[target_indices[:, 0], target_indices[:, 1]] = source_feats

        return inputs_embeds
    def forward(
            self,
            pixel_values: torch.FloatTensor,
            qformer_input_ids: torch.LongTensor,
            qformer_attention_mask: torch.LongTensor,
            input_ids: torch.LongTensor,
            attention_mask: torch.LongTensor = None,
            labels: torch.LongTensor = None,
            **kwargs
        ):
            # 1. 获取拆分后的视觉特征
            # history_feats: [B, 4*N, D], current_feats: [B, N, D]
            history_feats, current_feats = self.get_fused_visual_features(
                pixel_values, qformer_input_ids, qformer_attention_mask
            )
            history_token_id = self.config.history_token_id
            current_token_id = self.config.current_token_id
            # 2. 获取初始文本 Embeddings
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
            
            # 3. 执行“挖空填词”：将视觉特征填入对应的 token 位置
            inputs_embeds = self._replace_image_tokens(
                inputs_embeds, input_ids, 
                history_feats, current_feats, 
                history_token_id, current_token_id
            )
            
            # 4. Standard Forward
            if attention_mask is None:
                attention_mask = torch.ones(input_ids.shape, device=input_ids.device)

            return self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
                **kwargs
            )

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor,
        qformer_input_ids: torch.LongTensor,
        qformer_attention_mask: torch.LongTensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor = None,
        **generate_kwargs
    ):
        # 1. 获取特征
        history_feats, current_feats = self.get_fused_visual_features(
            pixel_values, qformer_input_ids, qformer_attention_mask
        )
        history_token_id = self.config.history_token_id
        current_token_id = self.config.current_token_id
        # 2. 准备 Embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        
        # 3. 替换特征
        inputs_embeds = self._replace_image_tokens(
            inputs_embeds, input_ids, 
            history_feats, current_feats, 
            history_token_id, current_token_id
        )
        
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, device=input_ids.device)

        return self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs
        )