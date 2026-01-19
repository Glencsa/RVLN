import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    InstructBlipForConditionalGeneration, 
    InstructBlipConfig, 
    ViTModel,
    ViTConfig
)
from models.fusion_depth import DepthCrossAttentionFusion

class DepthCrossAttentionFusion(nn.Module):
    def __init__(self, rgb_dim, depth_dim, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.depth_proj = nn.Linear(depth_dim, rgb_dim)
        
        self.norm_rgb = nn.LayerNorm(rgb_dim)
        self.norm_depth = nn.LayerNorm(rgb_dim)
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=rgb_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        
        self.norm_ffn = nn.LayerNorm(rgb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(rgb_dim, rgb_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rgb_dim * 4, rgb_dim)
        )
        self.dropout = nn.Dropout(dropout)
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
        
        nn.init.constant_(self.cross_attn.out_proj.weight, 0)
        nn.init.constant_(self.cross_attn.out_proj.bias, 0)

    def forward(self, rgb_embeds, depth_embeds):
        depth_feat = self.depth_proj(depth_embeds)
        depth_feat = self.norm_depth(depth_feat)
        rgb_feat_norm = self.norm_rgb(rgb_embeds)
        attn_output, _ = self.cross_attn(
            query=rgb_feat_norm,
            key=depth_feat,
            value=depth_feat
        )
        rgb_fused = rgb_embeds + self.dropout(attn_output)
        rgb_fused = rgb_fused + self.dropout(self.ffn(self.norm_ffn(rgb_fused)))
        return rgb_fused

    

class RvlnMultiTask(InstructBlipForConditionalGeneration):
    def __init__(self, config: InstructBlipConfig):
        super().__init__(config)
        if not hasattr(config, 'history_token_id') or config.history_token_id is None:
            raise ValueError("Config must contain 'history_token_id'")
        if not hasattr(config, 'current_token_id') or config.current_token_id is None:
            raise ValueError("Config must contain 'current_token_id'")
        depth_model_name = getattr(config, "depth_model_name_or_path", "google/vit-base-patch16-224")
        print(f"Building Depth Backbone structure from: {depth_model_name}")
        depth_config = ViTConfig.from_pretrained(depth_model_name)
        depth_config.add_pooling_layer = False
        self.depth_backbone = ViTModel(depth_config)

        for param in self.depth_backbone.parameters():
            param.requires_grad = False
            
        self.decay_rate = 0.8
        self.rgb_hidden_size = config.vision_config.hidden_size # 1408
        self.depth_hidden_size = self.depth_backbone.config.hidden_size # 768 (ViT-Base)
        self.qformer_hidden_size = config.qformer_config.hidden_size
        self.visual_fusion = DepthCrossAttentionFusion(
            rgb_dim=self.rgb_hidden_size,    
            depth_dim=self.depth_hidden_size, 
            num_heads=8 
        )
        self.itm_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.qformer_hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 2) 
        )
        self._init_weights(self.itm_head)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    def get_fused_visual_features(self, pixel_values, depth_pixel_values, qformer_input_ids, qformer_attention_mask):
        """
        pixel_values:       [B, Num_Images, 3, H, W] (RGB)
        depth_pixel_values: [B, Num_Images, 1(or 3), H, W] (Depth Input)
        """
        b, num_images, c, h, w = pixel_values.shape
        
        flat_pixel_values = pixel_values.view(b * num_images, c, h, w)
        rgb_outputs = self.vision_model(
            pixel_values=flat_pixel_values,
            return_dict=True,
        )
        rgb_embeds = rgb_outputs.last_hidden_state # [B*5, N_patches, 1408]

        flat_depth_values = depth_pixel_values.view(b * num_images, -1, h, w)
        if flat_depth_values.shape[1] == 1:
            flat_depth_values = flat_depth_values.repeat(1, 3, 1, 1)
        
        flat_depth_values = flat_depth_values.to(dtype=self.depth_backbone.dtype)

        # with torch.no_grad():
        #     self.depth_backbone.eval()
        depth_outputs = self.depth_backbone(pixel_values=flat_depth_values, return_dict=True)
        depth_raw = depth_outputs.last_hidden_state 
        
        if depth_raw.dtype != rgb_embeds.dtype:
            depth_raw = depth_raw.to(rgb_embeds.dtype)

        image_embeds = self.visual_fusion(rgb_embeds, depth_raw)
        target_device = image_embeds.device  

        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=target_device)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1).to(target_device)
        
        flat_qformer_input_ids = qformer_input_ids.unsqueeze(1).repeat(1, num_images, 1).view(b * num_images, -1).to(target_device)
        flat_qformer_attention_mask = qformer_attention_mask.unsqueeze(1).repeat(1, num_images, 1).view(b * num_images, -1).to(target_device)

        query_attention_mask = torch.ones(
            (b * num_images, query_tokens.shape[1]),
            dtype=torch.long,
            device=target_device 
        )
        
        qformer_attention_mask_full = torch.cat([query_attention_mask, flat_qformer_attention_mask], dim=1)

        query_outputs = self.qformer(
            input_ids=flat_qformer_input_ids,
            attention_mask=qformer_attention_mask_full,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        
        qformer_output = query_outputs.last_hidden_state

        num_queries = self.query_tokens.shape[1] 
        qformer_output = qformer_output[:, :num_queries, :] 
        qformer_output = self.language_projection(qformer_output)

        qformer_output = qformer_output.view(b, num_images, num_queries, qformer_output.shape[-1])
        
        _, num_frames, _, _ = qformer_output.shape
        decay_factors = torch.tensor([self.decay_rate ** (num_frames - 1 - i) for i in range(num_frames)])
        decay_factors = decay_factors.view(1, num_frames, 1, 1).to(device=target_device, dtype=qformer_output.dtype)

        qformer_output = qformer_output * decay_factors
        history_feats = qformer_output[:, :4, :, :].flatten(1, 2)
        current_feats = qformer_output[:, 4:, :, :].flatten(1, 2)
        
        return history_feats, current_feats

    def _replace_image_tokens(self, inputs_embeds, input_ids, history_feats, current_feats, history_token_id, current_token_id):
        history_feats = history_feats.to(inputs_embeds.dtype)
        current_feats = current_feats.to(inputs_embeds.dtype)
        
        flat_hist_feats = history_feats.flatten(0, 1)
        flat_curr_feats = current_feats.flatten(0, 1)
        
        # History
        history_mask = (input_ids == history_token_id)
        hist_indices = torch.nonzero(history_mask)
        num_hist_fill = min(hist_indices.shape[0], flat_hist_feats.shape[0])
        
        if num_hist_fill > 0:
            target_indices = hist_indices[:num_hist_fill]
            source_feats = flat_hist_feats[:num_hist_fill]
            inputs_embeds[target_indices[:, 0], target_indices[:, 1]] = source_feats

        # Current
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
            depth_pixel_values: torch.FloatTensor,
            qformer_input_ids: torch.LongTensor,
            qformer_attention_mask: torch.LongTensor,
            input_ids: torch.LongTensor,
            attention_mask: torch.LongTensor = None,
            labels: torch.LongTensor = None,
            **kwargs
        ):
            history_feats, current_feats = self.get_fused_visual_features(
                pixel_values, depth_pixel_values, qformer_input_ids, qformer_attention_mask
            )
            
            history_token_id = self.config.history_token_id
            current_token_id = self.config.current_token_id
            
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
            inputs_embeds = self._replace_image_tokens(
                inputs_embeds, input_ids, 
                history_feats, current_feats, 
                history_token_id, current_token_id
            )
            
            if attention_mask is None:
                attention_mask = torch.ones(input_ids.shape, device=input_ids.device)

            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            return outputs

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor,
        depth_pixel_values: torch.FloatTensor, 
        qformer_input_ids: torch.LongTensor,
        qformer_attention_mask: torch.LongTensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor = None,
        **generate_kwargs
    ):
        history_feats, current_feats = self.get_fused_visual_features(
            pixel_values, depth_pixel_values, qformer_input_ids, qformer_attention_mask
        )
        
        history_token_id = self.config.history_token_id
        current_token_id = self.config.current_token_id
        
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        
        inputs_embeds = self._replace_image_tokens(
            inputs_embeds, input_ids, 
            history_feats, current_feats, 
            history_token_id, current_token_id
        )
        
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, device=input_ids.device)
        # print("inputs_embeds:", inputs_embeds)
        return self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs
        )
    
    
    def forward_itm(self, pixel_values, depth_pixel_values, input_ids, attention_mask):
        """
        Image-Text Matching 
        pixel_values:       [B, C, H, W] | [B, N, C, H, W]
        depth_pixel_values: [B, 1(or 3), H, W] | [B, N, 1(or 3), H, W]
        input_ids:          [B, Seq_Len] 
        attention_mask:     [B, Seq_Len]
        """
        c, h, w = pixel_values.shape[-3:]
        
        # [B, C, H, W] -> [B, C, H, W]
        # [B, N, C, H, W] -> [B*N, C, H, W]
        flat_pixel_values = pixel_values.view(-1, c, h, w)
        
        rgb_outputs = self.vision_model(
            pixel_values=flat_pixel_values,
            return_dict=True,
        )
        rgb_embeds = rgb_outputs.last_hidden_state # [B_flat, N_patches, 1408]

        d_c = depth_pixel_values.shape[-3]
        flat_depth_values = depth_pixel_values.view(-1, d_c, h, w)
        
        if flat_depth_values.shape[1] == 1:
            flat_depth_values = flat_depth_values.repeat(1, 3, 1, 1)
        
        flat_depth_values = flat_depth_values.to(dtype=self.depth_backbone.dtype)

        depth_outputs = self.depth_backbone(pixel_values=flat_depth_values, return_dict=True)
        depth_raw = depth_outputs.last_hidden_state # [B_flat, N_patches, 768]
        
        if depth_raw.dtype != rgb_embeds.dtype:
            depth_raw = depth_raw.to(rgb_embeds.dtype)

        image_embeds = self.visual_fusion(rgb_embeds, depth_raw)
        batch_size_total = image_embeds.shape[0]
        query_tokens = self.query_tokens.expand(batch_size_total, -1, -1)
        
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
        num_images_per_sample = batch_size_total // input_ids.shape[0]
        
        if num_images_per_sample > 1:
            # [B, Seq] -> [B, 1, Seq] -> [B, N, Seq] -> [B*N, Seq]
            flat_input_ids = input_ids.unsqueeze(1).repeat(1, num_images_per_sample, 1).view(-1, input_ids.shape[1])
            flat_attention_mask = attention_mask.unsqueeze(1).repeat(1, num_images_per_sample, 1).view(-1, attention_mask.shape[1])
        else:
            flat_input_ids = input_ids
            flat_attention_mask = attention_mask

        query_attention_mask = torch.ones(
            (batch_size_total, query_tokens.shape[1]), 
            dtype=torch.long, 
            device=input_ids.device
        )
        qformer_attention_mask = torch.cat([query_attention_mask, flat_attention_mask], dim=1)
        print("input:", flat_input_ids)
        query_outputs = self.qformer(
            input_ids=flat_input_ids,
            attention_mask=qformer_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        # print("query_outputs:", query_outputs)
        # [B_total, 32 + Seq_Len, 768]
        qformer_features = query_outputs.last_hidden_state
        
        # Use Visual Query [:32]
        num_queries = self.query_tokens.shape[1]
        qformer_features = qformer_features[:, :num_queries, :] # [B_total, 32, 768]
        all_logits = self.itm_head(qformer_features) # [B_total, 32, 2]
        itm_logits = torch.mean(all_logits, dim=1)

        
        if num_images_per_sample > 1:
            itm_logits = itm_logits.view(-1, num_images_per_sample, 2)
            itm_logits = torch.mean(itm_logits, dim=1)
        
        return itm_logits
