from __future__ import annotations

import torch

from sam2.modeling.dense_prompt_adapter import (
    DenseBoxPromptAdapter,
    GatedFeatureFusion,
    SignedDistanceBoundaryHead,
    TaskSpecificResidualHeads,
)
from sam2.sam2_video_trainer import SAM2VideoTrainer


class SAM2DensePromptVideoTrainer(SAM2VideoTrainer):
    def __init__(
        self,
        model_cfg,
        sam2_checkpoint,
        device,
        memory_size=7,
        mask_threshold=0.5,
        use_mask_threshold=False,
        fusion_level=1,
        use_dense_prompt=True,
        use_gated_fusion=True,
        use_task_head=True,
        use_boundary_head=True,
    ):
        super().__init__(
            model_cfg=model_cfg,
            sam2_checkpoint=sam2_checkpoint,
            device=device,
            memory_size=memory_size,
            mask_threshold=mask_threshold,
            use_mask_threshold=use_mask_threshold,
        )
        self.fusion_level = fusion_level
        self.use_dense_prompt = use_dense_prompt
        self.use_gated_fusion = use_gated_fusion
        self.use_task_head = use_task_head
        self.use_boundary_head = use_boundary_head

        hidden_dim = self.model.hidden_dim
        self.dense_prompt_adapter = DenseBoxPromptAdapter(
            in_channels=2,
            out_channels=hidden_dim,
        ).to(device=self.device)
        self.gated_fusion = GatedFeatureFusion(channels=hidden_dim).to(
            device=self.device
        )
        self.task_heads = TaskSpecificResidualHeads(
            num_tasks=2,
            hidden_channels=8,
        ).to(device=self.device)
        self.boundary_head = SignedDistanceBoundaryHead(hidden_channels=16).to(
            device=self.device
        )

    def _resolved_fusion_level(self, num_levels: int) -> int:
        level = self.fusion_level
        if level < 0:
            level = num_levels + level
        if not 0 <= level < num_levels:
            raise ValueError(
                f"fusion_level={self.fusion_level} invalid for "
                f"{num_levels} feature levels"
            )
        return level

    def _apply_dense_prompt_fusion(
        self,
        features: dict,
        dense_prompts: torch.Tensor,
        batch_size: int,
        num_frames: int,
    ) -> dict:
        if not self.use_dense_prompt:
            return features
        if dense_prompts.ndim != 5:
            raise ValueError(
                "dense_prompts must have shape (B,T,2,H,W), "
                f"got {tuple(dense_prompts.shape)}"
            )
        expected_shape = (batch_size, num_frames, 2)
        if tuple(dense_prompts.shape[:3]) != expected_shape:
            raise ValueError(
                "dense_prompts must have shape (B,T,2,H,W), "
                f"got {tuple(dense_prompts.shape)} for "
                f"B={batch_size}, T={num_frames}"
            )

        backbone_fpn = features["backbone_fpn"]
        level = self._resolved_fusion_level(len(backbone_fpn))
        image_feature = backbone_fpn[level]
        flat_prompts = dense_prompts.reshape(
            batch_size * num_frames,
            *dense_prompts.shape[2:],
        ).to(device=image_feature.device)
        prompt_feature = self.dense_prompt_adapter(
            flat_prompts,
            target_size=image_feature.shape[-2:],
        )
        if prompt_feature.dtype != image_feature.dtype:
            prompt_feature = prompt_feature.to(dtype=image_feature.dtype)

        if self.use_gated_fusion:
            fused_feature = self.gated_fusion(image_feature, prompt_feature)
        else:
            fused_feature = image_feature + prompt_feature

        fused_features = dict(features)
        fused_backbone_fpn = list(backbone_fpn)
        fused_backbone_fpn[level] = fused_feature
        fused_features["backbone_fpn"] = fused_backbone_fpn
        if level == len(fused_backbone_fpn) - 1:
            fused_features["vision_features"] = fused_feature
        return fused_features

    def _apply_task_head(
        self,
        logits: torch.Tensor,
        task_ids: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_task_head:
            return logits
        return self.task_heads(logits, task_ids)

    def _mask_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        masks = torch.sigmoid(logits)
        if self.use_mask_threshold:
            masks = (masks > self.mask_threshold).float()
        return masks

    def _predict_boundary(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.use_boundary_head:
            return torch.zeros_like(logits)
        return self.boundary_head(logits)

    def forward(self, videos, bboxes, dense_prompts, task_ids, labels=None):
        self.init_state()
        batch_size, num_frames, channels, height, width = videos.shape
        self.num_frames = num_frames
        self._orig_hw = [height, width]
        self.batch_size = batch_size

        flat_videos = videos.view(batch_size * num_frames, channels, height, width)
        features = self.model.forward_image(flat_videos)
        features = self._apply_dense_prompt_fusion(
            features,
            dense_prompts,
            batch_size,
            num_frames,
        )
        features = {
            key: (
                value.view(batch_size, num_frames, *value.shape[1:])
                if not isinstance(value, list)
                else [
                    item.view(batch_size, num_frames, *item.shape[1:])
                    for item in value
                ]
            )
            for key, value in features.items()
        }
        frame_features = self.preprocess_frame_features(
            features,
            batch_size,
            num_frames,
        )

        first_frame_features = frame_features[0]
        first_frame_bbox = bboxes.view(batch_size, 4)
        first_frame_masks, first_frame_logits, first_frame_ious, object_score_logits = (
            self._predict_first_frame(first_frame_features, first_frame_bbox)
        )
        first_frame_logits = self._apply_task_head(first_frame_logits, task_ids)
        first_frame_masks = self._mask_from_logits(first_frame_logits)
        first_frame_sdt = self._predict_boundary(first_frame_logits)

        prev_pred_mask = first_frame_masks if labels is None else labels[:, 0]
        memory = self._initialize_memory(
            first_frame_features,
            prev_pred_mask,
            object_score_logits,
        )

        all_masks = [first_frame_masks]
        all_logits = [first_frame_logits]
        all_ious = [first_frame_ious]
        all_sdt = [first_frame_sdt]
        for frame_index in range(1, num_frames):
            self.current_frame_idx = frame_index
            frame_feature = frame_features[frame_index]
            masks, logits, ious, object_score_logits = self._predict_frame(
                frame_feature,
                memory,
                prev_pred_mask,
            )
            logits = self._apply_task_head(logits, task_ids)
            masks = self._mask_from_logits(logits)
            sdt = self._predict_boundary(logits)

            all_masks.append(masks)
            all_logits.append(logits)
            all_ious.append(ious)
            all_sdt.append(sdt)

            if frame_index < num_frames - 1:
                prev_pred_mask = masks if labels is None else labels[:, frame_index]
                memory = self._update_memory(
                    frame_feature,
                    prev_pred_mask,
                    memory,
                    object_score_logits,
                )

        self.reset_state()
        return all_masks, all_logits, all_ious, all_sdt
