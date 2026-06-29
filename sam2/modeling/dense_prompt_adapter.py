from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(num_channels: int) -> nn.GroupNorm:
    for groups in (32, 16, 8, 4, 2, 1):
        if num_channels % groups == 0:
            return nn.GroupNorm(groups, num_channels)
    return nn.GroupNorm(1, num_channels)


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            _group_norm(out_channels),
            nn.GELU(),
        )


class DenseBoxPromptAdapter(nn.Module):
    def __init__(self, in_channels: int = 2, out_channels: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvNormAct(in_channels, 32, stride=1),
            ConvNormAct(32, 64, stride=2),
            ConvNormAct(64, 128, stride=2),
            nn.Conv2d(128, out_channels, kernel_size=1),
            _group_norm(out_channels),
            nn.GELU(),
        )

    def forward(
        self,
        dense_prompts: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        if dense_prompts.ndim != 4:
            raise ValueError(
                f"dense_prompts must have shape (N,C,H,W), got {tuple(dense_prompts.shape)}"
            )
        prompt_feature = self.encoder(dense_prompts.float())
        if prompt_feature.shape[-2:] != target_size:
            prompt_feature = F.interpolate(
                prompt_feature,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
        return prompt_feature


class GatedFeatureFusion(nn.Module):
    def __init__(self, channels: int = 256):
        super().__init__()
        self.gate = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        image_feature: torch.Tensor,
        prompt_feature: torch.Tensor,
    ) -> torch.Tensor:
        if image_feature.shape != prompt_feature.shape:
            raise ValueError(
                "image_feature and prompt_feature must have identical shapes, "
                f"got {tuple(image_feature.shape)} and {tuple(prompt_feature.shape)}"
            )
        concat = torch.cat([image_feature, prompt_feature], dim=1)
        gate = torch.sigmoid(self.gate(concat))
        return image_feature + self.gamma * gate * prompt_feature


class TaskSpecificResidualHeads(nn.Module):
    def __init__(
        self,
        num_tasks: int = 2,
        hidden_channels: int = 8,
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(hidden_channels, 1, kernel_size=1),
                )
                for _ in range(num_tasks)
            ]
        )

    def forward(self, logits: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError(f"logits must have shape (B,1,H,W), got {tuple(logits.shape)}")
        task_ids = task_ids.to(device=logits.device, dtype=torch.long).view(-1)
        if task_ids.numel() != logits.shape[0]:
            raise ValueError(
                f"task_ids length {task_ids.numel()} does not match batch {logits.shape[0]}"
            )
        residual = torch.zeros_like(logits)
        for task_index, head in enumerate(self.heads):
            mask = task_ids == task_index
            if bool(mask.any()):
                residual[mask] = head(logits[mask])
        return logits + residual


class SignedDistanceBoundaryHead(nn.Module):
    def __init__(self, hidden_channels: int = 16):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError(f"logits must have shape (B,1,H,W), got {tuple(logits.shape)}")
        return self.head(logits)
