import typing as tp

import torch
import torch.utils.checkpoint

from .base_reward import BaseModel

processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"


class CombinedRewardModel(BaseModel):
    def __init__(self, models, device):
        super().__init__()

        self.models = models

        self.device = device

    def tokenize(
        self, batch: tp.Dict[str, torch.Tensor], caption_column: str
    ) -> tp.Dict[str, torch.Tensor]:
        for model in self.models:
            batch = model.tokenize(batch, caption_column)
        return batch

    def score_grad(
        self,
        batch: tp.Dict[str, torch.Tensor],
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        result_loss = None
        result_log = None
        for model in self.models:
            if result_loss is None:
                result_loss, result_log = model.score_grad(batch, image)
            else:
                loss, log = model.score_grad(batch, image)
                result_loss += loss
                result_log.update(log)
        result_loss /= len(self.models)
        return result_loss, result_log
