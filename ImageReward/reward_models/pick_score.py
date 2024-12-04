import typing as tp

import torch
import torch.utils.checkpoint
from transformers import AutoModel, AutoProcessor

from .base_reward import BaseModel

PROCESSOR_NAME = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
PRETRAINED_MODEL_NAME = "yuvalkirstain/PickScore_v1"


class PickScore(BaseModel):
    def __init__(self, device: torch.device):
        super().__init__()

        self.processor = AutoProcessor.from_pretrained(PROCESSOR_NAME)
        self.model = AutoModel.from_pretrained(PRETRAINED_MODEL_NAME).eval()

        self.device = device
        self.mean = 21.38
        self.std = 1.18
        self.model_suffix = "pick_score"

    def tokenize(
        self, batch: tp.Dict[str, torch.Tensor], caption_column: str
    ) -> tp.Dict[str, torch.Tensor]:
        caption = batch[caption_column]
        self.reward_model_1.tokenize()
        processed_caption = self.processor(
            text=caption,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        batch[f"input_ids{self.model_suffix}"] = processed_caption.input_ids
        batch[f"attention_mask{self.model_suffix}"] = processed_caption.attention_mask
        return batch

    def score_grad(
        self,
        batch: tp.Dict[str, torch.Tensor],
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        image_inputs = {"pixel_values": image}

        image_embs = self.model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

        text_embs = self.model.get_text_features(
            input_ids=batch[f"input_ids{self.model_suffix}"],
            attention_mask=batch[f"attention_mask{self.model_suffix}"],
        )
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

        rewards = self.model.logit_scale.exp() * torch.diag(text_embs @ image_embs.T)
        normalized_reward = -(rewards - self.mean) / self.std
        logg_reward = rewards.mean().detach().item()

        return normalized_reward, {self.model_suffix: logg_reward}
