"""
@File       :   ImageReward.py
@Time       :   2023/01/28 19:53:00
@Auther     :   Jiazheng Xu
@Contact    :   xjz22@mails.tsinghua.edu.cn
@Description:   ImageReward Reward model.
* Based on CLIP code base and improved-aesthetic-predictor code base
* https://github.com/openai/CLIP
* https://github.com/christophschuhmann/improved-aesthetic-predictor
"""

import typing as tp

import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

from ImageReward.models.BLIP.blip_pretrain import BLIP_Pretrain

from .base_reward import BaseModel

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose(
        [
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size

        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            # nn.ReLU(),
            nn.Linear(16, 1),
        )

        # initial MLP param
        for name, param in self.layers.named_parameters():
            if "weight" in name:
                nn.init.normal_(param, mean=0.0, std=1.0 / (self.input_size + 1))
            if "bias" in name:
                nn.init.constant_(param, val=0)

    def forward(self, input):
        return self.layers(input)


class ImageReward(BaseModel):
    def __init__(self, med_config, device="cpu"):
        super().__init__()
        self.device = device

        self.blip = BLIP_Pretrain(image_size=224, vit="large", med_config=med_config)
        self.preprocess = _transform(224)
        self.mlp = MLP(768)

        self.mean = 0.16717362830052426
        self.std = 1.0333394966054072
        self.model_suffix = "_image_reward"

    def tokenize(
        self, batch: tp.Dict[str, torch.Tensor], caption_column: str
    ) -> tp.Dict[str, torch.Tensor]:
        caption = batch[caption_column]
        self.reward_model_1.tokenize()
        processed_caption = self.blip.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=35,
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
        image_embeds = self.blip.visual_encoder(image)
        # text encode cross attention with image
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            self.device
        )
        text_output = self.blip.text_encoder(
            batch[f"input_ids{self.model_suffix}"],
            attention_mask=batch[f"attention_mask{self.model_suffix}"],
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        txt_features = text_output.last_hidden_state[:, 0, :]  # (feature_dim)
        rewards = self.mlp(txt_features)
        normalized_reward = -(rewards - self.mean) / self.std
        logg_reward = rewards.mean().detach().item()

        return normalized_reward, {self.model_suffix: logg_reward}
