from transformers import AutoProcessor, AutoModel
import torch

processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"


class PickScore(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.processor = AutoProcessor.from_pretrained(processor_name_or_path)
        self.model = AutoModel.from_pretrained(
            model_pretrained_name_or_path
        ).eval()

        self.mean = 21.38
        self.std = 1.18

    def tokenize(self, *args, **kwargs):
        return self.processor.tokenize(*args, **kwargs)

    def score_gard(self, rm_input_ids, rm_attention_mask, image):
        image_inputs = self.processor(
            images=image,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        image_embs = self.model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

        text_embs = self.model.get_text_features(input_ids=rm_input_ids, attention_mask=rm_attention_mask)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

        rewards = self.model.logit_scale.exp() * torch.diag(text_embs @ image_embs.T)
        rewards = (rewards - self.mean) / self.std

        return rewards

    def score(self, prompt, image):
        image_inputs = self.processor(
            images=image,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        text_inputs = self.processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        image_embs = self.model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1,
                                             keepdim=True)

        text_embs = self.model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

        rewards = self.model.logit_scale.exp() * torch.diag(
            text_embs @ image_embs.T)
        rewards = (rewards - self.mean) / self.std

        return rewards.detach().cpu().numpy().item()