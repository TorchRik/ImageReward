import typing as tp
from abc import abstractmethod

import torch.utils.checkpoint


class BaseModel(torch.nn.Module):
    @abstractmethod
    def tokenize(
        self, batch: tp.Dict[str, torch.Tensor], caption_column: str
    ) -> tp.Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def score_grad(
        self,
        batch: tp.Dict[str, torch.Tensor],
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        pass

    #
    # @abstractmethod
    # def score(self, prompts: tp.Sequence[str], image: torch.Tensor) -> torch.Tensor:
    #     pass
    #
