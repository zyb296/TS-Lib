import torch
import torch.nn as nn
from torch import Tensor
from typing import Mapping, Iterable, Tuple, Callable, Optional

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask
    
    
class RandomMasker(nn.Module):
    def __init__(self, mask_ratio, mask_length=1) -> None:
        """
        mask the input tensor and store the mask in property.
        mask_ratio (float) [0, 1): The ratio of masked tokens in the input sequence.
            A mask_ration = 0 will implement Identity input-output.
        mask_length (int, optional): The length of the masked tokens. Defaults to 1.
            A mask_length > 1 will implement patch masking, where the length of mask and non-masked values are both n*mask_length.
        TODO: learnable mask token
        TODO: customize loss params
        FIXME: not test for mask_ratio == 0
        """
        super().__init__()
        assert 0 <= mask_ratio < 1
        self.mask_ratio = mask_ratio
        self.mask_length = mask_length
        self.mask: Tensor = torch.zeros(1, 1, 1, dtype=torch.bool)

    def forward(self, input_x: Tensor) -> Tensor:
        """
        tensor: (batch_size, seq_len, n_features)
        """
        mask = self.create_mask(input_x)
        masked_tensor, mask = self.mask_tensor(input_x, mask)
        self.mask = mask
        return masked_tensor, mask

    def create_mask(self, input_x: Tensor) -> Tensor:
        """
        input: Tensor of shape (b, l, d)
        output: mask of shape (b, l, d) where 1 denotes mask and 0 denotes non-mask
        """
        assert (
            input_x.shape[1] % self.mask_length == 0
        ), f"mask_length should be a divisor of sequence length, but got {self.mask_length} and {input_x.shape[1]}"
        mask = (
            torch.rand(input_x.shape[0], input_x.shape[1] // self.mask_length)
            < self.mask_ratio
        ).to(input_x.device)
        mask = mask.repeat_interleave(self.mask_length, dim=1)
        mask = mask.unsqueeze(dim=-1).repeat(1, 1, input_x.shape[-1])
        return mask

    def mask_tensor(self, input_x: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        # here I simply set the masked values to 0, which may not be the best choice.
        masked_tensor = input_x.where(~mask, 0)
        return masked_tensor, mask
    

# if __name__ == "__main__":
#     random_masker = RandomMasker(mask_ratio=0.3, mask_length=3)

#     x = torch.rand(3, 18, 2)
#     masked_x, mask = random_masker(x)
#     mask = random_masker.mask
