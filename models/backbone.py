import torch.nn as nn
import torch

DINOV2_ARCHS = {
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024,
    'dinov2_vitg14': 1536,
}


class DINOv2(nn.Module):

    def __init__(
            self,
            model_name='dinov2_vitb14',
            num_trainable_blocks=1,
            norm_layer=False,
            return_token=False
    ):
        super().__init__()
        assert model_name in DINOV2_ARCHS.keys(), f'Unknown model name {model_name}'
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.num_channels = DINOV2_ARCHS[model_name]
        self.num_trainable_blocks = num_trainable_blocks
        self.norm_layer = norm_layer
        self.return_token = return_token

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.model.prepare_tokens_with_masks(x)
        with torch.no_grad():
            for blk in self.model.blocks[:-self.num_trainable_blocks]:
                x = blk(x)
        x = x.detach()
        for blk in self.model.blocks[-self.num_trainable_blocks:]:
            x = blk(x)
        if self.norm_layer:
            x = self.model.norm(x)
        t = x[:, 0]
        f = x[:, 1:]
        f = f.reshape((B, H // 14, W // 14, self.num_channels)).permute(0, 3, 1, 2)

        if self.return_token:
            return f, t
        return f

