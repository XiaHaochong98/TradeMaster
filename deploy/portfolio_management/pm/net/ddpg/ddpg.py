import torch
import torch.nn as nn
import math
from typing import List
from functools import partial
from torch.distributions.normal import Normal
from timm.models.layers import Mlp
import torch.nn.functional as F

from pm.registry import NET
from pm.registry import EMBED

@NET.register_module(force=True)
class ActorDDPG(nn.Module):
    def __init__(self,
                 *args,
                 embed_type: str = "TimesEmbed",
                 feature_size: List[int] = (10, 99),
                 t_patch_size: int = 1,
                 num_stocks: int = 420,
                 input_dim = 102,
                 temporal_dim = 3,
                 in_chans: int = 1,
                 embed_dim: int = 128,
                 depth: int = 2,
                 norm_layer: nn.LayerNorm = partial(nn.LayerNorm, eps=1e-6),
                 cls_embed: bool = True,
                 explore_noise_std: float = 0.1,
                 **kwargs
                 ):
        super(ActorDDPG, self).__init__()
        self.embed_type = embed_type,
        self.input_dim = input_dim
        self.temporal_dim = temporal_dim
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depth = depth
        self.norm_layer = norm_layer
        self.cls_embed = cls_embed
        self.explore_noise_std = explore_noise_std

        self.emb_config = dict(
            type=embed_type,
            img_size=feature_size,
            patch_size=feature_size,
            in_chans=in_chans,
            input_dim=input_dim,
            temporal_dim=temporal_dim,
            embed_dim=embed_dim,
            frames=num_stocks,
            t_patch_size=t_patch_size,
        )
        self.patch_embed = EMBED.build(self.emb_config)

        if self.cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList(
            [
                Mlp(in_features=embed_dim, hidden_features=embed_dim, out_features=embed_dim)
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)

        self.decoder_pred = nn.Linear(
            embed_dim,
            1,
            bias=True,
        )

        self.initialize_weights()

    def initialize_weights(self):
        if self.cls_embed:
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight, 1.0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 1e-6)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x):
        """
        b, n, d, f = x.shape # batch size, num stocks, days, features
        """
        if len(x.shape) == 4:
            x = x.unsqueeze(1)

        x = self.patch_embed(x)

        if self.cls_embed:
            cls_token = self.cls_token
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward_decoder(self, x):
        x = self.decoder_pred(x)
        return x

    def forward(self, x):
        latent = self.forward_encoder(x)
        pred = self.forward_decoder(latent)

        pred = pred.squeeze(-1)

        return pred

    def get_action(self, x):

        latent = self.forward_encoder(x)
        pred = self.forward_decoder(latent)

        pred = pred.squeeze(-1)
        noise = (torch.randn_like(pred) * self.explore_noise_std)

        pred = pred + noise

        return pred

    def get_action_noise(self, x, action_std):

        latent = self.forward_encoder(x)
        pred = self.forward_decoder(latent)

        pred = pred.squeeze(-1)

        noise = (torch.randn_like(pred) * action_std)

        pred = pred + noise

        return pred

@NET.register_module(force=True)
class CriticDDPG(nn.Module):
    def __init__(self,
                 *args,
                 embed_type: str = "TimesEmbed",
                 feature_size: List[int] = (10, 99),
                 t_patch_size: int = 1,
                 num_stocks: int = 420,
                 input_dim = 102,
                 temporal_dim = 3,
                 in_chans: int = 1,
                 embed_dim: int = 128,
                 depth: int = 2,
                 norm_layer: nn.LayerNorm = partial(nn.LayerNorm, eps=1e-6),
                 cls_embed: bool = True,
                 **kwargs
                 ):
        super(CriticDDPG, self).__init__()
        self.embed_type = embed_type,
        self.input_dim = input_dim
        self.temporal_dim = temporal_dim
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depth = depth
        self.norm_layer = norm_layer
        self.cls_embed = cls_embed

        self.emb_config = dict(
            type=embed_type,
            img_size=feature_size,
            patch_size=feature_size,
            in_chans=in_chans,
            input_dim=input_dim,
            temporal_dim=temporal_dim,
            embed_dim=embed_dim,
            frames=num_stocks,
            t_patch_size=t_patch_size,
        )
        self.patch_embed = EMBED.build(self.emb_config)

        if self.cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList(
            [
                Mlp(in_features=embed_dim+1, hidden_features=embed_dim+1, out_features=embed_dim+1)
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim+1)

        self.decoder_pred = nn.Linear(
            embed_dim+1,
            1,
            bias=True,
        )

        self.initialize_weights()


    def initialize_weights(self):
        if self.cls_embed:
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight, 1.0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 1e-6)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x, action):
        """
        b, n, d, f = x.shape # batch size, num stocks, days, features
        """
        if len(x.shape) == 4:
            x = x.unsqueeze(1)

        x = self.patch_embed(x)

        if self.cls_embed:
            cls_token = self.cls_token
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        if len(action.shape) == 2:
            action = action.unsqueeze(-1)

        x = torch.concat([x, action], dim=-1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x

    def forward_decoder(self, x):
        # predictor projection
        x = self.decoder_pred(x)
        return x


    def forward(self, x, action):
        latent = self.forward_encoder(x, action)
        latent = torch.sum(latent, dim=1)
        pred = self.forward_decoder(latent)
        value = pred.squeeze(-1)
        return value



if __name__ == '__main__':
    device = torch.device("cpu")

    feature = torch.randn(4, 1, 420, 30, 99)
    temporal = torch.zeros(4, 1, 420, 30, 3)
    noise = float(0.1)

    batch = torch.cat([feature, temporal], dim=-1).to(device)

    model = ActorDDPG().to(device)
    pred = model(batch)
    print(pred.shape)

    action= model.get_action(batch)
    print(action.shape)

    action= model.get_action_noise(batch, noise)
    print(action.shape)

    action = torch.randn((4, 421)).to(device)
    model = CriticDDPG().to(device)
    values = model(batch, action)
    print(values.shape)