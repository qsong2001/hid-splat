import torch
import torch.nn as nn

from models.transformer import Encoder, Group, TransformerEncoder


ATTRIBUTE_DIMS = {
    "xyz": 3,
    "opacity": 1,
    "scale": 3,
    "rotation": 4,
    "sh": 3,
}


def attribute_dim(attributes):
    return sum(ATTRIBUTE_DIMS[name] for name in attributes)


class GaussianWatermarkEmbedder(nn.Module):
    def __init__(self, num_bits=64, feature_dim=14, hidden_dim=256, depth=4, delta_scale=0.01):
        super().__init__()
        self.delta_scale = delta_scale
        layers = []
        in_dim = feature_dim + num_bits
        for i in range(depth):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.Linear(hidden_dim, feature_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, gaussians, messages):
        msg = messages.unsqueeze(1).expand(-1, gaussians.shape[1], -1)
        delta = torch.tanh(self.net(torch.cat([gaussians, msg], dim=-1))) * self.delta_scale
        return gaussians + delta, delta


class AttributeModulatedGaussianWatermarkEmbedder(nn.Module):
    def __init__(
        self,
        num_bits=64,
        feature_dim=14,
        hidden_dim=256,
        depth=4,
        delta_scale=0.01,
        attribute_budgets=(0.5, 1.0, 0.5, 0.25, 1.0),
        gate_floor=0.25,
        disable_carrier_gate=False,
    ):
        super().__init__()
        if feature_dim != 14:
            raise ValueError("Attribute-modulated embedding currently expects all 14 Gaussian channels.")
        if len(attribute_budgets) != 5:
            raise ValueError("attribute_budgets must contain xyz, opacity, scale, rotation, and SH values.")
        self.delta_scale = delta_scale
        self.gate_floor = gate_floor
        self.disable_carrier_gate = disable_carrier_gate
        expanded_budgets = (
            [attribute_budgets[0]] * 3
            + [attribute_budgets[1]]
            + [attribute_budgets[2]] * 3
            + [attribute_budgets[3]] * 4
            + [attribute_budgets[4]] * 3
        )
        self.register_buffer("attribute_budgets", torch.tensor(expanded_budgets).view(1, 1, feature_dim))

        self.point_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.message_encoder = nn.Sequential(
            nn.Linear(num_bits, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.message_film = nn.Linear(hidden_dim, hidden_dim * 2)
        residual_layers = []
        for _ in range(max(depth - 1, 1)):
            residual_layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                ]
            )
        self.residual_trunk = nn.Sequential(*residual_layers)
        self.residual_head = nn.Linear(hidden_dim, feature_dim)
        self.carrier_gate = None if disable_carrier_gate else nn.Linear(hidden_dim, 1)

    def forward(self, gaussians, messages):
        point_features = self.point_encoder(gaussians)
        message_features = self.message_encoder(messages)
        gamma, beta = self.message_film(message_features).chunk(2, dim=-1)
        fused = point_features * (1.0 + torch.tanh(gamma).unsqueeze(1)) + beta.unsqueeze(1)
        fused = self.residual_trunk(fused)

        if self.carrier_gate is None:
            gate = torch.ones((*fused.shape[:2], 1), dtype=fused.dtype, device=fused.device)
        else:
            gate = self.gate_floor + (1.0 - self.gate_floor) * torch.sigmoid(self.carrier_gate(fused))
            gate = gate / gate.mean(dim=1, keepdim=True).clamp_min(1e-6)
        raw_delta = torch.tanh(self.residual_head(fused))
        delta = raw_delta * gate * self.attribute_budgets * self.delta_scale
        return gaussians + delta, delta, gate


class GaussianWatermarkBackbone(nn.Module):
    def __init__(
        self,
        attribute=("xyz", "opacity", "scale", "rotation", "sh"),
        group_attribute=("xyz",),
        num_group=128,
        group_size=32,
        encoder_dims=384,
        trans_dim=384,
        depth=8,
        num_heads=6,
        drop_path_rate=0.0,
    ):
        super().__init__()
        self.group_attribute = tuple(group_attribute)
        self.group_divider = Group(num_group=num_group, group_size=group_size, attribute=list(group_attribute))
        self.encoder = Encoder(encoder_channel=encoder_dims, attribute=list(attribute))
        if encoder_dims != trans_dim:
            raise ValueError("Strict pretrained loading expects encoder_dims == trans_dim.")
        self.cls_token = nn.Parameter(torch.zeros(1, 1, trans_dim))
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, trans_dim))
        self.pos_index = self._pos_index(self.group_attribute)
        self.pos_embed = nn.Sequential(
            nn.Linear(len(self.pos_index), 128),
            nn.GELU(),
            nn.Linear(128, trans_dim),
        )
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = TransformerEncoder(embed_dim=trans_dim, depth=depth, num_heads=num_heads, drop_path_rate=dpr)
        self.norm = nn.LayerNorm(trans_dim)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.cls_pos, std=0.02)

    @staticmethod
    def _pos_index(group_attribute):
        idx = []
        if "xyz" in group_attribute:
            idx.extend([0, 1, 2])
        if "opacity" in group_attribute:
            idx.append(3)
        if "scale" in group_attribute:
            idx.extend([4, 5, 6])
        if "rotation" in group_attribute:
            idx.extend([7, 8, 9, 10])
        if "sh" in group_attribute:
            idx.extend([11, 12, 13])
        return idx

    def forward_features(self, gaussians):
        neighborhood, center = self.group_divider(gaussians)
        tokens = self.encoder(neighborhood)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        cls_pos = self.cls_pos.expand(tokens.shape[0], -1, -1)
        pos = self.pos_embed(center[..., self.pos_index])
        x = torch.cat([cls, tokens], dim=1)
        pos = torch.cat([cls_pos, pos], dim=1)
        x = self.norm(self.blocks(x, pos))
        return torch.cat([x[:, 0], x[:, 1:].max(dim=1)[0]], dim=-1)

    def forward(self, gaussians):
        return self.forward_features(gaussians)


class GaussianWatermarkExtractor(nn.Module):
    def __init__(self, num_bits=64, trans_dim=384, **kwargs):
        super().__init__()
        self.backbone = GaussianWatermarkBackbone(trans_dim=trans_dim, **kwargs)
        self.head = nn.Sequential(
            nn.Linear(trans_dim * 2, trans_dim),
            nn.GELU(),
            nn.Linear(trans_dim, num_bits),
        )

    def forward(self, gaussians):
        return self.head(self.backbone(gaussians))


class GaussianWatermarkModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        num_bits = kwargs.pop("num_bits", 64)
        attribute = tuple(kwargs.pop("attribute", ("xyz", "opacity", "scale", "rotation", "sh")))
        feature_dim = attribute_dim(attribute)
        embedder_type = kwargs.pop("embedder_type", "baseline")
        attribute_budgets = kwargs.pop("attribute_budgets", (0.5, 1.0, 0.5, 0.25, 1.0))
        gate_floor = kwargs.pop("gate_floor", 0.25)
        disable_carrier_gate = kwargs.pop("disable_carrier_gate", False)
        embedder_args = {
            "num_bits": num_bits,
            "feature_dim": feature_dim,
            "hidden_dim": kwargs.pop("embedder_hidden_dim", 256),
            "depth": kwargs.pop("embedder_depth", 4),
            "delta_scale": kwargs.pop("delta_scale", 0.01),
        }
        if embedder_type == "baseline":
            self.embedder = GaussianWatermarkEmbedder(**embedder_args)
        elif embedder_type == "attribute_modulated":
            self.embedder = AttributeModulatedGaussianWatermarkEmbedder(
                **embedder_args,
                attribute_budgets=attribute_budgets,
                gate_floor=gate_floor,
                disable_carrier_gate=disable_carrier_gate,
            )
        else:
            raise ValueError(f"Unknown embedder_type: {embedder_type}")
        self.embedder_type = embedder_type
        self.extractor = GaussianWatermarkExtractor(num_bits=num_bits, attribute=attribute, **kwargs)

    def forward(self, gaussians, messages, noise_std=0.0, dropout_prob=0.0, prune_ratio=0.0):
        embedder_output = self.embedder(gaussians, messages)
        watermarked, delta = embedder_output[:2]
        self.last_carrier_gate = embedder_output[2] if len(embedder_output) == 3 else None
        attacked = augment_gaussians(
            watermarked,
            noise_std=noise_std,
            dropout_prob=dropout_prob,
            prune_ratio=prune_ratio,
        )
        logits = self.extractor(attacked)
        return logits, watermarked, attacked, delta


def augment_gaussians(gaussians, noise_std=0.0, dropout_prob=0.0, prune_ratio=0.0):
    out = gaussians
    if noise_std > 0:
        noise = torch.randn_like(out)
        noise[..., :3] = noise[..., :3] * noise_std
        if out.shape[-1] > 3:
            noise[..., 3:] = noise[..., 3:] * (noise_std * 0.25)
        out = out + noise

    if dropout_prob > 0:
        keep = (torch.rand(out.shape[:2], device=out.device) > dropout_prob).float().unsqueeze(-1)
        out = out * keep + gaussians.detach() * (1.0 - keep)

    if prune_ratio > 0:
        keep_count = max(1, int(round(out.shape[1] * (1.0 - prune_ratio))))
        random_scores = torch.rand(out.shape[:2], device=out.device)
        indices = random_scores.topk(keep_count, dim=1, largest=True, sorted=False).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, out.shape[-1])
        out = torch.gather(out, dim=1, index=indices)

    permutation = torch.randperm(out.shape[1], device=out.device)
    return out[:, permutation, :]
