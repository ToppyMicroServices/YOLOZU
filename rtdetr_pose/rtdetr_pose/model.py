try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover - optional dependency for scaffolding
    from types import SimpleNamespace

    torch = None
    nn = SimpleNamespace(Module=object)
    F = SimpleNamespace()


# Import entropy_loss for MIM branch
def _entropy_loss_fallback(logits):
    """Fallback entropy loss if losses module not available."""
    if torch is None or F is None:
        raise RuntimeError("torch is required for entropy loss")
    probs = F.softmax(logits, dim=-1)
    log_probs = torch.log(torch.clamp(probs, min=1e-12))
    return -(probs * log_probs).sum(dim=-1).mean()


def entropy_loss(logits):
    try:
        from .losses import entropy_loss as losses_entropy_loss
    except ImportError:
        losses_entropy_loss = _entropy_loss_fallback
    return losses_entropy_loss(logits)

from .backbone_projector import BackboneProjector


def _normalize_activation_name(name: str) -> str:
    value = str(name or "silu").strip().lower().replace("-", "_")
    aliases = {
        "swish": "silu",
        "hard_swish": "hardswish",
        "leaky_relu": "leakyrelu",
    }
    return aliases.get(value, value)


def build_activation(name: str, *, inplace: bool = True):
    act = _normalize_activation_name(name)
    if act == "silu":
        return nn.SiLU(inplace=inplace)
    if act == "gelu":
        return nn.GELU()
    if act == "hardswish":
        return nn.Hardswish(inplace=inplace)
    if act == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
    raise ValueError(f"unsupported activation: {name}")


def rot6d_to_matrix(x):
    from .rotation import rot6d_to_matrix as convert_rot6d_to_matrix

    return convert_rot6d_to_matrix(x)


def _sincos_1d(pos, dim, device, dtype):
    half_dim = dim // 2
    omega = torch.arange(half_dim, device=device, dtype=dtype)
    omega = 1.0 / (10000 ** (omega / half_dim))
    out = pos[:, None] * omega[None, :]
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


class SinePositionEmbedding(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        if hidden_dim % 4 != 0:
            raise ValueError("hidden_dim must be divisible by 4 for 2D sin/cos")
        self.hidden_dim = hidden_dim

    def forward(self, height, width, device, dtype):
        xs = torch.arange(width, device=device, dtype=dtype)
        ys = torch.arange(height, device=device, dtype=dtype)
        # Avoid repeat_interleave here: it can introduce CPU tensors during ONNX tracing
        # in some environments. Expand/reshape keeps everything on the target device.
        grid_x = xs.unsqueeze(0).expand(height, width).reshape(-1)
        grid_y = ys.unsqueeze(1).expand(height, width).reshape(-1)
        dim_half = self.hidden_dim // 2
        pos_y = _sincos_1d(grid_y, dim_half, device, dtype)
        pos_x = _sincos_1d(grid_x, dim_half, device, dtype)
        pos = torch.cat([pos_y, pos_x], dim=1)
        return pos.unsqueeze(0)


class ConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation="silu"):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = build_activation(str(activation), inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=0.5, shortcut=True, activation="silu"):
        super().__init__()
        hidden = int(out_channels * expansion)
        self.conv1 = ConvNormAct(in_channels, hidden, kernel_size=1, activation=activation)
        self.conv2 = ConvNormAct(hidden, out_channels, kernel_size=3, padding=1, activation=activation)
        self.shortcut = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.shortcut:
            y = y + x
        return y


class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=1, expansion=0.5, activation="silu"):
        super().__init__()
        hidden = int(out_channels * expansion)
        self.conv1 = ConvNormAct(in_channels, hidden, kernel_size=1, activation=activation)
        self.conv2 = ConvNormAct(in_channels, hidden, kernel_size=1, activation=activation)
        self.blocks = nn.Sequential(
            *[Bottleneck(hidden, hidden, expansion=1.0, activation=activation) for _ in range(num_blocks)]
        )
        self.conv3 = ConvNormAct(hidden * 2, out_channels, kernel_size=1, activation=activation)

    def forward(self, x):
        y1 = self.blocks(self.conv1(x))
        y2 = self.conv2(x)
        return self.conv3(torch.cat([y1, y2], dim=1))


class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size=5, activation="silu"):
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels, kernel_size=1, activation=activation)
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=1, padding=pool_size // 2)
        self.conv2 = ConvNormAct(out_channels * 4, out_channels, kernel_size=1, activation=activation)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.conv2(torch.cat([x, y1, y2, y3], dim=1))


class CSPResNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        stem_channels=32,
        stage_channels=(64, 128, 256),
        stage_blocks=(1, 2, 2),
        use_sppf=True,
        activation="silu",
    ):
        super().__init__()
        self.stem = nn.Sequential(
            ConvNormAct(in_channels, stem_channels, kernel_size=3, stride=2, padding=1, activation=activation),
            ConvNormAct(stem_channels, stem_channels, kernel_size=3, padding=1, activation=activation),
            ConvNormAct(stem_channels, stem_channels * 2, kernel_size=3, padding=1, activation=activation),
        )
        in_ch = stem_channels * 2
        stages = []
        for out_ch, num_blocks in zip(stage_channels, stage_blocks):
            stages.append(
                nn.Sequential(
                    ConvNormAct(in_ch, out_ch, kernel_size=3, stride=2, padding=1, activation=activation),
                    CSPBlock(out_ch, out_ch, num_blocks=num_blocks, activation=activation),
                )
            )
            in_ch = out_ch
        self.stages = nn.ModuleList(stages)
        self.sppf = SPPF(stage_channels[-1], stage_channels[-1], activation=activation) if use_sppf else None

    def forward(self, x):
        x = self.stem(x)
        outputs = []
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        if self.sppf is not None:
            outputs[-1] = self.sppf(outputs[-1])
        return outputs


class FPNPAN(nn.Module):
    def __init__(self, in_channels, out_channels, activation="silu"):
        super().__init__()
        self.lateral = nn.ModuleList(
            [ConvNormAct(ch, out_channels, kernel_size=1, activation=activation) for ch in in_channels]
        )
        self.fpn_convs = nn.ModuleList(
            [ConvNormAct(out_channels, out_channels, kernel_size=3, padding=1, activation=activation) for _ in in_channels]
        )
        self.downsample = nn.ModuleList(
            [ConvNormAct(out_channels, out_channels, kernel_size=3, stride=2, padding=1, activation=activation) for _ in in_channels[1:]]
        )
        self.pan_convs = nn.ModuleList(
            [ConvNormAct(out_channels, out_channels, kernel_size=3, padding=1, activation=activation) for _ in in_channels[1:]]
        )

    def forward(self, features):
        feats = [lat(feat) for lat, feat in zip(self.lateral, features)]
        p5 = feats[-1]
        p4 = feats[-2] + F.interpolate(p5, size=feats[-2].shape[-2:], mode="nearest")
        p3 = feats[-3] + F.interpolate(p4, size=feats[-3].shape[-2:], mode="nearest")
        p3 = self.fpn_convs[0](p3)
        p4 = self.fpn_convs[1](p4)
        p5 = self.fpn_convs[2](p5)
        n4 = self.pan_convs[0](p4 + self.downsample[0](p3))
        n5 = self.pan_convs[1](p5 + self.downsample[1](n4))
        return [p3, n4, n5]


class HybridEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_dim,
        num_layers=1,
        nhead=8,
        dim_feedforward=None,
        use_level_embed=True,
        activation="silu",
        depth_mode="none",
        depth_dropout=0.0,
    ):
        super().__init__()
        self.projector = BackboneProjector(in_channels=in_channels, d_model=hidden_dim)
        self.fpn = FPNPAN(
            in_channels=(hidden_dim, hidden_dim, hidden_dim),
            out_channels=hidden_dim,
            activation=activation,
        )
        self.num_layers = num_layers
        self.depth_mode = str(depth_mode or "none").strip().lower()
        self.depth_dropout = max(0.0, min(1.0, float(depth_dropout)))
        self.depth_proj = None
        self.depth_gate = None
        if self.depth_mode == "fuse_mid":
            self.depth_proj = nn.ModuleList(
                [ConvNormAct(1, hidden_dim, kernel_size=3, padding=1, activation=activation) for _ in range(3)]
            )
            self.depth_gate = nn.ModuleList([nn.Conv2d(hidden_dim, 1, kernel_size=1) for _ in range(3)])
        self.use_level_embed = bool(use_level_embed)
        self.level_embed = None
        if self.use_level_embed:
            self.level_embed = nn.Parameter(torch.zeros(3, hidden_dim))
        dim_feedforward = dim_feedforward or hidden_dim * 4
        if num_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=0.0,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        else:
            self.encoder = None

    def forward(self, features, pos_embed, depth=None, depth_valid=None):
        features = self.projector(features)
        if self.depth_mode == "fuse_mid" and self.depth_proj is not None and self.depth_gate is not None:
            if depth is not None and isinstance(depth, torch.Tensor) and depth.ndim == 4 and int(depth.shape[1]) == 1:
                depth_tensor = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
                finite_valid = torch.isfinite(depth).all(dim=(1, 2, 3))
                if depth_valid is not None and isinstance(depth_valid, torch.Tensor):
                    valid_mask = finite_valid & depth_valid.to(device=depth.device, dtype=torch.bool)
                else:
                    valid_mask = finite_valid
                if self.training and self.depth_dropout > 0.0:
                    keep = torch.rand((depth.shape[0],), device=depth.device) >= float(self.depth_dropout)
                    valid_mask = valid_mask & keep

                fused_features = []
                for idx, feat in enumerate(features):
                    dep = F.interpolate(
                        depth_tensor,
                        size=feat.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    dep_feat = self.depth_proj[idx](dep)
                    gate = torch.sigmoid(self.depth_gate[idx](dep_feat))
                    gate = gate * valid_mask.view(-1, 1, 1, 1).to(dtype=gate.dtype)
                    fused_features.append(feat + gate * dep_feat)
                features = fused_features
        features = self.fpn(features)
        memories = []
        pos_list = []
        for idx, feat in enumerate(features):
            mem = feat.flatten(2).permute(0, 2, 1)
            pos = pos_embed[idx]
            if self.level_embed is not None:
                level = self.level_embed[idx].view(1, 1, -1)
                pos = pos + level
            memories.append(mem)
            pos_list.append(pos)
        memory = torch.cat(memories, dim=1)
        pos = torch.cat(pos_list, dim=1)
        memory = memory + pos
        if self.encoder is not None:
            memory = self.encoder(memory)
        return memory, features


class RTDETRDecoder(nn.Module):
    def __init__(self, hidden_dim, num_layers=3, nhead=8, dim_feedforward=None):
        super().__init__()
        dim_feedforward = dim_feedforward or hidden_dim * 4
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)

    def forward(self, tgt, memory):
        return self.decoder(tgt, memory)


class HeadFast(nn.Module):
    def __init__(self, hidden_dim, num_classes, use_uncertainty=False):
        super().__init__()
        self.cls = nn.Linear(hidden_dim, num_classes)
        self.box = nn.Linear(hidden_dim, 4)
        self.log_z = nn.Linear(hidden_dim, 1)
        self.rot6d = nn.Linear(hidden_dim, 6)
        self.use_uncertainty = use_uncertainty
        if use_uncertainty:
            self.log_sigma_z = nn.Linear(hidden_dim, 1)
            self.log_sigma_rot = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out = {
            "logits": self.cls(x),
            "bbox": self.box(x),
            "log_z": self.log_z(x),
            "rot6d": self.rot6d(x),
        }
        if self.use_uncertainty:
            out["log_sigma_z"] = self.log_sigma_z(x)
            out["log_sigma_rot"] = self.log_sigma_rot(x)
        return out


class CenterOffsetHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.offset = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        return self.offset(x)


class GlobalKHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.delta = nn.Linear(hidden_dim, 4)

    def forward(self, x):
        return self.delta(x)


class KeypointsHead(nn.Module):
    def __init__(self, hidden_dim: int, num_keypoints: int):
        super().__init__()
        self.num_keypoints = int(num_keypoints)
        self.proj = nn.Linear(int(hidden_dim), int(num_keypoints) * 2)

    def forward(self, x):
        # x: (B, Q, hidden_dim) -> (B, Q, K, 2) in normalized coords (sigmoid)
        b, q, _ = x.shape
        k = int(self.num_keypoints)
        out = self.proj(x).sigmoid()
        return out.reshape(b, q, k, 2)


class RenderTeacher(nn.Module):
    """Geometry-derived teacher for MIM (train-only).
    
    Processes geometry tensors (mask + normalized depth) to produce
    a feature map for masked reconstruction supervision.
    """
    def __init__(self, hidden_dim=256, in_channels=2, activation="silu"):
        super().__init__()
        # Small CNN to encode geometry
        self.conv1 = ConvNormAct(in_channels, hidden_dim // 4, kernel_size=3, padding=1, activation=activation)
        self.conv2 = ConvNormAct(hidden_dim // 4, hidden_dim // 2, kernel_size=3, padding=1, activation=activation)
        self.conv3 = ConvNormAct(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1, activation=activation)
        
    def forward(self, geom_input):
        """
        Args:
            geom_input: (B, C, H, W) where C=2 (mask + normalized depth)
        Returns:
            features: (B, hidden_dim, H, W) geometry-derived features
        """
        if torch is None:
            raise RuntimeError("torch is required for RenderTeacher")
        x = self.conv1(geom_input)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class DecoderMIM(nn.Module):
    """Masked feature reconstruction decoder (train-only).
    
    Reconstructs masked features to match geometry-derived teacher features.
    """
    def __init__(self, hidden_dim=256, activation="silu"):
        super().__init__()
        # Simple decoder with upsampling
        self.conv1 = ConvNormAct(hidden_dim, hidden_dim, kernel_size=3, padding=1, activation=activation)
        self.conv2 = ConvNormAct(hidden_dim, hidden_dim, kernel_size=3, padding=1, activation=activation)
        self.out = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        
    def forward(self, masked_features):
        """
        Args:
            masked_features: (B, hidden_dim, H, W) masked neck features
        Returns:
            reconstructed: (B, hidden_dim, H, W) reconstructed features
        """
        if torch is None:
            raise RuntimeError("torch is required for DecoderMIM")
        x = self.conv1(masked_features)
        x = self.conv2(x)
        x = self.out(x)
        return x


class RTDETRPose(nn.Module):
    def __init__(
        self,
        num_classes=80,
        num_keypoints: int = 0,
        hidden_dim=256,
        num_queries=300,
        use_uncertainty=False,
        stem_channels=32,
        backbone_channels=(64, 128, 256),
        stage_blocks=(1, 2, 2),
        use_sppf=True,
        num_encoder_layers=1,
        num_decoder_layers=3,
        nhead=8,
        encoder_dim_feedforward=None,
        decoder_dim_feedforward=None,
        backbone=None,
        use_level_embed=True,
        enable_mim=False,
        mim_geom_channels=2,
        depth_mode="none",
        depth_dropout=0.0,
        backbone_activation="silu",
        head_activation="silu",
    ):
        super().__init__()
        if backbone is None:
            backbone = CSPResNet(
                stem_channels=stem_channels,
                stage_channels=backbone_channels,
                stage_blocks=stage_blocks,
                use_sppf=bool(use_sppf),
                activation=backbone_activation,
            )
        self.backbone = backbone
        self.position = SinePositionEmbedding(hidden_dim)
        self.encoder = HybridEncoder(
            in_channels=backbone_channels,
            hidden_dim=hidden_dim,
            num_layers=num_encoder_layers,
            nhead=nhead,
            dim_feedforward=encoder_dim_feedforward,
            use_level_embed=use_level_embed,
            activation=head_activation,
            depth_mode=depth_mode,
            depth_dropout=depth_dropout,
        )
        self.decoder = RTDETRDecoder(
            hidden_dim=hidden_dim,
            num_layers=num_decoder_layers,
            nhead=nhead,
            dim_feedforward=decoder_dim_feedforward,
        )
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.head = HeadFast(hidden_dim, num_classes, use_uncertainty=use_uncertainty)
        self.offset_head = CenterOffsetHead(hidden_dim)
        self.k_head = GlobalKHead(hidden_dim)
        self.keypoints_head = None
        if int(num_keypoints) > 0:
            self.keypoints_head = KeypointsHead(int(hidden_dim), int(num_keypoints))
        
        # Masked reconstruction branch (optional, train-only by default)
        self.enable_mim = bool(enable_mim)
        self.render_teacher = None
        self.decoder_mim = None
        if self.enable_mim:
            self.render_teacher = RenderTeacher(
                hidden_dim=hidden_dim,
                in_channels=mim_geom_channels,
                activation=head_activation,
            )
            self.decoder_mim = DecoderMIM(hidden_dim=hidden_dim, activation=head_activation)

    def forward(self, x, geom_input=None, feature_mask=None, return_mim=False, depth=None, depth_valid=None):
        """Forward pass with optional masked reconstruction branch.
        
        Args:
            x: (B, 3, H, W) input RGB image
            geom_input: (B, C, H', W') optional geometry input (mask + depth) for MIM teacher
            feature_mask: (H', W') optional mask for feature masking
            return_mim: bool, whether to return MIM outputs (training mode)
            
        Returns:
            out: dict with detection outputs and optional MIM outputs
        """
        if torch is None:
            raise RuntimeError("torch is required for RTDETRPose")
        batch = x.shape[0]
        feats = self.backbone(x)
        pos_embed = []
        for feat in feats:
            _, _, height, width = feat.shape
            pos = self.position(height, width, feat.device, feat.dtype).repeat(batch, 1, 1)
            pos_embed.append(pos)
        memory, encoder_feats = self.encoder(feats, pos_embed, depth=depth, depth_valid=depth_valid)
        queries = self.query_embed.weight.unsqueeze(0).repeat(batch, 1, 1)
        tgt = torch.zeros_like(queries) + queries
        dec = self.decoder(tgt, memory)
        out = self.head(dec)
        out["offsets"] = self.offset_head(dec)
        out["k_delta"] = self.k_head(dec.mean(dim=1))
        if self.keypoints_head is not None:
            out["keypoints"] = self.keypoints_head(dec)
        
        # Masked reconstruction branch (train-only or TTT)
        if return_mim and self.enable_mim and self.render_teacher is not None and self.decoder_mim is not None:
            # Use P5 feature (last encoder feature) for MIM
            neck_feat = encoder_feats[-1]  # (B, hidden_dim, H, W)
            
            # Apply masking if provided
            masked_feat = neck_feat
            mask_resized = None
            if feature_mask is not None:
                # Resize mask to match neck feature spatial dimensions
                if feature_mask.ndim == 2:
                    mask_input = feature_mask.unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)
                elif feature_mask.ndim == 3:
                    mask_input = feature_mask.unsqueeze(1).float()  # (B, 1, H, W)
                else:
                    mask_input = feature_mask.float()
                
                if mask_input.shape[-2:] != neck_feat.shape[-2:]:
                    mask_resized = F.interpolate(
                        mask_input,
                        size=neck_feat.shape[-2:],
                        mode='nearest'
                    )
                else:
                    mask_resized = mask_input
                
                # Expand to batch and channels
                mask_expanded = mask_resized.expand(batch, neck_feat.shape[1], -1, -1)
                masked_feat = neck_feat.masked_fill(mask_expanded.to(dtype=torch.bool), 0.0)
            
            # Generate teacher features from geometry
            teacher_feat = None
            if geom_input is not None:
                # Resize geom_input to match neck feature spatial dims if needed
                if geom_input.shape[-2:] != neck_feat.shape[-2:]:
                    geom_input = F.interpolate(
                        geom_input, 
                        size=neck_feat.shape[-2:], 
                        mode='bilinear', 
                        align_corners=False
                    )
                teacher_feat = self.render_teacher(geom_input).detach()  # Stop gradient
            
            # Reconstruct features
            recon_feat = self.decoder_mim(masked_feat)
            
            out["mim"] = {
                "recon_feat": recon_feat,
                "teacher_feat": teacher_feat,
                "neck_feat": neck_feat,
                "mask": mask_resized.squeeze() if mask_resized is not None else None,
            }
            
            # Compute entropy loss for geometric consistency
            if "logits" in out:
                entropy = entropy_loss(out["logits"])
                out["mim"]["entropy"] = entropy
        
        return out
