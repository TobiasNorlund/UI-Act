import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
import pydantic
from dataclasses import dataclass
from einops import rearrange
from torchmetrics import Accuracy, MeanMetric
from typing import Optional, List, Union
from collections import namedtuple
from enum import Enum


class ActionType(Enum):
    LEFT_CLICK = "left_click"
    RIGHT_CLICK = "right_click"
    DOUBLE_CLICK = "double_click"
    END = "end"


CursorPosition = namedtuple("CursorPosition", ("x", "y"))
Resolution = namedtuple("Resolution", ("width", "height"))


@pydantic.dataclasses.dataclass
class UIActModelConfig:

    # TextEncoder
    text_conditioned: bool
    text_num_embeddings: int 
    text_max_length: int
    text_hidden_dim: int
    text_num_heads: int
    text_ffn_dim: int
    text_num_layers: int
    text_dropout_p: float

    # FrameEncoder
    frame_input_channels: int
    frame_resolution: Resolution
    frame_base_feature_maps: int
    frame_residual_layers: int
    frame_feature_maps: int

    # ActionDecoder
    action_max_length: int
    action_hidden_dim: int
    action_num_heads: int
    action_ffn_dim: int
    action_num_layers: int
    action_dropout_p: float

    # Heads
    event_classes: list


@dataclass
class UIActModelInput:
    frames: torch.Tensor
    frames_attention_mask: torch.Tensor
    text: torch.Tensor = None
    text_attention_mask: torch.Tensor = None
    target_events: torch.Tensor = None
    target_cursor_positions: torch.Tensor = None


@dataclass
class UIActModelOutput:
    event_logits: torch.Tensor
    cursor_position_logits: torch.Tensor
    hidden_states: Union[torch.Tensor, List[torch.Tensor]]
    text_hidden_states: Union[torch.Tensor, List[torch.Tensor]]
    loss: Optional[torch.Tensor]
    event_loss: Optional[torch.Tensor]
    cursor_position_loss: Optional[torch.Tensor]


# Credit: https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py
class RMSNorm(torch.nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
        Root Mean Square Layer Normalization
        
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias
        self.scale = torch.nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)
        if self.bias:
            self.offset = torch.nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)
            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size
        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)
        if self.bias:
            return self.scale * x_normed + self.offset
        return self.scale * x_normed


class MultiHeadAttention(torch.nn.Module):

    def __init__(
        self,
        v_tot_dim: int,  # the dim of v, total for all heads
        hidden_dim: int,
        dropout_p: float
    ):
        super().__init__()
        self.attn_dropout = torch.nn.Dropout(dropout_p)
        self.resid_dropout = torch.nn.Dropout(dropout_p)
        self.final_proj = torch.nn.Linear(in_features=v_tot_dim, out_features=hidden_dim, bias=False)

    def forward(self, q, k, v, attention_mask):
        """
        Args:
          q:                [..., heads, q_len, qk_dim]
          k:                [..., heads, kv_len, qk_dim]
          v:                [..., heads, kv_len, v_dim]
          attention_mask:   [..., heads, q_len, kv_len]
        """
        attention_scores = q @ k.swapdims(-1, -2)  # - [..., heads, q_len, kv_len]
        attention_scores = attention_scores + attention_mask
        attention_scores = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_scores = self.attn_dropout(attention_scores)
        output = attention_scores @ v  # [..., heads, q_len, v_dim]

        # Concat heads
        output = rearrange(output, "... h l d -> ... l (h d)")

        # Final linear projection
        output = self.final_proj(output)  # [..., q_len, hidden_dim]

        # Final dropout
        output = self.resid_dropout(output)

        return output, attention_scores


class TextEncoderSelfAttention(torch.nn.Module):

    def __init__(self, config: UIActModelConfig):
        super().__init__()
        self.config = config
        self.pre_norm = RMSNorm(config.text_hidden_dim)
        self.qkv_proj = torch.nn.Linear(config.text_hidden_dim, config.text_hidden_dim * 3, bias=False)
        self.attention = MultiHeadAttention(config.text_hidden_dim, config.text_hidden_dim, config.text_dropout_p)

    def forward(
        self,
        hidden_states,
        attention_mask
    ):
        """
        Args:
          hidden_states:    [batch, seq_len, hidden_dim]
          attention_mask:   [batch, seq_len]
        """
        hidden_states = self.pre_norm(hidden_states)
        q, k, v = rearrange(self.qkv_proj(hidden_states), "b l (i h d) -> i b h l d", i=3, h=self.config.text_num_heads)
        attention_mask = rearrange(attention_mask, "b l -> b 1 1 l")  # Add broadcast dimensions for heads and queries
        attention_mask = (1.0 - attention_mask) * -10000.0  # Add -10000 on masked positions in score matrix
        hidden_states, attention_scores = self.attention(q, k, v, attention_mask)
        return hidden_states


class TextEncoderBlock(torch.nn.Module):

    def __init__(self, config: UIActModelConfig):
        super().__init__()
        self.self_attention = TextEncoderSelfAttention(config)
        self.ffn = torch.nn.Sequential(
            RMSNorm(config.text_hidden_dim),
            torch.nn.Linear(config.text_hidden_dim, config.text_ffn_dim),
            torch.nn.GELU(),
            torch.nn.Linear(config.text_ffn_dim, config.text_hidden_dim)
        )

    def forward(
        self,
        hidden_states,
        attention_mask
    ):
        """
        Args:
          hidden_states:    [batch, seq_len, hidden_dim]
          attention_mask:   [batch, seq_len]
        """
        # Self attention
        hidden_states = hidden_states + self.self_attention(
            hidden_states,
            attention_mask
        )
        # Feed-forward network
        hidden_states = hidden_states + self.ffn(
            hidden_states
        )
        return hidden_states


class TextEncoder(torch.nn.Module):

    def __init__(self, config: UIActModelConfig):
        super().__init__()
        self.token_embeddings = torch.nn.Embedding(config.text_num_embeddings, config.text_hidden_dim)
        self.pos_embeddings = torch.nn.Embedding(config.text_max_length, config.text_hidden_dim)

        self.blocks = torch.nn.ModuleList([
            TextEncoderBlock(config)
            for _ in range(config.text_num_layers)
        ])

    def forward(
        self,
        input_ids,
        attention_mask,
        return_all_hidden_states: bool=False
    ):
        """
        Args: 
          input_ids:        [batch, seq_len]
          attention_mask:   [batch, seq_len]
        """
        hidden_states = self.token_embeddings(input_ids)  # [batch, seq_len, hidden_dim]
        hidden_states += self.pos_embeddings(
            torch.arange(0, input_ids.shape[1], device=hidden_states.device)
        )

        if return_all_hidden_states:
            hidden_states = [hidden_states]

        for block in self.blocks:
            out = block(
                hidden_states if not return_all_hidden_states else hidden_states[-1], 
                attention_mask
            )
            if return_all_hidden_states:
                hidden_states.append(out)
            else:
                hidden_states = out

        return hidden_states


class FrameEncoderResidualBlock(torch.nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(channels, channels, (1, 3, 3), stride=(1, 1, 1), padding=(0, 2, 2), dilation=(1, 2, 2), bias=False)
        self.bn1 = torch.nn.BatchNorm3d(channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv3d(channels, channels, (1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=False)
        self.bn2 = torch.nn.BatchNorm3d(channels)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + identity  # Residual connection
        x = self.relu(x)
        return x


class FrameEncoder(torch.nn.Module):

    def __init__(self, config: UIActModelConfig):
        super().__init__()
        num_downsample_layers = 4

        self.base = torch.nn.Sequential(
            torch.nn.Conv3d(config.frame_input_channels, config.frame_base_feature_maps, (1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False),
            torch.nn.BatchNorm3d(config.frame_base_feature_maps),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        self.residual_blocks = torch.nn.Sequential(*(
            FrameEncoderResidualBlock(config.frame_base_feature_maps)
            for _ in range(config.frame_residual_layers)
        ))
        self.downsample_layers = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv3d(
                    config.frame_base_feature_maps if layer_idx == 0 else config.frame_feature_maps, 
                    config.frame_feature_maps, 
                    kernel_size=(1, 3, 3), 
                    stride=(1, 2, 2), 
                    padding=(0, 1, 1), 
                    bias=False
                ),
                torch.nn.BatchNorm3d(config.frame_feature_maps),
                torch.nn.ReLU(inplace=True)
            )
            for layer_idx in range(num_downsample_layers)
        ])
        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, frames):
        """
        Args:
          frames:       [batch, num_frames, height, width, input_channels]
        
        Returns:
          output:       [batch, num_frames, feature_maps]
          base_output:  [batch, num_frames, height/4, width/4, feature_maps]
        """
        x = frames.type(torch.float32) / 255.0 - 0.5
        x = rearrange(x, "b t h w c -> b c t h w")
        x = self.base(x)
        residual_output = self.residual_blocks(x)
        x = residual_output
        for layer in self.downsample_layers:
            x = layer(x)
        x = self.avgpool(x)
        x = rearrange(x, "b c t 1 1 -> b t c")
        return x, rearrange(residual_output, "b c t h w -> b t h w c")


class ActionDecoderSelfAttention(torch.nn.Module):

    def __init__(self, config: UIActModelConfig):
        super().__init__()
        self.config = config
        self.pre_norm = RMSNorm(config.action_hidden_dim)
        self.qkv_proj = torch.nn.Linear(config.action_hidden_dim, config.action_hidden_dim * 3, bias=False)
        self.attention = MultiHeadAttention(config.action_hidden_dim, config.action_hidden_dim, config.action_dropout_p)

    def forward(self, hidden_states, attention_mask):
        """
        Args:
          hidden_states:    [batch, seq_len, hidden_dim]
          attention_mask:   [batch, seq_len]
        """
        hidden_states = self.pre_norm(hidden_states)
        q, k, v = rearrange(self.qkv_proj(hidden_states), "b l (i h d) -> i b h l d", i=3, h=self.config.action_num_heads)

        attention_mask = rearrange(attention_mask, "b l -> b 1 1 l")  # Add broadcast dimensions for heads and queries
        attention_mask = (1.0 - attention_mask) * -10000.0  # Add -10000 on masked positions in score matrix
        
        # Apply causal mask
        length = q.size(-2)
        causal_mask = (1.0 - torch.tril(torch.ones((length, length), dtype=torch.uint8, device=attention_mask.device)).view(1, 1, length, length)) * -10000.0
        attention_mask = attention_mask + causal_mask

        hidden_states, attention_scores = self.attention(q, k, v, attention_mask)
        return hidden_states


class ActionDecoderCrossAttention(torch.nn.Module):

    def __init__(self, hidden_dim: int, ca_hidden_dim: int, num_heads: int, dropout_p: float):
        super().__init__()
        self.num_heads = num_heads
        self.pre_norm = RMSNorm(hidden_dim)
        self.q_proj = torch.nn.Linear(hidden_dim, ca_hidden_dim, bias=False)
        self.kv_proj = torch.nn.Linear(ca_hidden_dim, ca_hidden_dim * 2, bias=False)
        self.attention = MultiHeadAttention(ca_hidden_dim, hidden_dim, dropout_p)

    def forward(
        self, 
        hidden_states, 
        ca_hidden_states, 
        ca_attention_mask
    ):
        """
        Args:
          hidden_states:        [..., seq_len, hidden_dim]
          ca_hidden_states:     [..., ca_seq_len, ca_hidden_dim]
          ca_attention_mask:    [..., ca_seq_len]
        """
        hidden_states = self.pre_norm(hidden_states)

        q = rearrange(self.q_proj(hidden_states), "... l (h d) -> ... h l d", h=self.num_heads)
        k, v = rearrange(self.kv_proj(ca_hidden_states), "... l (i h d) -> i ... h l d", i=2, h=self.num_heads)

        ca_attention_mask = rearrange(ca_attention_mask, "... l -> ... 1 1 l")  # Add broadcast dimensions for heads and queries
        ca_attention_mask = (1.0 - ca_attention_mask) * -10000.0  # Add -10000 on masked positions in score matrix
        
        hidden_states, attention_scores = self.attention(q, k, v, ca_attention_mask)
        return hidden_states


class ActionDecoderBlock(torch.nn.Module):

    def __init__(self, config: UIActModelConfig):
        super().__init__()
        self.config = config
        self.self_attention = ActionDecoderSelfAttention(config)
        if config.text_conditioned:
            self.text_cross_attention = ActionDecoderCrossAttention(
                hidden_dim=config.action_hidden_dim, 
                ca_hidden_dim=config.text_hidden_dim, 
                num_heads=config.action_num_heads,
                dropout_p=config.action_dropout_p
            )
        self.frame_cross_attention = ActionDecoderCrossAttention(
            hidden_dim=config.action_hidden_dim,
            ca_hidden_dim=config.frame_base_feature_maps,
            num_heads=config.action_num_heads,
            dropout_p=config.action_dropout_p
        )
        self.ffn = torch.nn.Sequential(
            RMSNorm(config.action_hidden_dim),
            torch.nn.Linear(config.action_hidden_dim, config.action_ffn_dim),
            torch.nn.GELU(),
            torch.nn.Linear(config.action_ffn_dim, config.action_hidden_dim)
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        encoded_text,
        text_attention_mask,
        frame_feature_maps,
    ):
        """
        Args:
          hidden_states:        [batch, num_frames, frame_dim]
          attention_mask:       [batch, num_frames]
          encoded_text:         [batch, text_len, text_dim]
          text_attention_mask:  [batch, text_len]
          frame_feature_maps:   [batch, num_frames, h*w, feature_maps]
        """
        hidden_states = hidden_states + self.self_attention(
            hidden_states,
            attention_mask
        )
        if self.config.text_conditioned:
            hidden_states = hidden_states + self.text_cross_attention(
                hidden_states,
                encoded_text,
                text_attention_mask
            )
        hidden_states = hidden_states + self.frame_cross_attention(
            hidden_states=rearrange(hidden_states, "b l d -> b l 1 d"),
            ca_hidden_states=frame_feature_maps,
            ca_attention_mask=torch.ones(frame_feature_maps.shape[:3], device=hidden_states.device)
        ).squeeze(2)
        hidden_states = hidden_states + self.ffn(
            hidden_states
        )
        return hidden_states


class UIActModel(torch.nn.Module):

    def __init__(self, config: UIActModelConfig):
        super().__init__()
        self.config = config

        self.text_encoder = TextEncoder(config) if config.text_conditioned else None
        self.frame_encoder = FrameEncoder(config)
        #self.frame_feature_maps_abs_pos_encodings = torch.nn.Parameter(
        #    torch.randn((1, 1, config.frame_resolution.height//4, config.frame_resolution.width//4, config.frame_base_feature_maps)) * 0.01
        #)
        self.frame_feature_maps_pos_encodings = torch.nn.Parameter(
            torch.randn((1, 1, config.frame_resolution.height//4, config.frame_resolution.width//4, config.frame_base_feature_maps)) * 0.01
        )

        #self.frame_proj = torch.nn.Linear(config.frame_feature_maps + config.frame_base_feature_maps + len(config.event_classes), config.action_hidden_dim, bias=False)
        self.frame_proj = torch.nn.Linear(config.frame_feature_maps, config.action_hidden_dim, bias=False)
        self.action_pos_embeddings = torch.nn.Parameter(torch.randn(1, config.action_max_length, config.action_hidden_dim) * 0.01)
        self.action_decoder = torch.nn.ModuleList([
            ActionDecoderBlock(config)
            for _ in range(config.action_num_layers)
        ])

        self.event_head = torch.nn.Linear(config.action_hidden_dim, len(config.event_classes), bias=False)
        self.cursor_position_head = torch.nn.Linear(config.action_hidden_dim, config.frame_base_feature_maps, bias=False)

        self._left_click_idx = self.config.event_classes.index(ActionType.LEFT_CLICK.value) if ActionType.LEFT_CLICK.value in self.config.event_classes else -1
        self._right_click_idx = self.config.event_classes.index(ActionType.RIGHT_CLICK.value) if ActionType.RIGHT_CLICK.value in self.config.event_classes else -1

    def _get_cursor_position_target_indices(self, target_cursor_positions, frames_width, target_events, frames_attention_mask) -> torch.Tensor:
        target_cursor_positions = (target_cursor_positions / 4).round().long()  # get coordinates in feature maps
        feature_maps_width = int(frames_width / 4)
        targets = target_cursor_positions[:, :, 1] * feature_maps_width + target_cursor_positions[:, :, 0]  # get flat height x width index
        targets[((target_events != self._left_click_idx) & (target_events != self._right_click_idx)) | (frames_attention_mask == 0)] = -100  # Only apply for click events
        return targets

    def forward(
        self,
        frames,
        frames_attention_mask,
        text=None,
        text_attention_mask=None,
        target_events=None,
        target_cursor_positions=None,
        return_all_hidden_states: bool=False,
        return_all_text_hidden_states: bool=False
    ):
        """
        Args:
          frames:                   [batch, num_frames, height, width, input_channels]
          frames_attention_mask:    [batch, num_frames]
          text:                     [batch, text_len]
          text_attention_mask:      [batch, text_len]
          target_events:            [batch, num_frames]
          target_cursor_positions:  [batch, num_frames, 2]
        """

        # Encode text (optionally)
        if self.config.text_conditioned:
            encoded_text = self.text_encoder(
                text, 
                text_attention_mask, 
                return_all_hidden_states=return_all_text_hidden_states
            )
            # encoded_text:     [batch, text_len, text_dim] (potentially list of tensors)
        else:
            encoded_text = None

        # Encode frames
        encoded_frames, frame_feature_maps = self.frame_encoder(frames)
        #  encoded_frames       [batch, num_frames, frame_dim]
        #  frame_feature_maps   [batch, num_frames, height/4, width/4, frame_base_feature_maps]

        # Add absolute positional embeddings
        frame_feature_maps = frame_feature_maps + self.frame_feature_maps_pos_encodings
        frame_feature_maps = rearrange(frame_feature_maps, "b l h w d -> b l (h w) d")

        frame_reps = encoded_frames

        # Action decoder
        hidden_states = self.frame_proj(frame_reps)
        hidden_states = hidden_states + self.action_pos_embeddings[:, :hidden_states.shape[1], :]
        if return_all_hidden_states:
            hidden_states = [hidden_states]
        for layer in self.action_decoder:
            out = layer(
                hidden_states if not return_all_hidden_states else hidden_states[-1],
                frames_attention_mask,
                encoded_text if not return_all_text_hidden_states else encoded_text[-1],
                text_attention_mask,
                frame_feature_maps
            )
            if return_all_hidden_states:
                hidden_states.append(out)
            else:
                hidden_states = out

        event_logits = self.event_head(hidden_states if not return_all_hidden_states else hidden_states[-1])
        # event_logits:  [batch, num_frames, num_events]
       
        cursor_pos_logits = self.cursor_position_head(hidden_states if not return_all_hidden_states else hidden_states[-1])
        cursor_pos_logits = frame_feature_maps @ cursor_pos_logits[:, :, :, None]
        cursor_pos_logits = rearrange(cursor_pos_logits, "b l hw 1 -> b l hw")
        # cursor_pos_logits  [batch, num_frames, frame_base_feature_maps]

        event_loss = None
        cursor_position_loss = None

        # Event classification head
        if target_events is not None:
            event_loss = torch.nn.functional.cross_entropy(
                event_logits.view(-1, event_logits.shape[-1]), 
                target_events.view(-1)
            )

        # Cursor position head
        if target_cursor_positions is not None:
            targets = self._get_cursor_position_target_indices(
                target_cursor_positions=target_cursor_positions,
                frames_width=frames.shape[3],
                target_events=target_events,
                frames_attention_mask=frames_attention_mask
            )
            cursor_position_loss = torch.nn.functional.cross_entropy(
                cursor_pos_logits.view(-1, cursor_pos_logits.shape[-1]),
                targets.view(-1)
            )

        return UIActModelOutput(
            event_logits=event_logits,
            cursor_position_logits=cursor_pos_logits,
            hidden_states=hidden_states,
            text_hidden_states=encoded_text,
            event_loss=event_loss,
            cursor_position_loss=cursor_position_loss,
            loss=((event_loss or 0) + (cursor_position_loss or 0)) or None,
        )


class UIActLightningModel(UIActModel, pl.LightningModule):

    def __init__(self, config: UIActModelConfig):
        super().__init__(config)
        self.save_hyperparameters()
        num_cursor_pos = config.frame_resolution.width * config.frame_resolution.height // (4 * 4)
        self.train_event_accuracy = Accuracy("multiclass", num_classes=2, ignore_index=-100)
        self.train_cursor_pos_accuracy = Accuracy("multiclass", num_classes=num_cursor_pos, ignore_index=-100)
        self.val_event_accuracy = Accuracy("multiclass", num_classes=2, ignore_index=-100)
        self.val_cursor_pos_accuracy = Accuracy("multiclass", num_classes=num_cursor_pos, ignore_index=-100)
        self.val_l2_error = MeanMetric()
        self.validation_step_outputs = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def training_step(self, batch: UIActModelInput, _):
        output = self.forward(
            frames=batch.frames, 
            frames_attention_mask=batch.frames_attention_mask, 
            text=batch.text, 
            text_attention_mask=batch.text_attention_mask, 
            target_events=batch.target_events, 
            target_cursor_positions=batch.target_cursor_positions
        )

        self.train_event_accuracy(
            output.event_logits.view(-1, output.event_logits.shape[-1]),
            batch.target_events.view(-1)
        )
        self.log("train/event_accuracy", self.train_event_accuracy, on_epoch=True, on_step=False, sync_dist=True)
        self.train_cursor_pos_accuracy(
            output.cursor_position_logits.view(-1, output.cursor_position_logits.shape[-1]),
            self._get_cursor_position_target_indices(batch.target_cursor_positions, batch.frames.shape[3], batch.target_events, batch.frames_attention_mask).view(-1)
        )
        self.log("train/cursor_position_accuracy", self.train_cursor_pos_accuracy, on_epoch=True, on_step=False, sync_dist=True)
        self.log("train/event_loss", output.event_loss)
        self.log("train/cursor_position_loss", output.cursor_position_loss)
        self.log("train/loss", output.loss)
        return output.loss

    def validation_step(self, batch: UIActModelInput, batch_idx):
        output = self.forward(
            frames=batch.frames, 
            frames_attention_mask=batch.frames_attention_mask, 
            text=batch.text, 
            text_attention_mask=batch.text_attention_mask, 
            target_events=batch.target_events, 
            target_cursor_positions=batch.target_cursor_positions
        )

        # log predicted cursor_position 
        if batch_idx == 0 and self.logger is not None:
            self.log_cursor_position_predictions(batch, output)

        # l2_error
        pred_pos_flat = output.cursor_position_logits.argmax(dim=-1)  # [batch, num_frames]
        pred_pos = torch.zeros(
            (pred_pos_flat.shape[0], pred_pos_flat.shape[1], 2),
            dtype=pred_pos_flat.dtype,
            device=pred_pos_flat.device
        )
        width = self.config.frame_resolution.width // 4
        pred_pos[..., 0] = pred_pos_flat % width
        pred_pos[..., 1] = pred_pos_flat // width
        l2_error = torch.sqrt(torch.sum((batch.target_cursor_positions/4 - pred_pos) ** 2, dim=-1))
        # we only want to compute l2_error on click frames
        click_frames = (((batch.target_events == self._left_click_idx) | (batch.target_events == self._right_click_idx)) & batch.frames_attention_mask).type(torch.bool)
        non_click_frames = torch.bitwise_not(click_frames)
        # set l2_error = -1 for frames that are not click_frames
        l2_error[non_click_frames] = -1

        self.val_l2_error.update(l2_error[click_frames])
        self.log("val/l2_error", self.val_l2_error, on_epoch=True)

        self.val_event_accuracy(
            output.event_logits.view(-1, output.event_logits.shape[-1]),
            batch.target_events.view(-1)
        )
        self.log("val/event_accuracy", self.val_event_accuracy)
        self.val_cursor_pos_accuracy(
            output.cursor_position_logits.view(-1, output.cursor_position_logits.shape[-1]),
            self._get_cursor_position_target_indices(batch.target_cursor_positions, batch.frames.shape[3], batch.target_events, batch.frames_attention_mask).view(-1)
        )
        self.log("val/cursor_position_accuracy", self.val_cursor_pos_accuracy)
        self.log("val/event_loss", output.event_loss, sync_dist=True)
        self.log("val/cursor_position_loss", output.cursor_position_loss, sync_dist=True)
        self.log("val/loss", output.loss, sync_dist=True)

        self.validation_step_outputs.append((output, l2_error))
        return output, l2_error
    
    def on_validation_epoch_end(self):
        if self.logger is not None:
            all_l2_errors = []
            for _, l2_error in self.validation_step_outputs:
                all_l2_errors.append(l2_error[l2_error >= 0.0])
            all_l2_errors = torch.concatenate(all_l2_errors).detach().cpu().numpy()
            self.logger.experiment.log({"val/l2_error_dist": wandb.Histogram(all_l2_errors)}, commit=False)
        self.validation_step_outputs.clear()

    def log_cursor_position_predictions(self, batch: UIActModelInput, output: UIActModelOutput):
        click_pos = torch.nn.functional.softmax(output.cursor_position_logits[0], dim=-1).cpu().detach().numpy()
        click_pos = click_pos.reshape(click_pos.shape[0], self.config.frame_resolution.height//4, self.config.frame_resolution.width//4)
        figs = []
        for i in range(batch.frames.shape[1]):
            fig = plt.figure(figsize=(10,6))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis('off')
            alpha = np.kron(click_pos[i], np.ones((4, 4)))[:, :, None]
            alpha /= alpha.max()
            alpha = alpha * 0.8 + 0.2
            frame = batch.frames[0, i, :, :, :3].cpu().detach().numpy() / 255.
            frame = np.concatenate([frame, alpha], axis=-1)
            ax.imshow(np.zeros((batch.frames.shape[2], batch.frames.shape[3], 3), dtype=np.byte))
            ax.imshow(frame)
            ax.scatter(batch.target_cursor_positions[0, i, 0].item(), batch.target_cursor_positions[0, i, 1].item(), s=10, c='red', marker='o')
            figs.append(fig)
        self.logger.experiment.log({f"val/frame_{i}": fig for i, fig in enumerate(figs)}, commit=False)
        for fig in figs:
            plt.close(fig)
