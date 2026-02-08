import torch
from typing import List, Union, Optional, Dict, Any, Callable
from diffusers.models.attention_processor import Attention, F
import math
import numpy as np

def scaled_dot_product_attention(query, key, value, fgmask, bgmask, attn_mask=None, dropout_p=0.0,
                is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    import torch.nn.functional as F
    import math

    if isinstance(fgmask, np.ndarray):
        fgmask = torch.from_numpy(fgmask).to(dtype=torch.float32, device=query.device)

    if isinstance(bgmask, np.ndarray):
        bgmask = torch.from_numpy(bgmask).to(dtype=torch.float32, device=query.device)

    L, S = query.size(-2), key.size(-2)
    a = int(L - 1241)
    a = int(a / 2)
    a = int(math.sqrt(a))
    b = a * 2
    c = (a * b)

    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)

    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_bias.masked_fill_(~temp_mask, float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(~attn_mask, float("-inf"))
        else:
            attn_bias += attn_mask

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight_old = attn_weight.clone()

    fgmask = fgmask.to(dtype=torch.float32, device=attn_weight.device)
    bgmask = bgmask.to(dtype=torch.float32, device=attn_weight.device)

    while fgmask.dim() < 4:
        fgmask = fgmask.unsqueeze(0)
    while bgmask.dim() < 4:
        bgmask = bgmask.unsqueeze(0)

    fgmask = fgmask.permute(0, 3, 1, 2)
    if fgmask.shape[1] != 1:
        fgmask = fgmask.mean(dim=1, keepdim=True)

    bgmask = bgmask.permute(0, 3, 1, 2)
    if bgmask.shape[1] != 1:
        bgmask = bgmask.mean(dim=1, keepdim=True)

    token_mask1 = F.interpolate(fgmask, size=(a, b), mode="nearest")
    token_mask1 = (token_mask1 > 0.5).float().reshape(fgmask.shape[0], -1).bool()

    token_mask2 = F.interpolate(bgmask, size=(a, b), mode="nearest")
    token_mask2 = (token_mask2 > 0.5).float().reshape(bgmask.shape[0], -1).bool()

    B = token_mask1.shape[0]

    mask_region1 = token_mask1.view(B, 1, c, 1).expand(B, attn_weight.size(1), c, 1241)
    boost_value = attn_weight.std().detach() * 0.1
    attn_block = attn_weight[:, :, -c:, :1241]
    attn_block = torch.where(mask_region1, attn_block + boost_value, attn_block)
    attn_weight[:, :, -c:, :1241] = attn_block

    attn_last = attn_weight[:, :, -c:, -c:]
    mask_region2 = token_mask2.view(B, 1, 1, c)
    row_cond = mask_region2.permute(0, 1, 3, 2)
    col_cond = ~mask_region2.expand(B, 1, 1, c)
    target_pos = row_cond & col_cond

    attn_last = torch.where(target_pos.expand(-1, attn_weight.size(1), -1, -1),
                            attn_last + boost_value * 10,
                            attn_last)
    attn_weight[:, :, -c:, -c:] = attn_last

    attn_weight = torch.softmax(attn_weight + attn_bias, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)

    return attn_weight @ value






def attn_forward(
    attn: Attention,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor = None,
    attn_control: bool = False,
    fgmask: Optional[torch.FloatTensor] = None,
    bgmask: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    image_rotary_emb: Optional[torch.Tensor] = None,
    model_config: Optional[Dict[str, Any]] = {},
) -> torch.FloatTensor:
    batch_size, _, _ = (
        hidden_states.shape
        if encoder_hidden_states is None
        else encoder_hidden_states.shape
    )
    #print(attn_control)

    
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
    if encoder_hidden_states is not None:
        # `context` projections.
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)

        if attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = attn.norm_added_q(
                encoder_hidden_states_query_proj
            )
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(
                encoder_hidden_states_key_proj
            )

        # attention
        # print(encoder_hidden_states_query_proj.shape)
        # print(query.shape)
        query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
        key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

    if image_rotary_emb is not None:
        from diffusers.models.embeddings import apply_rotary_emb

        query = apply_rotary_emb(query, image_rotary_emb)
        key = apply_rotary_emb(key, image_rotary_emb)


    if attn_control:
        if encoder_hidden_states is not None:
            hidden_states = scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                fgmask=fgmask,
                bgmask=bgmask,
                dropout_p=0.0,
                is_causal=False
            )
        else:
            #print(222)
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=False
            )
    else:
        #print(333)
        hidden_states = F.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=False
            )
    hidden_states = hidden_states.transpose(1, 2).reshape(
        batch_size, -1, attn.heads * head_dim
    )
    hidden_states = hidden_states.to(query.dtype)
    #print('ddd')

    if encoder_hidden_states is not None:

        encoder_hidden_states, hidden_states = (
            hidden_states[:, : encoder_hidden_states.shape[1]],
            hidden_states[:, encoder_hidden_states.shape[1] :],
        )


        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)


        return (
            (hidden_states, encoder_hidden_states)
        )

    else:
        return hidden_states


def block_forward(
    self,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor,
    temb: torch.FloatTensor,
    attn_control: bool = False,
    image_rotary_emb=None,
    fgmask: Optional[torch.FloatTensor] = None,
    bgmask: Optional[torch.FloatTensor] = None,
    model_config: Optional[Dict[str, Any]] = {},
):

    norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
        hidden_states, emb=temb
    )
    norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
        self.norm1_context(encoder_hidden_states, emb=temb)
    )

    # Attention.
    result = attn_forward(
        self.attn,
        model_config=model_config,
        hidden_states=norm_hidden_states,
        encoder_hidden_states=norm_encoder_hidden_states,
        image_rotary_emb=image_rotary_emb,
        fgmask=fgmask,
        bgmask=bgmask,
        attn_control=attn_control
    )
    attn_output, context_attn_output = result[:2]

    # Process attention outputs for the `hidden_states`.
    # 1. hidden_states
    attn_output = gate_msa.unsqueeze(1) * attn_output
    hidden_states = hidden_states + attn_output
    # 2. encoder_hidden_states
    context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
    encoder_hidden_states = encoder_hidden_states + context_attn_output


    # LayerNorm + MLP.
    # 1. hidden_states
    norm_hidden_states = self.norm2(hidden_states)
    norm_hidden_states = (
        norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
    )
    # 2. encoder_hidden_states
    norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
    norm_encoder_hidden_states = (
        norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
    )


    ff_output = self.ff(norm_hidden_states)
    ff_output = gate_mlp.unsqueeze(1) * ff_output
    # 2. encoder_hidden_states
    context_ff_output = self.ff_context(norm_encoder_hidden_states)
    context_ff_output = c_gate_mlp.unsqueeze(1) * context_ff_output


    # Process feed-forward outputs.
    hidden_states = hidden_states + ff_output
    encoder_hidden_states = encoder_hidden_states + context_ff_output


    # Clip to avoid overflow.
    if encoder_hidden_states.dtype == torch.float16:
        encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

    return encoder_hidden_states, hidden_states


def single_block_forward(
    self,
    hidden_states: torch.FloatTensor,
    temb: torch.FloatTensor,
    image_rotary_emb=None,
    model_config: Optional[Dict[str, Any]] = {},
):


    residual = hidden_states
    norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
    mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

    attn_output = attn_forward(
        self.attn,
        model_config=model_config,
        hidden_states=norm_hidden_states,
        image_rotary_emb=image_rotary_emb,

    )


    hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
    gate = gate.unsqueeze(1)
    hidden_states = gate * self.proj_out(hidden_states)
    hidden_states = residual + hidden_states

    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)

    return hidden_states