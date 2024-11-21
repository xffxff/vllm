import math
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import LlamaConfig
from transformers.utils import logging
from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, LoRAConfig, VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.inputs import INPUT_REGISTRY, token_inputs
from vllm.model_executor.layers.fused_moe import fused_moe
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput, SamplingMetadata
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaMLP,
    LlamaModel,
    RMSNorm,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    make_layers,
    maybe_prefix,
    merge_multimodal_embeddings,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.base import MultiModalInputs
from vllm.multimodal.image import cached_get_image_processor
from vllm.multimodal.utils import (
    cached_get_tokenizer,
    repeat_and_pad_placeholder_tokens,
)
from vllm.sequence import IntermediateTensors

logger = logging.get_logger(__name__)

from torch.nn.init import trunc_normal_
from vllm.config import QuantizationConfig
from vllm.model_executor.models.idefics2_vision_model import Idefics2VisionTransformer
from vllm.transformers_utils.configs.aria import AriaVisionConfig, AriaMoELMConfig
from vllm.model_executor.layers.activation import get_act_fn


class AriaVisionTransformer(Idefics2VisionTransformer):

    def __init__(
        self,
        config: AriaVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config, prefix)
        self.post_layernorm = nn.Identity()


class AriaVisionModel(nn.Module):
    config_class = AriaVisionConfig

    def __init__(
        self,
        config: AriaVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.vision_model = AriaVisionTransformer(
            config,
            quant_config,
            prefix=f"{prefix}.vision_model",
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.BoolTensor]]:
        patch_attention_mask = self._create_patch_attention_mask(pixel_mask)

        vit_oup = self.vision_model(
            pixel_values=pixel_values,
            patch_attention_mask=patch_attention_mask,
        )

        image_atts = self._create_image_attention_mask(patch_attention_mask)

        return vit_oup, image_atts

    def _create_patch_attention_mask(self, pixel_mask):
        if pixel_mask is None:
            return None

        patches_subgrid = pixel_mask.unfold(
            dimension=1,
            size=self.vision_model.config.patch_size,
            step=self.vision_model.config.patch_size,
        ).unfold(
            dimension=2,
            size=self.vision_model.config.patch_size,
            step=self.vision_model.config.patch_size,
        )
        return (patches_subgrid.sum(dim=(-1, -2)) > 0).bool()

    def _create_image_attention_mask(self, patch_attention_mask):
        if patch_attention_mask is None:
            return None

        flattened_mask = patch_attention_mask.flatten(1)
        return torch.logical_not(flattened_mask)


class FFN(nn.Module):
    """
    Feed-Forward Network module.

    Args:
        embed_dim (int): Input embedding dimension.
        ff_dim (int): Hidden dimension of the feed-forward network.
        output_dim (int): Output dimension.
    """

    def __init__(self, embed_dim, ff_dim, output_dim):
        super().__init__()
        self.linear_in = ColumnParallelLinear(embed_dim, ff_dim, bias=False)
        self.linear_out = RowParallelLinear(ff_dim, output_dim, bias=False)
        self.act = get_act_fn("gelu_new")

    def forward(self, hidden_states):
        hidden_states, _ = self.linear_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.linear_out(hidden_states)
        return hidden_states


class CrossAttention(nn.Module):
    """
    Cross-Attention module.

    Args:
        kv_dim (int): Dimension of key and value.
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        drop_out_rate (float): Dropout rate. Default is 0.
    """

    def __init__(self, kv_dim, embed_dim, num_heads, drop_out_rate=0):
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(kv_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(kv_dim, embed_dim, bias=False)

        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(drop_out_rate)

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.ln_kv = nn.LayerNorm(kv_dim)

    def forward(self, x, hidden_states, attn_mask=None, add_residual=False):
        """
        Forward pass of the CrossAttention module.

        Args:
            x (torch.Tensor): Input tensor for key and value.
            hidden_states (torch.Tensor): Input tensor for query.
            attn_mask (torch.Tensor, optional): Attention mask. Default is None.
            add_residual (bool): Whether to add residual connection. Default is False.

        Returns:
            torch.Tensor: Output tensor after cross-attention.
        """
        normed_hidden_states = self.layer_norm(hidden_states)
        query = self.q_proj(normed_hidden_states).permute(1, 0, 2)

        x = self.ln_kv(x)
        key = self.k_proj(x).permute(1, 0, 2)
        value = self.v_proj(x).permute(1, 0, 2)

        attn_output, _ = self.multihead_attn(query,
                                             key,
                                             value,
                                             attn_mask=attn_mask)

        attn_output = attn_output.permute(1, 0, 2)

        if add_residual:
            attn_output = hidden_states + self.dropout(
                self.linear(attn_output))
        else:
            attn_output = self.dropout(self.linear(attn_output))

        return attn_output


class AriaProjector(nn.Module):
    """
    A projection module with one cross attention layer and one FFN layer, which projects ViT's outputs into MoE's inputs.

    Args:
        patch_to_query_dict (dict): Maps patch numbers to their corresponding query numbers,
            e.g., {1225: 128, 4900: 256}. This allows for different query sizes based on image resolution.
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        kv_dim (int): Dimension of key and value.
        ff_dim (int): Hidden dimension of the feed-forward network.
        output_dim (int): Output dimension.
        norm_layer (nn.Module): Normalization layer. Default is nn.LayerNorm.

    Outputs:
        A tensor with the shape of (batch_size, query_number, output_dim)
    """

    def __init__(
        self,
        patch_to_query_dict,
        embed_dim,
        num_heads,
        kv_dim,
        ff_dim,
        output_dim,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.patch_to_query_dict = patch_to_query_dict
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.query = nn.Parameter(
            torch.zeros(max(patch_to_query_dict.values()), self.embed_dim))

        trunc_normal_(self.query, std=0.02)

        self.cross_attn = CrossAttention(kv_dim, embed_dim, num_heads)

        self.ln_ffn = norm_layer(embed_dim)
        self.ffn = FFN(embed_dim, ff_dim, output_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, attn_mask=None):
        """
        Forward pass of the Projector module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_patches, kv_dim).
            attn_mask (torch.Tensor, optional): Attention mask. Default is None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, query_number, output_dim).
        """
        bs = x.shape[0]
        queries = self.query.unsqueeze(0).repeat(bs, 1, 1)

        query_num = self.patch_to_query_dict.get(x.shape[1], None)
        assert (query_num is not None
                ), f"Query number for {x.shape[1]} patches is not provided"

        queries = queries[:, :query_num, :]

        if attn_mask is not None:
            attn_mask = attn_mask.repeat_interleave(self.num_heads, 0)
            attn_mask = attn_mask.unsqueeze(1).expand(-1, queries.size(1), -1)

        attention_out = self.cross_attn(x, queries, attn_mask=attn_mask)

        out = self.ffn(self.ln_ffn(attention_out))

        return out


class Experts(nn.Module):

    def __init__(self, config: AriaMoELMConfig):
        super().__init__()
        self.config = config

        self.router_weight = nn.Parameter(
            torch.empty(
                (self.config.moe_num_experts, self.config.hidden_size)))

        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        if self.tp_size > config.moe_num_experts:
            raise ValueError(
                f"Tensor model parallel size {self.tp_size} is greater than the number of experts {config.moe_num_experts}"
            )

        self.w1 = nn.Parameter(
            torch.empty((
                config.moe_num_experts,
                config.moe_intermediate_size * 2 // self.tp_size,
                config.hidden_size,
            )))
        self.w2 = nn.Parameter(
            torch.empty((
                config.moe_num_experts,
                config.hidden_size,
                config.moe_intermediate_size // self.tp_size,
            )))
        set_weight_attrs(self.router_weight,
                         {"weight_loader": self._weight_loader_for_router})
        set_weight_attrs(self.w1,
                         {"weight_loader": self._weight_loader_for_w1})
        set_weight_attrs(self.w2,
                         {"weight_loader": self._weight_loader_for_w2})

    def _weight_loader_for_router(self, param: nn.Parameter,
                                  loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def _weight_loader_for_w1(self, param: nn.Parameter,
                              loaded_weight: torch.Tensor):
        # the shape of loaded_weight is (num_experts, hidden_size, 2 * moe_intermediate_size)
        if self.tp_size > 1:
            up, gate = loaded_weight.chunk(2, dim=-1)
            up_current_rank = up.chunk(self.tp_size, dim=-1)[self.tp_rank]
            gate_current_rank = gate.chunk(self.tp_size, dim=-1)[self.tp_rank]
            up_and_gate = torch.cat([up_current_rank, gate_current_rank],
                                    dim=-1).transpose(1, 2)
            param.data.copy_(up_and_gate)
        else:
            param.data.copy_(loaded_weight.transpose(1, 2))

    def _weight_loader_for_w2(self, param: nn.Parameter,
                              loaded_weight: torch.Tensor):
        # the shape of loaded_weight is (num_experts, moe_intermediate_size, hidden_size)
        if self.tp_size > 1:
            down_current_rank = loaded_weight.chunk(self.tp_size,
                                                    dim=1)[self.tp_rank]
            param.data.copy_(down_current_rank.transpose(1, 2))
        else:
            param.data.copy_(loaded_weight.transpose(1, 2))

    def forward(self, hidden_states):
        router_output = torch.nn.functional.linear(hidden_states,
                                                   self.router_weight)

        def custom_routing_function(hidden_states, router_output, topk,
                                    renormalize):
            top_logits, top_indices = torch.topk(router_output,
                                                 k=self.config.moe_topk,
                                                 dim=1)
            scores = torch.softmax(top_logits, dim=-1, dtype=torch.float32)
            return scores, top_indices.to(torch.int32)

        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
        final_hidden_states = fused_moe(
            hidden_states,
            self.w1,
            self.w2,
            router_output,
            self.config.moe_topk,
            False,
            inplace=True,
            custom_routing_function=custom_routing_function,
        )
        final_hidden_states = final_hidden_states.view(hidden_states_shape)
        final_hidden_states = tensor_model_parallel_all_reduce(
            final_hidden_states)
        return final_hidden_states


class MoELayer(nn.Module):
    """
    Mixture of Experts (MoE) Layer for the AriaMoE model.

    This layer implements the MoE mechanism, which routes input tokens to different experts
    based on a routing algorithm, processes them through the experts, and then combines
    the outputs.
    """

    def __init__(
        self,
        config: AriaMoELMConfig,
        quant_config: Optional[QuantizationConfig],
        lora_config: Optional[LoRAConfig],
    ) -> None:
        super().__init__()
        self.config = config

        self.experts = Experts(config)
        self.shared_experts = LlamaMLP(
            config.hidden_size,
            config.moe_intermediate_size * config.moe_num_shared_experts,
            "silu",
            quant_config=quant_config,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MoE Layer.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape (batch_size, sequence_length, hidden_size).

        Returns:
            torch.Tensor: Output tensor after passing through the MoE layer.
        """

        shared_expert_output = self.shared_experts(hidden_states)
        sparse_expert_output = self.experts(hidden_states)

        return sparse_expert_output + shared_expert_output


class MoEDecoderLayer(LlamaDecoderLayer):
    """
    Custom Decoder Layer for the AriaMoE model which modifies the standard `LlamaDecoderLayer` by
    replacing the traditional MLP with a Mixture of Experts (MoE) Layer.
    """

    def __init__(
        self,
        config: LlamaConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
                config, "original_max_position_embeddings", None):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        # Support abacusai/Smaug-72B-v0.1 with attention_bias
        # Support internlm/internlm-7b with bias
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False)
        self.self_attn = LlamaAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads",
                                 config.num_attention_heads),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = MoELayer(config,
                            quant_config=quant_config,
                            lora_config=lora_config)
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)


class AriaMoELMModel(LlamaModel):
    """
    Custom LlamaModel for the AriaMoE model which modifies the standard LlamaModel by
    replacing the `LlamaDecoderLayer` with `MoEDecoderLayer`.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        # FIXME: this is a hack to disable the compilation of the model
        self.do_not_compile = True

        self.layers = None

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: MoEDecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )


def build_mm_projector(config):
    """
    Builds and returns an AriaProjector instance based on the provided configuration.

    Args:
        config (AriaConfig): The configuration object containing necessary parameters.

    Returns:
        AriaProjector: An instance of the AriaProjector class.
    """
    return AriaProjector(
        patch_to_query_dict=config.projector_patch_to_query_dict,
        embed_dim=config.vision_config.hidden_size,
        num_heads=config.vision_config.num_attention_heads,
        kv_dim=config.vision_config.hidden_size,
        ff_dim=config.text_config.hidden_size,
        output_dim=config.text_config.hidden_size,
    )


def get_max_multimodal_tokens(ctx):
    return max(ctx.model_config.hf_config.image_size2tokens.values())


def input_mapper_for_aria(ctx, data):
    return MultiModalInputs(data)


def repeat_image_tokens(token_ids: list, image_token_id: int,
                        repeat_times: list) -> list:
    """
    Repeats the image token in the token_ids list according to the repeat_times list.

    Args:
        token_ids (list): List of token IDs.
        image_token_id (int): The token ID that represents an image.
        repeat_times (list): List of integers specifying how many times to repeat the image token.

    Returns:
        list: A new list with the image token repeated as specified.

    Example:
        token_ids = [1, 2, 3, 4, 3, 5]
        image_token_id = 3
        repeat_times = [2, 3]
        result = repeat_image_tokens(token_ids, image_token_id, repeat_times)
        # result will be [1, 2, 3, 3, 4, 3, 3, 3, 5]
    """
    if len(repeat_times) != token_ids.count(image_token_id):
        raise ValueError(
            "The length of repeat_times is not equal to the number of images.")

    result = []
    repeat_iter = iter(repeat_times)

    for x in token_ids:
        if x == image_token_id:
            result.extend([image_token_id] * next(repeat_iter))
        else:
            result.append(x)

    return result


def input_processor(ctx, llm_inputs):
    multi_modal_data = llm_inputs.get("multi_modal_data")
    # if it is pure text input, use it as is
    if multi_modal_data is None or "image" not in multi_modal_data:
        return llm_inputs

    model_config = ctx.model_config

    tokenizer = cached_get_tokenizer(model_config.tokenizer)
    image_processor = cached_get_image_processor(
        model_config.model, trust_remote_code=model_config.trust_remote_code)
    hf_config = model_config.hf_config

    # prepare image tokens, the max_image_size is used to determine the number of patch_size for every image
    max_image_size = multi_modal_data.pop("max_image_size", 980)
    _split_image = multi_modal_data.pop("split_image", False)

    assert isinstance(max_image_size, (int, float)), "max_image_size should be float or int"
    images = (multi_modal_data["image"] if isinstance(
        multi_modal_data["image"], list) else [multi_modal_data["image"]])

    image_inputs = image_processor.preprocess(images,
                                              max_image_size=max_image_size,
                                              split_image=_split_image,
                                              return_tensors="pt").data
    num_crops = image_inputs.pop("num_crops")

    prompt_token_ids = llm_inputs["prompt_token_ids"]
    prompt_token_ids = repeat_image_tokens(prompt_token_ids,
                                           hf_config.image_token_index,
                                           num_crops)

    repeat_count = [hf_config.image_size2tokens[max_image_size]
                    ] * sum(num_crops).item()
    new_prompt, new_token_ids, _ = repeat_and_pad_placeholder_tokens(
        tokenizer,
        None,
        prompt_token_ids,
        placeholder_token_id=hf_config.image_token_index,
        repeat_count=repeat_count,
    )

    return token_inputs(
        prompt_token_ids=new_token_ids,
        prompt=new_prompt,
        multi_modal_data={"image": image_inputs},
    )


@MULTIMODAL_REGISTRY.register_max_image_tokens(get_max_multimodal_tokens)
@MULTIMODAL_REGISTRY.register_image_input_mapper(input_mapper_for_aria)
@INPUT_REGISTRY.register_input_processor(input_processor)
class AriaForConditionalGeneration(nn.Module, SupportsMultiModal):
    """
    Aria model for conditional generation tasks.

    This model combines a vision tower, a multi-modal projector, and a language model
    to perform tasks that involve both image and text inputs.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        # prepare the image_size to tokens mapping for the image preprocess, see input_processor
        config.image_size2tokens = {
            int(math.sqrt(k) * config.vision_config.patch_size): v
            for k, v in config.projector_patch_to_query_dict.items()
        }
        self.config = config
        self.vision_tower = AriaVisionModel(config.vision_config)
        self.multi_modal_projector = build_mm_projector(config)
        self.vocab_size = config.text_config.vocab_size
        self.language_model = AriaMoELMModel(
            vllm_config=vllm_config.with_hf_config(config.text_config),
            prefix=maybe_prefix(prefix, "language_model.model"),
        )
        self.pad_token_id = (self.config.pad_token_id
                             if self.config.pad_token_id is not None else -1)
        self.unpadded_vocab_size = config.text_config.vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.text_config.hidden_size,
            org_num_embeddings=self.language_model.org_vocab_size,
            quant_config=quant_config,
        )
        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                self.vocab_size, logit_scale)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs: object,
    ):
        # 1. Extra the input embeddings
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        pixel_values = kwargs.get("pixel_values", None)
        pixel_mask = kwargs.get("pixel_mask", None)

        # 2. Merge text and images
        if pixel_values is not None:
            pixel_values = pixel_values.view(-1, *pixel_values.shape[-3:]).to(
                torch.bfloat16)
            pixel_mask = pixel_mask.view(-1, *pixel_mask.shape[-2:])
            selected_image_feature, image_attn_mask = self.vision_tower(
                pixel_values,
                pixel_mask=pixel_mask,
            )

            image_features = self.multi_modal_projector(
                selected_image_feature, attn_mask=image_attn_mask)

            inputs_embeds = inputs_embeds.to(image_features.dtype)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, image_features,
                self.config.image_token_index)

        hidden_states = self.language_model(
            input_ids,
            positions,
            kv_caches,
            attn_metadata,
            None,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        hf_to_vllm_mapper = WeightsMapper(
            orig_to_new_prefix={
                "language_model.model": "language_model",
                "language_model.lm_head": "lm_head",
            },
            orig_to_new_suffix={
                "experts.fc1.weight": "experts.w1",
                "experts.fc2.weight": "experts.w2",
                "router.weight": "experts.router_weight",
            },
        )

        loader = AutoWeightsLoader(self)
        loader.load_weights(weights, mapper=hf_to_vllm_mapper)
