import math
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
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
from vllm.utils import is_list_of

logger = logging.get_logger(__name__)

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from transformers.activations import ACT2FN
from transformers.models.idefics2.configuration_idefics2 import Idefics2VisionConfig
from vllm.config import QuantizationConfig
from vllm.model_executor.models.idefics2_vision_model import Idefics2VisionTransformer


class AriaVisionConfig(Idefics2VisionConfig):
    model_type = "aria_vision_model"


class IdentityOp(torch.nn.Module):
    """
    An identity operation that returns the input unchanged.

    This can be used as a placeholder or to maintain architectural consistency
    when a specific operation is not needed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x


class AriaVisionTransformer(Idefics2VisionTransformer):

    def __init__(
        self,
        config: AriaVisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config, prefix)
        self.post_layernorm = IdentityOp()


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
        self.linear_in = nn.Linear(embed_dim, ff_dim, bias=False)
        self.linear_out = nn.Linear(ff_dim, output_dim, bias=False)
        self.act = ACT2FN["gelu_new"]

    def forward(self, hidden_states):
        hidden_states = self.act(self.linear_in(hidden_states))
        hidden_states = self.linear_out(hidden_states)
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


class AriaMoELMConfig(LlamaConfig):
    """
    Configuration class for AriaMoE language model.

    This class extends the LlamaConfig to include additional parameters specific to the Mixture of Experts (MoE) architecture.
    """

    model_type = "aria_moe_lm"

    def __init__(
        self,
        moe_intermediate_size: int = 4096,
        moe_num_experts: int = 8,
        moe_topk: int = 2,
        moe_z_loss_coeff: float = 1e-5,
        moe_aux_loss_coeff: float = 1e-3,
        moe_num_shared_experts: int = 2,
        **kwargs,
    ):
        """
        Initialize the AriaMoELMConfig.

        Args:
            moe_intermediate_size (int): The intermediate size for MoE layers. Default is 4096.
            moe_num_experts (int): The number of experts in the MoE layer. Default is 8.
            moe_topk (int): The number of top experts to route to for each token. Default is 2.
            moe_z_loss_coeff (float): The coefficient for the auxiliary z-loss. Default is 1e-5.
            moe_aux_loss_coeff (float): The coefficient for the auxiliary load balancing loss. Default is 1e-3.
            moe_num_shared_experts (int): The number of shared experts. Default is 2.
            **kwargs: Additional keyword arguments to be passed to the parent LlamaConfig.
        """
        super().__init__(**kwargs)
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_num_experts = moe_num_experts
        self.moe_topk = moe_topk
        self.moe_z_loss_coeff = moe_z_loss_coeff
        self.moe_aux_loss_coeff = moe_aux_loss_coeff
        self.moe_num_shared_experts = moe_num_shared_experts


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


def _select_best_resolution(img_width: int, img_height: int,
                            target_ratios: List[List[int]], patch_size: int):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        img_width: the original widths of images.
        img_height: the original heights of images.
        target_ratios (2d numpy array): dimension size (M,2)
        patch_size (int): image patch size

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """

    aspect_ratio = img_width / img_height
    best_ratio_diff = float("inf")
    best_ratio_w, best_ratio_h = 1, 1
    area = np.int32(img_width) * np.int32(img_height)
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio_w, best_ratio_h = ratio[0], ratio[1]
        elif (ratio_diff == best_ratio_diff
              and area > 0.5 * patch_size * patch_size * ratio[0] * ratio[1]):
            best_ratio_w, best_ratio_h = ratio[0], ratio[1]

    return best_ratio_w, best_ratio_h


def split_image(
    image: Image.Image,
    split_image: bool,
    split_ratio: List[List[int]] = [
        [1, 2],
        [1, 3],
        [1, 4],
        [1, 5],
        [1, 6],
        [1, 7],
        [1, 8],
        [2, 4],
        [2, 3],
        [2, 2],
        [2, 1],
        [3, 1],
        [3, 2],
        [4, 1],
        [4, 2],
        [5, 1],
        [6, 1],
        [7, 1],
        [8, 1],
    ],
    patch_size: int = 980,
) -> List[Image.Image]:
    """
    Split image into multiple patches

    Args:
        image (PIL.Image): Input image.
        split_image (bool): Whether to split the image into patches.
        split_ratio (2d numpy array): dimension size (M,2)
        patch_size (int): image patch size

    Returns:
        List[PIL.Image]: List of splitted images.
    """
    if split_image:
        ratio_width, ratio_height = _select_best_resolution(
            image.width, image.height, split_ratio, patch_size)
        resize_width = patch_size * ratio_width
        resize_height = patch_size * ratio_height
        blocks = ratio_width * ratio_height
        resized_img = image.resize((resize_width, resize_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (resize_width // patch_size)) * patch_size,
                (i // (resize_width // patch_size)) * patch_size,
                ((i % (resize_width // patch_size)) + 1) * patch_size,
                ((i // (resize_width // patch_size)) + 1) * patch_size,
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if len(processed_images) != 1:
            processed_images.insert(0, image)
        return processed_images
    else:
        return [image]


def get_max_multimodal_tokens(ctx):
    return max(ctx.model_config.hf_config.image_size2tokens.values())


def input_mapper_for_aria(ctx, data):
    """
    This is almost same with _default_input_mapper from vllm.multimodal.image.py.
    Args:
        ctx (ModelExecutorContext): The context object containing necessary parameters.
        data (Union[Image.Image, torch.Tensor, List[Union[Image.Image, torch.Tensor]]]): The input data to be processed.
    The only different is we would like to support runtime max_image_size adjustment.
    """
    model_config = ctx.model_config
    max_image_size = getattr(model_config.multimodal_config, "max_image_size",
                             980)

    # PIL image
    if isinstance(data, Image.Image) or is_list_of(data, Image.Image):
        image_processor = cached_get_image_processor(
            model_config.model,
            trust_remote_code=model_config.trust_remote_code)
        if image_processor is None:
            raise RuntimeError("No HuggingFace processor is available "
                               "to process the image object")
        try:
            batch_data = image_processor.preprocess(
                data, max_image_size=max_image_size, return_tensors="pt").data
            batch_data.pop("num_crops")
        except Exception:
            logger.error("Failed to process image (%s)", data)
            raise

        return MultiModalInputs(batch_data)

    # Image embedding
    elif isinstance(data, torch.Tensor) or is_list_of(data, torch.Tensor):
        return MultiModalInputs({"image_embeds": data})

    raise TypeError(f"Invalid image type: {type(data)}")


def input_processor(ctx, llm_inputs):
    multi_modal_data = llm_inputs.get("multi_modal_data")
    # if it is pure text input, use it as is
    if multi_modal_data is None or "image" not in multi_modal_data:
        return llm_inputs

    model_config = ctx.model_config

    tokenizer = cached_get_tokenizer(model_config.tokenizer)
    hf_config = model_config.hf_config

    # prepare image tokens, the max_image_size is used to determine the number of patch_size for every image
    max_image_size = multi_modal_data.pop("max_image_size", 980)
    _split_image = multi_modal_data.pop("split_image", False)

    assert isinstance(max_image_size, int) or isinstance(
        max_image_size, float), "max_image_size should be float or int"
    images = (multi_modal_data["image"] if isinstance(
        multi_modal_data["image"], list) else [multi_modal_data["image"]])
    num_crops = []
    splitted_images = []
    for image in images:
        splitted_image = split_image(image,
                                     _split_image,
                                     patch_size=max_image_size)
        splitted_images.extend(splitted_image)
        num_crops.append(len(splitted_image))
    max_image_size = [max_image_size] * len(images)
    # reassign the image because we might split them into mini-patches
    multi_modal_data["image"] = splitted_images

    # Mapping the image patch size to the corresponding number of tokens for each image
    image_feature_sizes = []
    for image_size, num_crop in zip(max_image_size, num_crops):
        assert (
            image_size in hf_config.image_size2tokens
        ), f"Invalid image size: {image_size}, available options: {list(hf_config.image_size2tokens.keys())}"
        image_feature_sizes.append(hf_config.image_size2tokens[image_size] *
                                   num_crop)

    # Set up the max_image_size and split_image in the RuntimeContext for the image processor
    # TODO: Supports dynamic image size support
    setattr(model_config.multimodal_config, "max_image_size",
            max(max_image_size))

    new_prompt, new_token_ids, ranges = repeat_and_pad_placeholder_tokens(
        tokenizer,
        llm_inputs.get("prompt"),
        llm_inputs["prompt_token_ids"],
        placeholder_token_id=hf_config.image_token_index,
        repeat_count=image_feature_sizes,
    )

    return token_inputs(
        prompt_token_ids=new_token_ids,
        prompt=new_prompt,
        multi_modal_data=multi_modal_data,
        # multi_modal_placeholders={"image": ranges},
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
        setattr(
            config,
            "image_size2tokens",
            {
                int(math.sqrt(k) * config.vision_config.patch_size): v
                for k, v in config.projector_patch_to_query_dict.items()
            },
        )
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
