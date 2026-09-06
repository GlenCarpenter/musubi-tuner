from __future__ import annotations

import os
from typing import Optional, TYPE_CHECKING, Union

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from musubi_tuner.dataset.architectures import (
    ARCHITECTURE_FRAMEPACK_FULL,
    ARCHITECTURE_FLUX_KONTEXT_FULL,
    ARCHITECTURE_HIDREAM_O1_FULL,
    ARCHITECTURE_HUNYUAN_VIDEO_FULL,
    ARCHITECTURE_HUNYUAN_VIDEO_1_5_FULL,
    ARCHITECTURE_IDEOGRAM4_FULL,
    ARCHITECTURE_KANDINSKY5_FULL,
    ARCHITECTURE_KREA2_FULL,
    ARCHITECTURE_KREA2_EDIT_FULL,
    ARCHITECTURE_QWEN_IMAGE_FULL,
    ARCHITECTURE_WAN_FULL,
    ARCHITECTURE_Z_IMAGE_FULL,
)
from musubi_tuner.utils import safetensors_utils
from musubi_tuner.utils.model_utils import dtype_to_str, remove_dtype_suffix

if TYPE_CHECKING:
    from musubi_tuner.dataset.image_video_dataset import ItemInfo

import logging

logger = logging.getLogger(__name__)

KREA2_EDIT_CACHE_SCHEMA_VERSION = "1"
KREA2_EDIT_FIT_PROTOCOL_VERSION = "1.0.0"
KREA2_EDIT_TEXT_CACHE_SCHEMA_VERSION = "1"
KREA2_EDIT_GROUNDING_PROTOCOL_VERSION = "1.0.0"


# We use simple if-else approach to support multiple architectures.
# Maybe we can use a plugin system in the future.

# the keys of the dict are `<content_type>_FxHxW_<dtype>` for latents
# and `<content_type>_<dtype|mask>` for other tensors


def save_latent_cache(item_info: ItemInfo, latent: torch.Tensor):
    """HunyuanVideo architecture. HunyuanVideo doesn't support I2V and control latents"""
    assert latent.dim() == 4, "latent should be 4D tensor (frame, channel, height, width)"

    _, F, H, W = latent.shape
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu()}

    save_latent_cache_common(item_info, sd, ARCHITECTURE_HUNYUAN_VIDEO_FULL)


def save_latent_cache_wan(
    item_info: ItemInfo,
    latent: torch.Tensor,
    clip_embed: Optional[torch.Tensor],
    image_latent: Optional[torch.Tensor],
    control_latent: Optional[torch.Tensor],
    f_indices: Optional[list[int]] = None,
):
    """Wan architecture"""
    assert latent.dim() == 4, "latent should be 4D tensor (frame, channel, height, width)"

    _, F, H, W = latent.shape
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu()}

    if clip_embed is not None:
        sd[f"clip_{dtype_str}"] = clip_embed.detach().cpu()

    if image_latent is not None:
        sd[f"latents_image_{F}x{H}x{W}_{dtype_str}"] = image_latent.detach().cpu()

    if control_latent is not None:
        sd[f"latents_control_{F}x{H}x{W}_{dtype_str}"] = control_latent.detach().cpu()

    if f_indices is not None:
        dtype_str = dtype_to_str(torch.int32)
        sd[f"f_indices_{dtype_str}"] = torch.tensor(f_indices, dtype=torch.int32)

    save_latent_cache_common(item_info, sd, ARCHITECTURE_WAN_FULL)


def save_latent_cache_framepack(
    item_info: ItemInfo,
    latent: torch.Tensor,
    latent_indices: torch.Tensor,
    clean_latents: torch.Tensor,
    clean_latent_indices: torch.Tensor,
    clean_latents_2x: torch.Tensor,
    clean_latent_2x_indices: torch.Tensor,
    clean_latents_4x: torch.Tensor,
    clean_latent_4x_indices: torch.Tensor,
    image_embeddings: torch.Tensor,
):
    """FramePack architecture"""
    assert latent.dim() == 4, "latent should be 4D tensor (frame, channel, height, width)"

    _, F, H, W = latent.shape
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu().contiguous()}

    # `latents_xxx` must have {F, H, W} suffix
    indices_dtype_str = dtype_to_str(latent_indices.dtype)
    sd[f"image_embeddings_{dtype_str}"] = image_embeddings.detach().cpu()  # image embeddings dtype is same as latents dtype
    sd[f"latent_indices_{indices_dtype_str}"] = latent_indices.detach().cpu()
    sd[f"clean_latent_indices_{indices_dtype_str}"] = clean_latent_indices.detach().cpu()
    sd[f"latents_clean_{F}x{H}x{W}_{dtype_str}"] = clean_latents.detach().cpu().contiguous()
    if clean_latent_2x_indices is not None:
        sd[f"clean_latent_2x_indices_{indices_dtype_str}"] = clean_latent_2x_indices.detach().cpu()
    if clean_latents_2x is not None:
        sd[f"latents_clean_2x_{F}x{H}x{W}_{dtype_str}"] = clean_latents_2x.detach().cpu().contiguous()
    if clean_latent_4x_indices is not None:
        sd[f"clean_latent_4x_indices_{indices_dtype_str}"] = clean_latent_4x_indices.detach().cpu()
    if clean_latents_4x is not None:
        sd[f"latents_clean_4x_{F}x{H}x{W}_{dtype_str}"] = clean_latents_4x.detach().cpu().contiguous()

    # for key, value in sd.items():
    #     print(f"{key}: {value.shape}")
    save_latent_cache_common(item_info, sd, ARCHITECTURE_FRAMEPACK_FULL)


def save_latent_cache_flux_kontext(
    item_info: ItemInfo,
    latent: torch.Tensor,
    control_latent: torch.Tensor,
):
    """FLUX.1 Kontext architecture"""
    assert latent.dim() == 3, "latent should be 3D tensor (channel, height, width)"

    _, H, W = latent.shape
    F = 1
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu().contiguous()}

    _, H, W = control_latent.shape
    F = 1
    sd[f"latents_control_{F}x{H}x{W}_{dtype_str}"] = control_latent.detach().cpu().contiguous()

    save_latent_cache_common(item_info, sd, ARCHITECTURE_FLUX_KONTEXT_FULL)


def save_latent_cache_flux_2(
    item_info: ItemInfo, latent: torch.Tensor, control_latent: Optional[list[torch.Tensor]], arch_full: str
):
    """Flux 2 architecture"""
    assert latent.dim() == 3, "latent should be 3D tensor (channel, height, width)"
    assert control_latent is None or all(cl.dim() == 3 for cl in control_latent), (
        "control_latent should be 3D tensor (channel, height, width) or None"
    )

    _, H, W = latent.shape
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{H}x{W}_{dtype_str}": latent.detach().cpu().contiguous()}

    if control_latent is not None:
        for i, cl in enumerate(control_latent):
            _, H, W = cl.shape
            sd[f"latents_control_{i}_{H}x{W}_{dtype_str}"] = cl.detach().cpu().contiguous()

    save_latent_cache_common(item_info, sd, arch_full)


def save_latent_cache_qwen_image(item_info: ItemInfo, latent: torch.Tensor, control_latent: Optional[list[torch.Tensor]]):
    """Qwen-Image architecture"""
    assert latent.dim() == 4, "latent should be 4D tensor (frame, channel, height, width)"
    assert control_latent is None or all(cl.dim() == 4 for cl in control_latent), (
        "control_latent should be 4D tensor (frame, channel, height, width) or None"
    )

    _, F, H, W = latent.shape
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu().contiguous()}

    if control_latent is not None:
        for i, cl in enumerate(control_latent):
            _, F, H, W = cl.shape
            sd[f"latents_control_{i}_{F}x{H}x{W}_{dtype_str}"] = cl.detach().cpu().contiguous()

    save_latent_cache_common(item_info, sd, ARCHITECTURE_QWEN_IMAGE_FULL)


def save_latent_cache_krea2(item_info: ItemInfo, latent: torch.Tensor):
    """Krea 2 (K2) architecture. Single image (F=1), Qwen-Image VAE latents (normalized).

    The latent uses the *same* normalization as the Qwen-Image VAE
    (`(raw - mean) / std`), which is exactly what K2's decoder inverts, so the
    Qwen-Image latent caching is reused as-is. No control latent for plain t2i.
    """
    assert latent.dim() == 4, "latent should be 4D tensor (channel, frame, height, width)"

    _, F, H, W = latent.shape
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu().contiguous()}

    save_latent_cache_common(item_info, sd, ARCHITECTURE_KREA2_FULL)


def save_latent_cache_krea2_edit(
    item_info: ItemInfo,
    target_latent: torch.Tensor,
    reference_latents: list[torch.Tensor],
    *,
    reference_pixel_sizes: Optional[list[tuple[int, int]]] = None,
    patch_size: int = 2,
    fit_protocol_version: str = KREA2_EDIT_FIT_PROTOCOL_VERSION,
):
    """Save one Krea 2 edit target and its ordered fitted reference latents."""
    if target_latent.dim() != 4:
        raise ValueError(f"target_latent must have shape (C, F, H, W), got {tuple(target_latent.shape)}")
    if not 1 <= len(reference_latents) <= 2:
        raise ValueError(f"Krea 2 edit cache requires one or two reference latents, got {len(reference_latents)}")
    if any(reference.dim() != 4 for reference in reference_latents):
        raise ValueError("each reference latent must have shape (C, F, H, W)")
    if any(reference.shape[0] != target_latent.shape[0] for reference in reference_latents):
        raise ValueError("reference latent channels must match target latent channels")
    if patch_size < 1:
        raise ValueError(f"patch_size must be positive, got {patch_size}")

    _, target_frames, target_height, target_width = target_latent.shape
    if target_height % patch_size or target_width % patch_size:
        raise ValueError("target latent dimensions must be divisible by the DiT patch size")
    target_dtype = dtype_to_str(target_latent.dtype)
    tensors = {
        f"latents_{target_frames}x{target_height}x{target_width}_{target_dtype}": target_latent.detach().cpu().contiguous()
    }
    metadata = {
        "krea2_edit_cache_schema": KREA2_EDIT_CACHE_SCHEMA_VERSION,
        "fit_protocol_version": str(fit_protocol_version),
        "reference_count": str(len(reference_latents)),
        "target_pixel_height": str(target_height * 8),
        "target_pixel_width": str(target_width * 8),
        "target_grid_height": str(target_height // patch_size),
        "target_grid_width": str(target_width // patch_size),
    }
    if reference_pixel_sizes is not None and len(reference_pixel_sizes) != len(reference_latents):
        raise ValueError("reference_pixel_sizes must contain one entry per reference latent")

    for index, reference in enumerate(reference_latents):
        _, frames, height, width = reference.shape
        if height % patch_size or width % patch_size:
            raise ValueError(f"reference latent {index} dimensions must be divisible by the DiT patch size")
        reference_dtype = dtype_to_str(reference.dtype)
        tensors[f"latents_control_{index}_{frames}x{height}x{width}_{reference_dtype}"] = (
            reference.detach().cpu().contiguous()
        )
        grid_height, grid_width = height // patch_size, width // patch_size
        metadata[f"reference_{index}_grid_height"] = str(grid_height)
        metadata[f"reference_{index}_grid_width"] = str(grid_width)
        metadata[f"reference_{index}_offset_height"] = str(max(0.0, (target_height // patch_size - grid_height) / 2))
        metadata[f"reference_{index}_offset_width"] = str(max(0.0, (target_width // patch_size - grid_width) / 2))
        pixel_height, pixel_width = (
            reference_pixel_sizes[index] if reference_pixel_sizes is not None else (height * 8, width * 8)
        )
        metadata[f"reference_{index}_pixel_height"] = str(pixel_height)
        metadata[f"reference_{index}_pixel_width"] = str(pixel_width)

    save_latent_cache_common(item_info, tensors, ARCHITECTURE_KREA2_EDIT_FULL, additional_metadata=metadata)


def load_krea2_edit_latent_cache(path: str) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, str]]:
    """Load and validate a Krea 2 edit latent cache."""
    with safe_open(path, framework="pt", device="cpu") as reader:
        metadata = dict(reader.metadata() or {})
        keys = list(reader.keys())
        if metadata.get("architecture") != ARCHITECTURE_KREA2_EDIT_FULL:
            raise ValueError(f"Krea 2 edit cache architecture mismatch: {metadata.get('architecture')!r}")
        if metadata.get("krea2_edit_cache_schema") != KREA2_EDIT_CACHE_SCHEMA_VERSION:
            raise ValueError("Unsupported Krea 2 edit cache schema")
        if metadata.get("fit_protocol_version") != KREA2_EDIT_FIT_PROTOCOL_VERSION:
            raise ValueError("Unsupported Krea 2 edit fit protocol")

        target_keys = [key for key in keys if key.startswith("latents_") and not key.startswith("latents_control_")]
        reference_keys = sorted(
            (key for key in keys if key.startswith("latents_control_")),
            key=lambda key: int(key.split("_")[2]),
        )
        if len(target_keys) != 1:
            raise ValueError(f"Krea 2 edit cache must contain exactly one target latent, found {len(target_keys)}")
        try:
            expected_reference_count = int(metadata["reference_count"])
        except (KeyError, ValueError) as exc:
            raise ValueError("Krea 2 edit cache has invalid reference_count metadata") from exc
        if not 1 <= expected_reference_count <= 2 or len(reference_keys) != expected_reference_count:
            raise ValueError(
                f"Krea 2 edit cache reference count mismatch: metadata={expected_reference_count}, tensors={len(reference_keys)}"
            )

        target = reader.get_tensor(target_keys[0])
        references = [reader.get_tensor(key) for key in reference_keys]
        actual_indices = [int(key.split("_")[2]) for key in reference_keys]
        if actual_indices != list(range(expected_reference_count)):
            raise ValueError(f"Krea 2 edit cache reference indices must be contiguous from zero, got {actual_indices}")

        def require_number(key: str, number_type):
            try:
                return number_type(metadata[key])
            except (KeyError, ValueError) as exc:
                raise ValueError(f"Krea 2 edit cache has invalid {key} metadata") from exc

        target_grid = (target.shape[-2] // 2, target.shape[-1] // 2)
        if target.shape[-2] % 2 or target.shape[-1] % 2:
            raise ValueError("Krea 2 edit target latent dimensions must be divisible by patch size 2")
        stored_target_grid = (require_number("target_grid_height", int), require_number("target_grid_width", int))
        if stored_target_grid != target_grid:
            raise ValueError(f"Krea 2 edit target geometry mismatch: metadata={stored_target_grid}, tensor={target_grid}")

        for index, reference in enumerate(references):
            if reference.shape[-2] % 2 or reference.shape[-1] % 2:
                raise ValueError(f"Krea 2 edit reference latent {index} dimensions must be divisible by patch size 2")
            reference_grid = (reference.shape[-2] // 2, reference.shape[-1] // 2)
            stored_grid = (
                require_number(f"reference_{index}_grid_height", int),
                require_number(f"reference_{index}_grid_width", int),
            )
            expected_offsets = (
                max(0.0, (target_grid[0] - reference_grid[0]) / 2),
                max(0.0, (target_grid[1] - reference_grid[1]) / 2),
            )
            stored_offsets = (
                require_number(f"reference_{index}_offset_height", float),
                require_number(f"reference_{index}_offset_width", float),
            )
            if stored_grid != reference_grid or stored_offsets != expected_offsets:
                raise ValueError(
                    f"Krea 2 edit reference {index} geometry mismatch: "
                    f"metadata grid/offset={stored_grid}/{stored_offsets}, tensor expects={reference_grid}/{expected_offsets}"
                )
    return target, references, metadata


def save_latent_cache_kandinsky5(
    item_info: ItemInfo,
    latent: torch.Tensor,
    image_latent: Optional[torch.Tensor] = None,
    control_latent: Optional[torch.Tensor] = None,
    scaling_factor: Optional[float] = None,
):
    """Kandinsky 5 architecture (image/video), with optional source/control latents for i2v/control."""
    assert latent.dim() == 3 or latent.dim() == 4, "latent should be 3D (C,H,W) or 4D (F,C,H,W) tensor"

    if latent.dim() == 4:
        _, F, H, W = latent.shape
    else:
        F, H, W = 1, latent.shape[1], latent.shape[2]
        latent = latent.unsqueeze(0)
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu().contiguous().clone()}

    if image_latent is not None:
        _, F_img, H_img, W_img = image_latent.shape
        sd[f"latents_image_{F_img}x{H_img}x{W_img}_{dtype_str}"] = image_latent.detach().cpu().contiguous().clone()

    if control_latent is not None:
        _, F_ctrl, H_ctrl, W_ctrl = control_latent.shape
        sd[f"latents_control_{F_ctrl}x{H_ctrl}x{W_ctrl}_{dtype_str}"] = control_latent.detach().cpu().contiguous().clone()

    if scaling_factor is not None:
        sd["vae_scaling_factor"] = torch.tensor(float(scaling_factor))

    save_latent_cache_common(item_info, sd, ARCHITECTURE_KANDINSKY5_FULL)


def save_latent_cache_hunyuan_video_1_5(
    item_info: ItemInfo,
    latent: torch.Tensor,
    image_latent: Optional[torch.Tensor],
    vision_feature: Optional[torch.Tensor],
):
    """HunyuanVideo 1.5 architecture"""
    _, F, H, W = latent.shape
    dtype_str = dtype_to_str(latent.dtype)
    sd: dict[str, torch.Tensor] = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu()}

    if image_latent is not None:
        dtype_str = dtype_to_str(image_latent.dtype)
        _, F, H, W = image_latent.shape
        sd[f"latents_image_{F}x{H}x{W}_{dtype_str}"] = image_latent.detach().cpu()

    if vision_feature is not None:
        dtype_str = dtype_to_str(vision_feature.dtype)
        sd[f"siglip_{dtype_str}"] = vision_feature.detach().cpu()

    save_latent_cache_common(item_info, sd, ARCHITECTURE_HUNYUAN_VIDEO_1_5_FULL)


def save_latent_cache_z_image(item_info: ItemInfo, latent: torch.Tensor):
    """Z-Image architecture. No control latent is supported."""
    assert latent.dim() == 3, "latent should be 3D tensor (channel, height, width)"

    C, H, W = latent.shape
    F = 1
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu().contiguous()}

    save_latent_cache_common(item_info, sd, ARCHITECTURE_Z_IMAGE_FULL)


def save_pixel_cache_hidream_o1(
    item_info: ItemInfo, pixel_tokens: torch.Tensor, control_pixel_tokens: Optional[Union[list[torch.Tensor], torch.Tensor]] = None
):
    """HiDream-O1 architecture. Cache normalized 32x32 pixel patch tokens."""
    assert pixel_tokens.dim() == 3, "pixel_tokens should be 3D tensor (height_patches, width_patches, patch_dim)"

    height_patches, width_patches, _ = pixel_tokens.shape
    dtype_str = dtype_to_str(pixel_tokens.dtype)
    sd = {f"latents_1x{height_patches}x{width_patches}_{dtype_str}": pixel_tokens.detach().cpu().contiguous()}

    if control_pixel_tokens is not None:
        if torch.is_tensor(control_pixel_tokens):
            assert control_pixel_tokens.dim() == 4, (
                "control_pixel_tokens should be 4D tensor (num_controls, height_patches, width_patches, patch_dim)"
            )
            control_pixel_tokens = list(control_pixel_tokens)
        assert all(cl.dim() == 3 for cl in control_pixel_tokens), (
            "control_pixel_tokens should contain 3D tensors (height_patches, width_patches, patch_dim)"
        )
        for i, cl in enumerate(control_pixel_tokens):
            control_height_patches, control_width_patches, _ = cl.shape
            control_dtype_str = dtype_to_str(cl.dtype)
            sd[f"latents_control_{i}_{control_height_patches}x{control_width_patches}_{control_dtype_str}"] = (
                cl.detach().cpu().contiguous()
            )

    save_latent_cache_common(item_info, sd, ARCHITECTURE_HIDREAM_O1_FULL)


def save_latent_cache_ideogram4(item_info: ItemInfo, latent: torch.Tensor):
    """Ideogram 4 architecture."""
    assert latent.dim() == 3, "latent should be 3D tensor (channel, height, width)"

    _, H, W = latent.shape
    F = 1
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu().contiguous()}

    save_latent_cache_common(item_info, sd, ARCHITECTURE_IDEOGRAM4_FULL)


def save_latent_cache_common(
    item_info: ItemInfo,
    sd: dict[str, torch.Tensor],
    arch_fullname: str,
    additional_metadata: Optional[dict[str, str]] = None,
):
    metadata = {
        "architecture": arch_fullname,
        "width": f"{item_info.original_size[0]}",
        "height": f"{item_info.original_size[1]}",
        "format_version": "1.0.1",
    }
    if item_info.frame_count is not None:
        metadata["frame_count"] = f"{item_info.frame_count}"
    if additional_metadata:
        metadata.update(additional_metadata)

    for key, value in sd.items():
        # NaN check and show warning, replace NaN with 0
        if torch.isnan(value).any():
            logger.warning(f"{key} tensor has NaN: {item_info.item_key}, replace NaN with 0")
            value[torch.isnan(value)] = 0

    latent_dir = os.path.dirname(item_info.latent_cache_path)
    os.makedirs(latent_dir, exist_ok=True)

    save_file(sd, item_info.latent_cache_path, metadata=metadata)


def save_text_encoder_output_cache(item_info: ItemInfo, embed: torch.Tensor, mask: Optional[torch.Tensor], is_llm: bool):
    """HunyuanVideo architecture"""
    assert embed.dim() == 1 or embed.dim() == 2, (
        f"embed should be 2D tensor (feature, hidden_size) or (hidden_size,), got {embed.shape}"
    )
    assert mask is None or mask.dim() == 1, f"mask should be 1D tensor (feature), got {mask.shape}"

    sd = {}
    dtype_str = dtype_to_str(embed.dtype)
    text_encoder_type = "llm" if is_llm else "clipL"
    sd[f"{text_encoder_type}_{dtype_str}"] = embed.detach().cpu()
    if mask is not None:
        sd[f"{text_encoder_type}_mask"] = mask.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_HUNYUAN_VIDEO_FULL)


def save_text_encoder_output_cache_wan(item_info: ItemInfo, embed: torch.Tensor):
    """Wan architecture. Wan2.1 only has a single text encoder"""

    sd = {}
    dtype_str = dtype_to_str(embed.dtype)
    text_encoder_type = "t5"
    sd[f"varlen_{text_encoder_type}_{dtype_str}"] = embed.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_WAN_FULL)


def save_text_encoder_output_cache_framepack(
    item_info: ItemInfo, llama_vec: torch.Tensor, llama_attention_mask: torch.Tensor, clip_l_pooler: torch.Tensor
):
    """FramePack architecture."""
    sd = {}
    dtype_str = dtype_to_str(llama_vec.dtype)
    sd[f"llama_vec_{dtype_str}"] = llama_vec.detach().cpu()
    sd["llama_attention_mask"] = llama_attention_mask.detach().cpu()
    dtype_str = dtype_to_str(clip_l_pooler.dtype)
    sd[f"clip_l_pooler_{dtype_str}"] = clip_l_pooler.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_FRAMEPACK_FULL)


def save_text_encoder_output_cache_flux_kontext(item_info: ItemInfo, t5_vec: torch.Tensor, clip_l_pooler: torch.Tensor):
    """Flux Kontext architecture."""

    sd = {}
    dtype_str = dtype_to_str(t5_vec.dtype)
    sd[f"t5_vec_{dtype_str}"] = t5_vec.detach().cpu()
    dtype_str = dtype_to_str(clip_l_pooler.dtype)
    sd[f"clip_l_pooler_{dtype_str}"] = clip_l_pooler.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_FLUX_KONTEXT_FULL)


def save_text_encoder_output_cache_flux_2(item_info: ItemInfo, ctx_vec: torch.Tensor, arch_full: str):
    """Flux 2 architecture."""

    sd = {}
    dtype_str = dtype_to_str(ctx_vec.dtype)
    sd[f"ctx_vec_{dtype_str}"] = ctx_vec.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, arch_full)


def save_text_encoder_output_cache_qwen_image(item_info: ItemInfo, embed: torch.Tensor):
    """Qwen-Image architecture."""
    sd = {}
    dtype_str = dtype_to_str(embed.dtype)
    sd[f"varlen_vl_embed_{dtype_str}"] = embed.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_QWEN_IMAGE_FULL)


def save_text_encoder_output_cache_krea2(item_info: ItemInfo, embed: torch.Tensor):
    """Krea 2 (K2) architecture.

    `embed` is the per-item stack of *selected* Qwen3-VL hidden-state layers for the
    valid (non-padding) tokens only: shape (valid_len, num_select_layers, hidden).
    Stored varlen (no padding, no mask): K2 gives text tokens zero RoPE position and
    masks padding in attention, so dropping padding is lossless for the image outputs.
    The layerwise fusion (TextFusionTransformer) is trainable and lives in the DiT, so
    the raw selected-layer stack is what gets cached.
    """
    assert embed.dim() == 3, "embed should be 3D tensor (valid_len, num_select_layers, hidden)"

    sd = {}
    dtype_str = dtype_to_str(embed.dtype)
    sd[f"varlen_krea2_vl_embed_{dtype_str}"] = embed.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_KREA2_FULL)


def save_text_encoder_output_cache_krea2_edit(
    item_info: ItemInfo,
    embed: torch.Tensor,
    *,
    grounding_pixels: int,
    reference_pixel_sizes: list[tuple[int, int]],
):
    """Save fixed-scale image-grounded Qwen3-VL features for Krea 2 edit."""
    if embed.dim() != 3:
        raise ValueError(f"embed must have shape (valid_len, selected_layers, hidden), got {tuple(embed.shape)}")
    if grounding_pixels <= 0:
        raise ValueError(f"grounding_pixels must be positive for a fixed-scale cache, got {grounding_pixels}")
    if not 1 <= len(reference_pixel_sizes) <= 2:
        raise ValueError(f"Krea 2 edit text cache requires one or two reference images, got {len(reference_pixel_sizes)}")

    dtype_str = dtype_to_str(embed.dtype)
    metadata = {
        "krea2_edit_text_cache_schema": KREA2_EDIT_TEXT_CACHE_SCHEMA_VERSION,
        "grounding_protocol_version": KREA2_EDIT_GROUNDING_PROTOCOL_VERSION,
        "grounding_mode": "fixed",
        "grounding_pixels": str(grounding_pixels),
        "reference_count": str(len(reference_pixel_sizes)),
    }
    for index, (height, width) in enumerate(reference_pixel_sizes):
        metadata[f"reference_{index}_pixel_height"] = str(height)
        metadata[f"reference_{index}_pixel_width"] = str(width)

    save_text_encoder_output_cache_common(
        item_info,
        {f"varlen_krea2_vl_embed_{dtype_str}": embed.detach().cpu().contiguous()},
        ARCHITECTURE_KREA2_EDIT_FULL,
        merge_existing=False,
        additional_metadata=metadata,
    )


def validate_krea2_edit_text_encoder_cache(
    path: str,
    *,
    expected_grounding_pixels: Optional[int] = None,
    expected_reference_count: Optional[int] = None,
) -> dict[str, str]:
    """Validate a fixed-scale Krea 2 edit text cache and return its metadata."""
    with safe_open(path, framework="pt", device="cpu") as reader:
        metadata = dict(reader.metadata() or {})
        keys = list(reader.keys())
        if metadata.get("architecture") != ARCHITECTURE_KREA2_EDIT_FULL:
            raise ValueError(f"Krea 2 edit text cache architecture mismatch: {metadata.get('architecture')!r}")
        if metadata.get("krea2_edit_text_cache_schema") != KREA2_EDIT_TEXT_CACHE_SCHEMA_VERSION:
            raise ValueError("Unsupported Krea 2 edit text cache schema")
        if metadata.get("grounding_protocol_version") != KREA2_EDIT_GROUNDING_PROTOCOL_VERSION:
            raise ValueError("Unsupported Krea 2 edit grounding protocol")
        if metadata.get("grounding_mode") != "fixed":
            raise ValueError("Krea 2 edit cached text conditioning must declare grounding_mode='fixed'")
        embed_keys = [key for key in keys if key.startswith("varlen_krea2_vl_embed_")]
        if len(embed_keys) != 1:
            raise ValueError(f"Krea 2 edit text cache must contain exactly one embedding tensor, found {len(embed_keys)}")
        embed = reader.get_tensor(embed_keys[0])
        if embed.dim() != 3:
            raise ValueError(f"Krea 2 edit cached embedding must be 3D, got {tuple(embed.shape)}")
        try:
            grounding_pixels = int(metadata["grounding_pixels"])
            reference_count = int(metadata["reference_count"])
        except (KeyError, ValueError) as exc:
            raise ValueError("Krea 2 edit text cache has invalid grounding metadata") from exc
        if grounding_pixels <= 0 or not 1 <= reference_count <= 2:
            raise ValueError("Krea 2 edit text cache grounding metadata is out of range")
        if expected_grounding_pixels is not None and grounding_pixels != expected_grounding_pixels:
            raise ValueError(
                f"Krea 2 edit text cache grounding scale mismatch: cached={grounding_pixels}, expected={expected_grounding_pixels}"
            )
        if expected_reference_count is not None and reference_count != expected_reference_count:
            raise ValueError(
                f"Krea 2 edit text cache reference count mismatch: cached={reference_count}, expected={expected_reference_count}"
            )
    return metadata


def save_text_encoder_output_cache_kandinsky5(
    item_info: ItemInfo, text_embeds: torch.Tensor, pooled_embed: torch.Tensor, attention_mask: torch.Tensor
):
    """Kandinsky 5 architecture."""
    sd = {}
    dtype_str = dtype_to_str(text_embeds.dtype)
    sd[f"text_embeds_{dtype_str}"] = text_embeds.detach().cpu()
    dtype_str = dtype_to_str(pooled_embed.dtype)
    sd[f"pooled_embed_{dtype_str}"] = pooled_embed.detach().cpu()
    sd["attention_mask"] = attention_mask.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_KANDINSKY5_FULL)


def save_text_encoder_output_cache_hunyuan_video_1_5(item_info: ItemInfo, embed: torch.Tensor, byt5_embed: torch.Tensor):
    """Hunyuan-Video 1.5 architecture."""
    sd = {}
    dtype_str = dtype_to_str(embed.dtype)
    sd[f"varlen_vl_embed_{dtype_str}"] = embed.detach().cpu()
    dtype_str = dtype_to_str(byt5_embed.dtype)
    sd[f"varlen_byt5_embed_{dtype_str}"] = byt5_embed.detach().cpu()
    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_HUNYUAN_VIDEO_1_5_FULL)


def save_text_encoder_output_cache_z_image(item_info: ItemInfo, embed: torch.Tensor):
    """Z-Image architecture."""
    sd = {}
    dtype_str = dtype_to_str(embed.dtype)
    sd[f"varlen_llm_embed_{dtype_str}"] = embed.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_Z_IMAGE_FULL)


def save_text_encoder_output_cache_ideogram4(item_info: ItemInfo, features: torch.Tensor):
    """Ideogram 4 architecture."""
    sd = {}
    dtype_str = dtype_to_str(features.dtype)
    sd[f"varlen_i4_llm_features_{dtype_str}"] = features.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_IDEOGRAM4_FULL)


def save_text_encoder_output_cache_hidream_o1(
    item_info: ItemInfo,
    input_ids: torch.Tensor,
    input_embeds: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    token_types: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    image_grid_thw: Optional[torch.Tensor] = None,
):
    """HiDream-O1 architecture. Cache tokenized prompt and optional initial text token embeddings."""
    # The dtype suffix is parsed back on load (see bucket.py), so it must be built per tensor here; absent optionals
    # are simply skipped. HiDream-O1 writes its full key set in a single pass, so the cache is overwritten fresh
    # (merge_existing=False) instead of merged, dropping any stale optional/dtype keys left from a previous run.
    tensors = {
        "varlen_input_ids": input_ids,
        "varlen_input_embeds": input_embeds,
        "varlen_position_ids": position_ids,
        "varlen_token_types": token_types,
        "varlen_pixel_values": pixel_values,
        "varlen_image_grid_thw": image_grid_thw,
    }
    sd = {f"{name}_{dtype_to_str(t.dtype)}": t.detach().cpu() for name, t in tensors.items() if t is not None}

    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_HIDREAM_O1_FULL, merge_existing=False)


def save_text_encoder_output_cache_common(
    item_info: ItemInfo,
    sd: dict[str, torch.Tensor],
    arch_fullname: str,
    merge_existing: bool = True,
    additional_metadata: Optional[dict[str, str]] = None,
):
    # merge_existing keeps keys written by previous passes (e.g. HunyuanVideo caches LLM and CLIP separately).
    # Single-pass architectures that write their full key set at once should pass merge_existing=False so the
    # cache is overwritten fresh, dropping any stale keys (e.g. optionals/dtypes) left from an earlier run.
    for key, value in sd.items():
        # NaN check and show warning, replace NaN with 0
        if torch.isnan(value).any():
            logger.warning(f"{key} tensor has NaN: {item_info.item_key}, replace NaN with 0")
            value[torch.isnan(value)] = 0

    metadata = {
        "architecture": arch_fullname,
        "caption1": item_info.caption,
        "format_version": "1.0.1",
    }
    if merge_existing and os.path.exists(item_info.text_encoder_output_cache_path):
        # load existing cache and update metadata
        new_key_bases = {remove_dtype_suffix(key) for key in sd}  # logical keys (dtype stripped) just written
        with safetensors_utils.MemoryEfficientSafeOpen(item_info.text_encoder_output_cache_path) as f:
            existing_metadata = f.metadata()
            for key in f.keys():
                # Skip any existing key superseded by a freshly written one. Comparing on the dtype-stripped base
                # (not the exact key) also drops a stale copy written in another precision, e.g. re-caching after
                # toggling fp8; otherwise both dtype variants would survive and collide under one key on load.
                if remove_dtype_suffix(key) in new_key_bases:
                    continue
                sd[key] = f.get_tensor(key)

        assert existing_metadata["architecture"] == metadata["architecture"], "architecture mismatch"
        if existing_metadata["caption1"] != metadata["caption1"]:
            logger.warning(f"caption mismatch: existing={existing_metadata['caption1']}, new={metadata['caption1']}, overwrite")
        # TODO verify format_version

        existing_metadata.pop("caption1", None)
        existing_metadata.pop("format_version", None)
        metadata.update(existing_metadata)  # copy existing metadata except caption and format_version
    else:
        text_encoder_output_dir = os.path.dirname(item_info.text_encoder_output_cache_path)
        os.makedirs(text_encoder_output_dir, exist_ok=True)

    if additional_metadata:
        metadata.update(additional_metadata)

    safetensors_utils.mem_eff_save_file(sd, item_info.text_encoder_output_cache_path, metadata=metadata)
