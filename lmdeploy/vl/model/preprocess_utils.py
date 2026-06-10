# Copyright (c) OpenMMLab. All rights reserved.
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from lmdeploy.utils import get_logger
from lmdeploy.vl.constants import Modality

if TYPE_CHECKING:
    from lmdeploy.vl.model.base import MultimodalSpecialTokens

logger = get_logger('lmdeploy')


def get_mm_items_offset(input_ids: torch.Tensor, mm_token_id: int) -> list[tuple[int, int]]:
    """Return (start, end) ranges of contiguous mm_token_id runs in input_ids.

    Example:
        input_ids = [1, 2, 3, 3, 3, 4, 3, 3], mm_token_id = 3
        returns [(2, 5), (6, 8)]
        end_positions + 1 to turn it into exclusive end index for pytorch engine
    """
    mask = (input_ids == mm_token_id)
    prev_is_false = ~F.pad(mask[:-1], (1, 0), value=False)  # [True] + ~mask[:-1]
    next_is_false = ~F.pad(mask[1:], (0, 1), value=False)   # ~mask[1:] + [True]
    start_positions = (mask & prev_is_false).nonzero(as_tuple=True)[0]
    end_positions = (mask & next_is_false).nonzero(as_tuple=True)[0] + 1
    return list(zip(start_positions.tolist(), end_positions.tolist()))


def get_override_size(processor, mm_processor_kwargs: dict[str, Any] | None = None, modality: str = ''):
    """Return overridden size dict from mm_processor_kwargs, or None if not
    applicable."""
    if not mm_processor_kwargs:
        return None
    try:
        default_min = processor.size['shortest_edge']
        default_max = processor.size['longest_edge']
    except (AttributeError, KeyError, TypeError):
        tag = f'[{modality}] ' if modality else ''
        logger.warning(f'{tag}processor does not expose size[shortest_edge/longest_edge], '
                       f'mm_processor_kwargs size override will be skipped.')
        return None
    override_min = mm_processor_kwargs.get('min_pixels', default_min)
    override_max = mm_processor_kwargs.get('max_pixels', default_max)
    tag = f'[{modality}] ' if modality else ''
    if override_min > override_max:
        logger.warning(
            f'{tag}Overriding min_pixels {override_min} > max_pixels {override_max}, '
            f'falling back to defaults, min_pixels={default_min} and max_pixels={default_max}.'
        )
        return None
    logger.info(f'{tag}Overriding processor size with min_pixels={override_min} and max_pixels={override_max}.')
    return {'shortest_edge': override_min, 'longest_edge': override_max}


def get_expanded_input_ids(input_prompt, collected_mm_items, processor,
                           mm_tokens: 'MultimodalSpecialTokens') -> torch.Tensor:
    """Return input_ids with each image placeholder expanded to its actual
    token count."""
    image_token_id = mm_tokens.image_token_id
    image_grid_thw = collected_mm_items.get(Modality.IMAGE, {}).get('image_grid_thw', None)
    merge_length = processor.image_processor.merge_size ** 2
    image_index = 0
    input_ids = []
    for token in input_prompt:
        if token == image_token_id:
            image_tokens = image_grid_thw[image_index].prod() // merge_length
            input_ids.extend([image_token_id] * image_tokens)
            image_index += 1
        else:
            input_ids.append(token)
    return torch.tensor(input_ids)


def _expand_bundled_image_items(item: dict, token_id: int) -> list[dict]:
    expanded_mm_items = []
    image_grid_thw = item['image_grid_thw']
    num_items = len(item['offset'])
    num_images = len(image_grid_thw)

    if num_items != num_images:
        raise ValueError(f'Image offsets must match image grids: got {num_items} offsets for {num_images} images.')

    patches_per_item = []
    for grid in image_grid_thw:
        grid_tensor = torch.as_tensor(grid, dtype=torch.long)
        patches_per_item.append(int(torch.prod(grid_tensor).item()))

    cumulative = torch.cumsum(torch.tensor(patches_per_item, dtype=torch.long), dim=0)
    slice_indices = [0] + cumulative.tolist()

    for i in range(num_items):
        start_idx, end_idx = slice_indices[i], slice_indices[i + 1]
        expanded_mm_items.append(
            dict(
                modality=Modality.IMAGE,
                pixel_values=item['feature'][start_idx:end_idx].clone(),
                image_grid_thw=image_grid_thw[i],
                offset=item['offset'][i],
                image_token_id=token_id,
            ))
    return expanded_mm_items


def _expand_bundled_video_items(item: dict, token_id: int) -> list[dict]:
    expanded_mm_items = []
    video_grid_thw = item['video_grid_thw']
    num_items = len(item['offset'])
    num_videos = video_grid_thw.shape[0]

    # calculate patches for each video (i.e. feature rows): T * H * W
    patches_per_video = []
    for i in range(num_videos):
        grid = video_grid_thw[i]
        patches_per_video.append(
            int(torch.prod(grid).item()) if isinstance(grid, torch.Tensor) else int(
                torch.prod(torch.as_tensor(grid, dtype=torch.long)).item()))

    # get frames for each video: T
    frames_per_video = []
    for i in range(num_videos):
        grid = video_grid_thw[i]
        frames = int(grid[0].item()) if isinstance(grid, torch.Tensor) else int(
            torch.as_tensor(grid, dtype=torch.long)[0].item())
        frames_per_video.append(frames)

    cumulative = torch.cumsum(torch.tensor(patches_per_video, dtype=torch.long), dim=0)
    slice_indices = [0] + cumulative.tolist()

    # qwen3-omni emits one offset per video: keep each [T, H, W] grid intact.
    if num_items == num_videos:
        for video_idx in range(num_videos):
            feature_start, feature_end = slice_indices[video_idx], slice_indices[video_idx + 1]

            second_per_grid = item.get('video_second_per_grid')
            if second_per_grid is not None:
                second_per_grid = second_per_grid[video_idx].item()

            expanded_mm_items.append(
                dict(
                    modality=Modality.VIDEO,
                    pixel_values_videos=item['feature'][feature_start:feature_end].clone(),
                    video_grid_thw=video_grid_thw[video_idx],
                    offset=item['offset'][video_idx],
                    second_per_grid=second_per_grid,
                    video_token_id=token_id,
                ))
        return expanded_mm_items

    total_frames = sum(frames_per_video)
    if num_items != total_frames:
        raise ValueError('Video offsets must be per-video or per-frame: '
                         f'got {num_items} offsets for {num_videos} videos and {total_frames} frames.')

    # qwen3-vl emits one offset per frame: split into [1, H, W] items.
    # because MultiModalData carries one prompt span per item
    frame_start_indices = [0]
    for i in range(num_videos):
        frame_start_indices.append(frame_start_indices[-1] + frames_per_video[i])

    for video_idx in range(num_videos):
        feature_start, feature_end = slice_indices[video_idx], slice_indices[video_idx + 1]
        frame_start, frame_end = frame_start_indices[video_idx], frame_start_indices[video_idx + 1]
        video_feature = item['feature'][feature_start:feature_end]
        frame_offsets = item['offset'][frame_start:frame_end]

        second_per_grid = item.get('video_second_per_grid')
        if second_per_grid is not None:
            second_per_grid = second_per_grid[video_idx].item()

        t, h, w = video_grid_thw[video_idx].tolist()
        for frame_idx in range(t):
            grid_start = frame_idx * h * w
            grid_end = (frame_idx + 1) * h * w
            expanded_mm_items.append(
                dict(
                    modality=Modality.VIDEO,
                    pixel_values_videos=video_feature[grid_start:grid_end].clone(),
                    video_grid_thw=torch.tensor([1, h, w]),
                    offset=frame_offsets[frame_idx],
                    second_per_grid=second_per_grid,
                    video_token_id=token_id,
                ))
    return expanded_mm_items


def _expand_bundled_audio_items(item: dict, token_id: int) -> list[dict]:
    expanded_mm_items = []
    for i in range(len(item['offset'])):
        feature_attention_mask = item.get('feature_attention_mask')
        if feature_attention_mask is not None:
            feature_attention_mask = feature_attention_mask[i:i + 1].clone()
        expanded_mm_items.append(
            dict(
                modality=Modality.AUDIO,
                input_features=item['feature'][i:i + 1].clone(),
                feature_attention_mask=feature_attention_mask,
                offset=item['offset'][i],
                audio_token_id=token_id,
            ))
    return expanded_mm_items


# adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/mm_utils.py
def get_expanded_mm_items(collected_mm_items, mm_tokens: 'MultimodalSpecialTokens'):
    """When multiple mm items of the same modality present, HF processors
    return them in a bundled format (e.g., all images are combined into a
    single tensor).

    This function expands such bundled HF processor outputs into per-image / video entries for better cache locality.
    """

    expanded_mm_items = []
    for modality, item in collected_mm_items.items():
        token_id = mm_tokens.get_token_id_by_modality(modality)
        is_bundled = item.get('offset', None) is not None and len(item['offset']) > 1

        # non-bundled case
        if not is_bundled:
            if modality == Modality.IMAGE:
                expanded_mm_items.append(
                    dict(
                        modality=modality,
                        pixel_values=item['feature'],
                        image_grid_thw=item['image_grid_thw'][0],
                        offset=item['offset'][0],
                        image_token_id=token_id,
                    ))
            elif modality == Modality.VIDEO:
                second_per_grid = item.get('video_second_per_grid')
                if second_per_grid is not None:
                    second_per_grid = second_per_grid[0].item()
                expanded_mm_items.append(
                    dict(
                        modality=modality,
                        pixel_values_videos=item['feature'],
                        video_grid_thw=item['video_grid_thw'][0],
                        offset=item['offset'][0],
                        second_per_grid=second_per_grid,
                        video_token_id=token_id,
                    ))
            elif modality == Modality.AUDIO:
                expanded_mm_items.append(
                    dict(
                        modality=modality,
                        input_features=item['feature'],
                        feature_attention_mask=item.get('feature_attention_mask'),
                        offset=item['offset'][0],
                        audio_token_id=token_id,
                    ))
            elif modality == Modality.TIME_SERIES:
                expanded_mm_items.append(
                    dict(
                        modality=modality,
                        ts_values=item['feature'],
                        ts_sr=item['ts_sr'],
                        ts_lens=item['ts_lens'],
                        offset=item['offset'][0],
                        ts_token_id=token_id,
                    ))
            continue

        # bundled case, expand into per-item entries
        if modality == Modality.IMAGE:
            expanded_mm_items.extend(_expand_bundled_image_items(item, token_id))
        elif modality == Modality.VIDEO:
            expanded_mm_items.extend(_expand_bundled_video_items(item, token_id))
        elif modality == Modality.AUDIO:
            expanded_mm_items.extend(_expand_bundled_audio_items(item, token_id))

    # HF processors return features grouped by modality; offsets restore prompt order for mixed inputs.
    expanded_mm_items.sort(key=lambda item: item['offset'][0])
    return expanded_mm_items
