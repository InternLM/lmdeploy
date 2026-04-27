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
        returns [(2, 6), (6, 9)]
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


# adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/mm_utils.py
def get_expanded_mm_items(collected_mm_items, mm_tokens: 'MultimodalSpecialTokens'):
    """Expand bundled hf processor outputs into per-image/video entries for
    cache locality and scheduling."""
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

        # bundled case
        num_items = len(item['offset'])
        if modality == Modality.IMAGE:
            image_grid_thw = item['image_grid_thw']

            patches_per_item = []
            for grid in image_grid_thw:
                grid_tensor = torch.as_tensor(grid, dtype=torch.long)
                patches_per_item.append(int(torch.prod(grid_tensor).item()))

            cumulative = torch.cumsum(torch.tensor(patches_per_item, dtype=torch.long), dim=0)
            slice_indices = [0] + cumulative.tolist()

            for i in range(num_items):
                start_idx, end_idx = slice_indices[i], slice_indices[i + 1]
                # TODO: zhouxinyu, compute mask and avoid passing token id
                expanded_mm_items.append(
                    dict(
                        modality=modality,
                        pixel_values=item['feature'][start_idx:end_idx],
                        image_grid_thw=image_grid_thw[i],
                        offset=item['offset'][i],
                        image_token_id=token_id,
                    ))
        elif modality == Modality.VIDEO:
            video_grid_thw = item['video_grid_thw']
            num_videos = video_grid_thw.shape[0]

            frames_per_video = []
            total_frames = 0
            for i in range(num_videos):
                grid = video_grid_thw[i]
                T = int(grid[0].item()) if isinstance(grid, torch.Tensor) else int(
                    torch.as_tensor(grid, dtype=torch.long)[0].item())
                frames_per_video.append(T)
                total_frames += T

            if num_items != total_frames:
                expanded_mm_items.append(item)
                continue

            patches_per_video = []
            for i in range(num_videos):
                grid = video_grid_thw[i]
                patches_per_video.append(
                    int(torch.prod(grid).item()) if isinstance(grid, torch.Tensor) else int(
                        torch.prod(torch.as_tensor(grid, dtype=torch.long)).item()))

            cumulative = torch.cumsum(torch.tensor(patches_per_video, dtype=torch.long), dim=0)
            slice_indices = [0] + cumulative.tolist()

            frame_start_indices = [0]
            for i in range(num_videos):
                frame_start_indices.append(frame_start_indices[-1] + frames_per_video[i])

            for video_idx in range(num_videos):
                start, end = slice_indices[video_idx], slice_indices[video_idx + 1]
                frame_start, frame_end = frame_start_indices[video_idx], frame_start_indices[video_idx + 1]

                # TODO: zhouxinyu, not sure per-frame split is good or not
                # TODO: zhouxinyu, grid_thw [1, h, w] is only for qwen3vl
                t, h, w = video_grid_thw[video_idx].tolist()
                for frame_idx in range(t):
                    video_feature = item['feature'][start:end]
                    expanded_mm_items.append(
                        dict(
                            modality=modality,
                            pixel_values_videos=video_feature[frame_idx * h * w:(frame_idx + 1) * h * w],
                            video_grid_thw=torch.tensor([1, h, w]),
                            offset=item['offset'][frame_start:frame_end][frame_idx],
                            video_token_id=token_id,
                        ))

    return expanded_mm_items
