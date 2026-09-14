# Copyright (c) OpenMMLab. All rights reserved.
# adapted from https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/video.py
# adapted from https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py

import math
import os
import tempfile
from abc import abstractmethod
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


def glm_sample_frame_indices(
    total_frames: int,
    source_fps: float,
    duration: float,
    *,
    target_fps: float | None = None,
    max_frame_count: int | None = None,
) -> list[int]:
    """Sample the deterministic temporal pairs expected by GLM video models.

    GLM constructs one visual unit from every two sampled source frames.  Its
    serving contract therefore differs from the generic uniform sampler in two
    ways: the default is 2 FPS with at most 2048 frames, and an odd result
    repeats its final frame to keep temporal pairs complete.
    """
    if total_frames <= 0:
        return []
    target_fps = 2.0 if target_fps is None else float(target_fps)
    max_frame_count = (2048 if max_frame_count is None else
                       int(max_frame_count))
    if target_fps <= 0 or max_frame_count <= 0:
        return []

    max_frame_idx = total_frames - 1
    if not duration:
        duration = (round(max_frame_idx / source_fps) + 1
                    if source_fps else 0)
    extract_t = min(int(duration * target_fps), max_frame_count)
    extract_t = max(1, extract_t)

    if source_fps:
        duration_per_frame = 1 / source_fps
        max_second = int(duration)
        indices = []
        current_second = 0.0
        interval = 1 / target_fps
        for frame_index in range(total_frames):
            timestamp = frame_index * duration_per_frame
            if timestamp >= current_second:
                current_second += interval
                indices.append(frame_index)
                if current_second >= max_second:
                    break
    else:
        indices = []

    if len(indices) < extract_t:
        start = indices[0] if indices else 0
        end = indices[-1] if indices else max(total_frames - 1, 0)
        indices = np.linspace(start, end, extract_t, dtype=int).tolist()
    elif len(indices) > extract_t:
        indices = np.linspace(0,
                              total_frames - 1,
                              extract_t,
                              dtype=int).tolist()

    # np.linspace can repeat indices for very small inputs.  GLM first removes
    # those repeats, then pads an odd number of frames with the final sample.
    unique_indices = list(dict.fromkeys(int(index) for index in indices))
    if len(unique_indices) & 1:
        unique_indices.append(unique_indices[-1])
    return unique_indices


class VideoLoader:

    @classmethod
    @abstractmethod
    def load_bytes(self, data: bytes, num_frames: int = -1, **kwargs) -> tuple[npt.NDArray, dict[str, Any]]:
        raise NotImplementedError

    @classmethod
    def smart_nframes(self,
                      total_frames_num: int,
                      num_frames: int,
                      fps: float,
                      duration: float,
                      sampling_strategy: str = 'uniform',
                      source_fps: float | None = None) -> tuple[int, list[int]]:
        # resample video to target num_frames and fps
        # - the minimum of the two will be used
        if sampling_strategy == 'glm':
            frame_idx = glm_sample_frame_indices(
                total_frames_num,
                source_fps=source_fps or 0,
                duration=duration,
                target_fps=None if fps <= 0 else fps,
                max_frame_count=None if num_frames <= 0 else num_frames,
            )
            return len(frame_idx), frame_idx
        if sampling_strategy != 'uniform':
            raise ValueError(
                f'Unknown video sampling strategy: {sampling_strategy!r}')
        num_frames_to_sample = total_frames_num
        if num_frames > 0:
            num_frames_to_sample = min(num_frames, total_frames_num)
        if fps > 0:
            num_frames_to_sample = min(num_frames_to_sample, math.floor(duration * fps))
        num_frames_to_sample = max(1, num_frames_to_sample)  # at least one sample

        if num_frames_to_sample == total_frames_num:
            frame_idx = list(range(0, num_frames_to_sample))
        else:
            uniform_sampled_frames = np.linspace(0, total_frames_num - 1, num_frames_to_sample, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
        return num_frames_to_sample, frame_idx


class OpenCVVideoLoader(VideoLoader):

    def get_cv2_video_api(self):
        import cv2.videoio_registry as vr

        api_pref = None
        for backend in vr.getStreamBufferedBackends():
            if not vr.hasBackend(backend):
                continue
            if not vr.isBackendBuiltIn(backend):
                _, abi, api = vr.getStreamBufferedBackendPluginVersion(backend)
                if abi < 1 or (abi == 1 and api < 2):
                    continue
            api_pref = backend
            break
        return api_pref

    @staticmethod
    def _read_frames(
        cap,
        frame_indices: set[int],
        num_expected_frames: int,
        max_frame_idx: int,
    ) -> tuple[npt.NDArray, int, list[int]]:
        import cv2

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = np.empty((num_expected_frames, height, width, 3), dtype=np.uint8)  # THWC

        i = 0
        valid_frame_indices = []
        for idx in range(max_frame_idx + 1):
            ok = cap.grab()
            if not ok:
                # Frame is broken/unreadable, log warning
                if idx in frame_indices:
                    logger.warning(
                        'Failed to grab frame %d during video loading. '
                        'This frame will be skipped.',
                        idx,
                    )
                continue
            if idx in frame_indices:
                ret, frame = cap.retrieve()
                if ret:
                    frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    valid_frame_indices.append(idx)
                    i += 1
                else:
                    # retrieve() failed even though grab() succeeded
                    logger.warning(
                        'Failed to retrieve frame %d during video loading. '
                        'This frame will be skipped.',
                        idx,
                    )

        valid_num_frames = len(valid_frame_indices)
        if valid_num_frames < num_expected_frames:
            logger.warning(
                'Video loading completed with %d broken/unreadable frames. '
                'Expected %d frames but only loaded %d frames.',
                num_expected_frames - valid_num_frames,
                num_expected_frames,
                valid_num_frames,
            )

        return frames[:valid_num_frames], valid_num_frames, valid_frame_indices

    @classmethod
    def load_file(
        self,
        filepath: Path,
        num_frames: int = -1,
        fps: float = -1,
        max_duration: int = 300,
        sampling_strategy: str = 'uniform',
        **kwargs,
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        with open(filepath, 'rb') as f:
            data = f.read()
        return self.load_bytes(data,
                               num_frames=num_frames,
                               fps=fps,
                               max_duration=max_duration,
                               sampling_strategy=sampling_strategy,
                               **kwargs)

    @classmethod
    def load_bytes(
        cls,
        data: bytes,
        num_frames: int = -1,
        fps: float = -1,
        max_duration: int = 300,
        sampling_strategy: str = 'uniform',
        **kwargs,
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        """Load video frames from bytes.

        Args:
            data: Raw video bytes
            num_frames: Target number of frames to sample (-1 for all)
            fps: Target FPS for sampling (-1 for original)
            max_duration: Maximum duration (unused in base backend)

        Returns:
            Tuple of (frames_array, metadata_dict)
        """
        import cv2

        backend = cls().get_cv2_video_api()
        cap = cv2.VideoCapture(BytesIO(data), backend, [])
        if not cap.isOpened():
            raise ValueError('Could not open video stream')

        total_frames_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames_num / original_fps if original_fps > 0 else 0

        _, frame_idx = cls.smart_nframes(
            total_frames_num,
            num_frames,
            fps,
            duration,
            sampling_strategy=sampling_strategy,
            source_fps=original_fps,
        )
        if not frame_idx:
            raise ValueError('Video sampling produced no frame indices.')

        unique_frame_indices = list(dict.fromkeys(frame_idx))
        frame_idx_set = set(unique_frame_indices)
        frames, _, valid_frame_indices = cls._read_frames(
            cap,
            frame_idx_set,
            len(unique_frame_indices),
            max(frame_idx),
        )
        # The GLM sampler may repeat the final frame to complete a temporal
        # pair.  OpenCV decodes each source index once, so restore the requested
        # order (including repeats) after decoding.
        decoded = dict(zip(valid_frame_indices, frames))
        ordered_indices = [index for index in frame_idx if index in decoded]
        if ordered_indices:
            frames = np.stack([decoded[index] for index in ordered_indices])
        else:
            frames = frames[:0]
        valid_frame_indices = ordered_indices

        # Use transformers transformers.video_utils.VideoMetadata format
        # For models like Qwen3-VL/GLM4.5V, this metadata
        # can cause incorrect timestamp calculation without num_frames=-1.
        # TODO: zhouxinyu, support per-request do_sample_frames
        metadata = {
            'total_num_frames': total_frames_num,
            'fps': original_fps,
            'duration': duration,
            'video_backend': 'opencv',
            'frames_indices': valid_frame_indices,
            # extra field used to control hf processor's video
            # sampling behavior
            # "do_sample_frames": valid_num_frames == total_frames_num,
        }
        return frames, metadata


class DecordVideoLoader(VideoLoader):

    @classmethod
    def load_file(self,
                  filepath: Path,
                  num_frames: int = -1,
                  fps: float = -1,
                  max_duration: int = 300,
                  sampling_strategy: str = 'uniform',
                  **kwargs) -> tuple[npt.NDArray, dict[str, Any]]:
        import decord
        vr = decord.VideoReader(str(filepath))
        total_frames_num = len(vr)
        original_fps = vr.get_avg_fps()
        duration = total_frames_num / original_fps if original_fps > 0 else 0

        _, frame_idx = self.smart_nframes(
            total_frames_num,
            num_frames,
            fps,
            duration,
            sampling_strategy=sampling_strategy,
            source_fps=original_fps,
        )
        if not frame_idx:
            raise ValueError('Video sampling produced no frame indices.')

        video = vr.get_batch(frame_idx).asnumpy()  # THWC
        metadata = {
            'total_num_frames': total_frames_num,
            'fps': original_fps,
            'duration': duration,
            'video_backend': 'decord',
            'frames_indices': frame_idx,
        }
        return video, metadata

    @classmethod
    def load_bytes(self,
                   data: bytes,
                   num_frames: int = -1,
                   fps: float = -1,
                   max_duration: int = 300,
                   sampling_strategy: str = 'uniform',
                   **kwargs) -> tuple[npt.NDArray, dict[str, Any]]:
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        try:
            tmp_file.write(data)
            tmp_file.close()
            return self.load_file(Path(tmp_file.name),
                                  num_frames=num_frames,
                                  fps=fps,
                                  max_duration=max_duration,
                                  sampling_strategy=sampling_strategy,
                                  **kwargs)
        finally:
            # always cleanup, even if load_file crashes
            try:
                os.unlink(tmp_file.name)
            except OSError:
                pass  # file might not exist if write failed


class TorchCodecVideoLoader(VideoLoader):

    @classmethod
    def load_file(self,
                  filepath: Path,
                  num_frames: int = -1,
                  fps: float = -1,
                  max_duration: int = 300,
                  sampling_strategy: str = 'uniform',
                  **kwargs) -> tuple[npt.NDArray, dict[str, Any]]:
        # torchcodec requires matched ffmpeg, torchcodec, and torch versions
        # ffmpeg 5.1.2, torch 2.8.0, torchcodec 0.7.0 are verified to work together
        from torchcodec.decoders import VideoDecoder

        torch_codec_num_threads = 8
        decoder = VideoDecoder(str(filepath), num_ffmpeg_threads=torch_codec_num_threads)
        total_frames_num = decoder.metadata.num_frames
        original_fps = decoder.metadata.average_fps
        duration = total_frames_num / original_fps if original_fps > 0 else 0

        _, frame_idx = self.smart_nframes(
            total_frames_num,
            num_frames,
            fps,
            duration,
            sampling_strategy=sampling_strategy,
            source_fps=original_fps,
        )
        if not frame_idx:
            raise ValueError('Video sampling produced no frame indices.')

        video = decoder.get_frames_at(frame_idx).data
        metadata = {
            'total_num_frames': total_frames_num,
            'fps': original_fps,
            'duration': duration,
            'video_backend': 'torchcodec',
            'frames_indices': frame_idx,
        }
        return video, metadata

    @classmethod
    def load_bytes(self,
                   data: bytes,
                   num_frames: int = -1,
                   fps: float = -1,
                   max_duration: int = 300,
                   sampling_strategy: str = 'uniform',
                   **kwargs) -> tuple[npt.NDArray, dict[str, Any]]:
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        try:
            tmp_file.write(data)
            tmp_file.close()
            return self.load_file(Path(tmp_file.name),
                                  num_frames=num_frames,
                                  fps=fps,
                                  max_duration=max_duration,
                                  sampling_strategy=sampling_strategy,
                                  **kwargs)
        finally:
            # always cleanup, even if load_file crashes
            try:
                os.unlink(tmp_file.name)
            except OSError:
                pass  # file might not exist if write failed


class TorchVisionVideoLoader(VideoLoader):

    @classmethod
    def load_file(self,
                  filepath: Path,
                  num_frames: int = -1,
                  fps: float = -1,
                  max_duration: int = 300,
                  sampling_strategy: str = 'uniform',
                  **kwargs) -> tuple[npt.NDArray, dict[str, Any]]:
        import torchvision

        video, audio, info = torchvision.io.read_video(
            filepath,
            pts_unit='sec',
            output_format='THWC',
        )
        total_frames_num = video.size(0)
        original_fps = info['video_fps']
        duration = total_frames_num / original_fps if original_fps > 0 else 0

        _, frame_idx = self.smart_nframes(
            total_frames_num,
            num_frames,
            fps,
            duration,
            sampling_strategy=sampling_strategy,
            source_fps=original_fps,
        )
        if not frame_idx:
            raise ValueError('Video sampling produced no frame indices.')

        video = video[frame_idx]
        metadata = {
            'total_num_frames': total_frames_num,
            'fps': original_fps,
            'duration': duration,
            'video_backend': 'torchvision',
            'frames_indices': frame_idx,
        }
        return video, metadata

    @classmethod
    def load_bytes(self,
                   data: bytes,
                   num_frames: int = -1,
                   fps: float = -1,
                   max_duration: int = 300,
                   sampling_strategy: str = 'uniform',
                   **kwargs) -> tuple[npt.NDArray, dict[str, Any]]:
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        try:
            tmp_file.write(data)
            tmp_file.close()
            return self.load_file(Path(tmp_file.name),
                                  num_frames=num_frames,
                                  fps=fps,
                                  max_duration=max_duration,
                                  sampling_strategy=sampling_strategy,
                                  **kwargs)
        finally:
            # always cleanup, even if load_file crashes
            try:
                os.unlink(tmp_file.name)
            except OSError:
                pass  # file might not exist if write failed
