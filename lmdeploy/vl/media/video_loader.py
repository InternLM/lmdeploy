# Copyright (c) OpenMMLab. All rights reserved.
# adapted from https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/video.py
# adapted from https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py

import math
import tempfile
from abc import abstractmethod
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


class VideoLoader:

    @classmethod
    @abstractmethod
    def load_bytes(self, data: bytes, num_frames: int = -1, **kwargs) -> tuple[npt.NDArray, dict[str, Any]]:
        raise NotImplementedError

    @classmethod
    def smart_nframes(self, total_frames_num: int, num_frames: int, fps: int, duration: int) -> list[int]:
        # resample video to target num_frames and fps
        # - the minimum of the two will be used
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
        frames = np.empty((num_expected_frames, height, width, 3), dtype=np.uint8)

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
        fps: int = -1,
        max_duration: int = 300,
        **kwargs,
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        with open(filepath, 'rb') as f:
            data = f.read()
        return self.load_bytes(data, num_frames=num_frames, fps=fps, max_duration=max_duration, **kwargs)

    @classmethod
    def load_bytes(
        cls,
        data: bytes,
        num_frames: int = -1,
        fps: int = -1,
        max_duration: int = 300,
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

        num_frames_to_sample, frame_idx = cls.smart_nframes(total_frames_num, num_frames, original_fps, duration)

        frame_idx_set = set(frame_idx)
        frames, valid_num_frames, valid_frame_indices = cls._read_frames(cap, frame_idx_set, num_frames_to_sample,
                                                                         max(frame_idx))

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

        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # TCHW
        return frames, metadata


class DecordVideoLoader(VideoLoader):

    @classmethod
    def load_file(self, filepath: Path, num_frames: int = -1, **kwargs) -> tuple[npt.NDArray, dict[str, Any]]:
        import decord
        vr = decord.VideoReader(str(filepath))
        total_frames_num = len(vr)
        fps = vr.get_avg_fps()
        duration = total_frames_num / fps if fps > 0 else 0

        num_frames_to_sample, frame_idx = self.smart_nframes(total_frames_num, num_frames, fps, duration)

        video = vr.get_batch(frame_idx).asnumpy()  # TWHC
        video = np.transpose(video, (0, 3, 1, 2))  # TCHW
        metadata = {
            'total_num_frames': total_frames_num,
            'fps': fps,
            'duration': duration,
            'video_backend': 'decord',
            'frames_indices': frame_idx,
        }
        return video, metadata

    @classmethod
    def load_bytes(self, data: bytes, num_frames: int = -1, **kwargs) -> tuple[npt.NDArray, dict[str, Any]]:
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tmp_file.write(data)
        tmp_file.close()

        return self.load_file(Path(tmp_file.name), num_frames=num_frames, **kwargs)


class TorchVisionVideoLoader(VideoLoader):

    @classmethod
    def load_file(self, filepath: Path, num_frames: int = -1, **kwargs) -> tuple[npt.NDArray, dict[str, Any]]:
        import torchvision

        video, audio, info = torchvision.io.read_video(
            filepath,
            pts_unit='sec',
            output_format='TCHW',
        )
        total_frames_num = video.size(0)
        fps = info['video_fps']
        duration = total_frames_num / fps if fps > 0 else 0

        num_frames_to_sample, frame_idx = self.smart_nframes(total_frames_num, num_frames, fps, duration)

        video = video[frame_idx]
        metadata = {
            'total_num_frames': total_frames_num,
            'fps': fps,
            'duration': duration,
            'video_backend': 'torchvision',
            'frames_indices': frame_idx,
        }
        return video, metadata

    @classmethod
    def load_bytes(self, data: bytes, num_frames: int = -1, **kwargs) -> tuple[npt.NDArray, dict[str, Any]]:
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tmp_file.write(data)
        tmp_file.close()

        self.load_file(Path(tmp_file.name), num_frames=num_frames, **kwargs)


class TorchCodecVideoLoader(VideoLoader):

    @classmethod
    def load_file(self, filepath: Path, num_frames: int = -1, **kwargs) -> tuple[npt.NDArray, dict[str, Any]]:
        # torchcodec requires matched ffmpeg, torchcodec, and torch versions
        # ffmpeg 5.1.2, torch 2.8.0, torchcodec 0.7.0 are verified to work together
        from torchcodec.decoders import VideoDecoder

        torch_codec_num_threads = 8
        decoder = VideoDecoder(str(filepath), num_ffmpeg_threads=torch_codec_num_threads)
        total_frames_num = decoder.metadata.num_frames
        fps = decoder.metadata.average_fps
        duration = total_frames_num / fps if fps > 0 else 0

        num_frames_to_sample, frame_idx = self.smart_nframes(total_frames_num, num_frames, fps, duration)

        video = decoder.get_frames_at(frame_idx).data
        import pdb
        pdb.set_trace()
        metadata = {
            'total_num_frames': total_frames_num,
            'fps': fps,
            'duration': duration,
            'video_backend': 'torchcodec',
            'frames_indices': frame_idx,
        }
        return video, metadata

    @classmethod
    def load_bytes(self, data: bytes, num_frames: int = -1, **kwargs) -> tuple[npt.NDArray, dict[str, Any]]:
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tmp_file.write(data)
        tmp_file.close()

        return self.load_file(Path(tmp_file.name), num_frames=num_frames, **kwargs)
