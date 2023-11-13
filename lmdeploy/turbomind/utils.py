# Copyright (c) OpenMMLab. All rights reserved.
import dataclasses
from pathlib import Path
from typing import Optional, Union

from huggingface_hub import snapshot_download


@dataclasses.dataclass
class GenParam:
    top_p: float
    top_k: float
    temperature: float
    repetition_penalty: float
    sequence_start: bool = False
    sequence_end: bool = False
    step: int = 0
    request_output_len: int = 512


def get_gen_param(cap,
                  sampling_param,
                  nth_round,
                  step,
                  request_output_len=512,
                  **kwargs):
    """return parameters used by token generation."""
    gen_param = GenParam(**dataclasses.asdict(sampling_param),
                         request_output_len=request_output_len)
    # Fix me later. turbomind.py doesn't support None top_k
    if gen_param.top_k is None:
        gen_param.top_k = 40

    if cap == 'chat':
        gen_param.sequence_start = (nth_round == 1)
        gen_param.sequence_end = False
        gen_param.step = step
    else:
        gen_param.sequence_start = True
        gen_param.sequence_end = True
        gen_param.step = 0
    return gen_param


def download_hf_repo(
    repo_id: str,
    *,
    revision: Optional[str] = None,
    cache_dir: Union[str, Path, None] = None,
    proxies: Optional[dict] = None,
    resume_download: bool = True,
    force_download: bool = False,
    token: Optional[Union[bool, str]] = None,
    local_files_only: bool = False,
) -> str:
    """Download huggingface repo."""

    download_kwargs = {
        'revision': revision,
        'cache_dir': cache_dir,
        'proxies': proxies,
        'resume_download': resume_download,
        'force_download': force_download,
        'token': token,
        'local_files_only': local_files_only
    }

    downloaded_folder = snapshot_download(repo_id, **download_kwargs)
    return downloaded_folder
